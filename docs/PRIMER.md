# Primer: the concepts behind this project

This primer is for readers who know standard machine learning but not spiking
networks or event cameras. It explains what the experiment compares, the
vocabulary used in the code and the decision logs, and where each number comes
from.

Related documents:
- [`DECISIONS.md`](DECISIONS.md) — what was decided, and why.
- [`IMPLEMENTATION_DECISIONS.md`](IMPLEMENTATION_DECISIONS.md) — every smaller
  choice made during implementation.
- [`COLAB_PILOT_WALKTHROUGH.md`](COLAB_PILOT_WALKTHROUGH.md) — how to run the
  pilot experiment.

---

## 1. The experiment in one picture

```
                 ┌──────────── Method B: HYBRID ────────────────────────────┐
 N-MNIST         │ train ANN ──► convert ANN → SNN ──► gate ──► fine-tune SNN │──► SNN_B
 recordings ─────┤                                                            │
 (same data,     └────────────────────────────────────────────────────────────┘
  same split)    ┌──────────── Method A: DIRECT SG ─────────────────────────┐
                 │ initialise SNN ──► train SNN with surrogate gradients     │──► SNN_A
                 └────────────────────────────────────────────────────────────┘

 Then, for each T in {3, 6, 12, 24, 48}: test accuracy of SNN_A and SNN_B
 Outcome: the smallest T at which each SNN is within 2 pp of the ANN's accuracy.
 Also reported: how much training compute each method used.
```

**The study does not compare an ANN with an SNN.** It compares **two ways of
producing the same SNN**:
1. Starting from a trained ANN (hybrid).
2. Training the SNN directly from random weights (direct SG).

The ANN plays two supporting roles:
- It is the first stage of the hybrid method.
- Its test accuracy defines the target that both SNNs must reach.

The question is which method yields an SNN that reaches the target with **fewer
simulation time-steps**, i.e. lower inference latency. Fewer steps also means
less computation per prediction.

## 2. Vocabulary

| Term | Meaning in this project |
|---|---|
| **ANN** | Ordinary neural network. Neurons output real numbers (here via ReLU). It processes an input in one pass. |
| **SNN** | Spiking neural network. Neurons hold an internal state (membrane potential) and output binary **spikes** (0 or 1). It runs for **T time-steps**, and the prediction comes from activity accumulated over those steps. |
| **Time-step, T** | One update of every neuron in the SNN. A larger T gives more time to integrate evidence, but means more latency and more operations per prediction. "Inference T" is the number of steps used at test time. |
| **LIF neuron** | Leaky integrate-and-fire neuron. Each step it adds its input to its membrane potential and leaks part of it (τ = 2 means half). When the potential crosses the threshold θ = 1, it emits a spike and subtracts θ ("soft reset"). |
| **Firing rate** | The fraction of time-steps in which a neuron spikes (between 0 and 1). Conversion relies on "SNN firing rate ≈ ANN activation". |
| **Surrogate gradient (SG)** | A spike is a step function with zero gradient almost everywhere. During backpropagation, SG training replaces its derivative with a smooth stand-in (here arctan-shaped), which makes gradient descent possible. Gradients flow backwards through all T steps (backpropagation through time), so training cost grows with T. |
| **ANN-to-SNN conversion** | Copy the trained ANN's weights into the SNN and rescale them so that the spiking neurons' firing rates reproduce the ANN's activations. No gradient training is involved. |
| **Fine-tuning** | A short SG training run that starts from the converted weights instead of random ones. |
| **Arm** | Experimental-design term for one condition being compared, as in a clinical trial. This study has two arms: **Method A, direct SG** and **Method B, hybrid**. |
| **Seed** | The random seed that fixes initialisation and data order. Each seed gives one paired observation: one run of each arm. Several seeds measure run-to-run variability. |
| **pp (percentage points)** | The absolute difference between two percentages. If the ANN reaches 91.3 %, then "within 2 pp" means ≥ 89.3 %. A relative 2 % would instead mean ≥ 89.47 %. The project always means percentage points. |
| **Validation vs test** | Validation (12 000 recordings carved from the training split) is used for every choice made during development: early stopping, the conversion gate, the init-rate sweep. Test (the official 10 000) is used only for the final accuracies and the 2-pp threshold. |

## 3. The data: event cameras, saccades and N-MNIST

- **Event camera (DVS / ATIS).** It has no frames. Each pixel independently
  reports an **event** whenever its brightness changes enough. An event is
  (x, y, t, p): pixel position, timestamp in µs, and polarity p. p = 1 is an
  **ON** event (brightness went up); p = 0 is an **OFF** event (it went down).
  A static scene produces no events.
- **Saccade.** A quick, small eye movement. Because a still camera in front of a
  still image sees nothing, N-MNIST was recorded by moving the sensor in **three
  quick motions of about 105 ms each**, mimicking eye saccades, in front of a
  monitor showing each MNIST digit. The digit's edges sweep across pixels, which
  produces events. The movement goes along different directions in the three
  saccades and pauses at each change of direction. That is why the event rate in
  `report/figures/fig1_event_rate_and_bins.pdf` has three bursts with quiet gaps.
- **N-MNIST.** The 60 000 / 10 000 MNIST digits re-recorded this way at
  34 × 34 pixels, giving about 4 000 events per recording over about 0.3 s.

Read more:
- Original paper: G. Orchard et al., "Converting Static Image Datasets to Spiking
  Neuromorphic Datasets Using Saccades", *Frontiers in Neuroscience* 9:437 (2015),
  https://doi.org/10.3389/fnins.2015.00437 (open access; Sections 2–3 describe the
  saccades, Fig. 2 shows the recording set-up).
- Dataset page and file format: https://www.garrickorchard.com/datasets/n-mnist
- Tonic, the loader used here: https://tonic.readthedocs.io (N-MNIST under "Datasets").
- SpikingJelly, the SNN framework: https://spikingjelly.readthedocs.io (its
  neuromorphic-datasets tutorial shows events being integrated into frames).

## 4. From events to network inputs

- **The SNN needs one input per time-step.** Events are grouped into **bins**: the
  0–315 ms window is cut into T equal slices, and each slice counts how many ON and
  OFF events hit each pixel. The result is T "frames" `[T, 2, 34, 34]`, presented to
  the SNN one per step, in time order. Timing information survives at the resolution
  of the bins.
- **The ANN has no time axis.** It receives the *average* of the bins, `[2, 34, 34]`.
  Timing is lost there, intentionally.
- **v1.0 got this wrong.** It gave the SNN the same averaged frame at every step,
  which threw away all timing. That error is why the protocol was revised (D-0).

## 5. Why evaluating at different T is subtle

The outcome is "minimum T that reaches the target", so we must test each SNN at
T = 3, 6, 12, 24 and 48. Two questions arise.

**(a) How is a T-step input made from a recording?** The field's standard tooling,
SpikingJelly's N-MNIST loader with `frames_number=T, split_by='time'`, splits the
*whole recording* into T equal-time frames. We do the same over a fixed 315-ms window
("re-binning"). As T changes, so does the bin width: T = 3 → 105 ms per bin,
T = 48 → 6.6 ms per bin.

**(b) Is the SNN trained at that same T?** Standard practice, and the setup in
Rathi et al., is **yes: one model per T**. The original plan instead trains once
(T_train = 24) and evaluates that one model at every T. The two options:

| Protocol | What happens at T = 3 | Pros | Cons |
|---|---|---|---|
| **P1 (current default):** train at 24, evaluate at all T | A model trained on 13-ms bins sees three 105-ms bins | 1 training run per arm and seed; matches the plan | Accuracy at T ≠ 24 mixes "fewer steps" with "inputs unlike training". This can make low T look worse than it really is. |
| **P2 (textbook):** train a separate model at each T | The model is trained on 105-ms bins | Measures what each T can really achieve; no train/test mismatch | About 3.9× the SNN training cost (24 vs 3+6+12+24+48 = 93 steps per seed). The hybrid arm reuses one ANN per seed. |

Both are now runnable:
- `run_pilot.py` = P1.
- `run_pilot.py --t-train T` for each T = P2.

For P1, `run_pilot.py` also reports two **sensitivity variants**, which separate the
effects:
- `rebin_raw`: same bins without the T/24 scaling. This tests whether the scaling
  matters.
- `prefix`: keep 13-ms bins and only show the first T of them. This tests whether
  the effect is about resolution or about how much of the recording is seen.

If the three curves disagree, the choice of protocol matters, and P2 is the safer
primary analysis. **The pilot is designed to answer this with data** (see the
walkthrough).

**Why T/24 scaling in P1?** A 105-ms bin holds about 8× more events than a 13-ms
bin. Without scaling, a model trained on 13-ms bins would receive inputs 8× too
large at T = 3. Scaling by T/24 keeps the average input per step equal to training.
P2 does not need this.

## 6. Why the conversion needed new maths

- **Classic conversion** (Rueckauer et al. 2017) assumes neurons *without* leak,
  whose firing rate is proportional to their input.
- **Our LIF neurons leak:** with τ = 2 they halve their membrane every step. They
  stay silent unless the input exceeds the threshold, and above it the rate rises
  in a steep, curved way (`report/figures/fig3_lif_transfer.pdf`).
- Copying weights with the classic method therefore gives chance accuracy (10 %).
- **The fix maps inputs onto the LIF's real rate curve.** It is fitted from a
  simulation of the neuron, then refined per layer on real training data, because
  real event input is bursty rather than constant.

## 7. The conversion gate

The gate is a quality check on **one step** of the hybrid method: after converting
and before fine-tuning, the converted SNN is compared with the ANN on validation
data.

- **Strict check**, always recorded: within 2 pp of the ANN, and the same
  prediction on ≥ 90 % of samples. It answers "is conversion near-lossless?".
- **Sanity check**, used if the strict check fails: the converted SNN keeps at least
  half of the ANN's accuracy above chance. It answers "did conversion transfer real
  knowledge?". A failed conversion (10 % or 23 %) is rejected; a lossy-but-useful
  one (74 %) proceeds to fine-tuning.

The **study's** 2-pp target is a different threshold. It is applied to the
*fine-tuned* SNN on *test* data and is unchanged. Hybrid training is defined as
"lossy conversion, then fine-tuning recovers the loss" (Rathi et al. 2020), so
demanding lossless conversion would test a stronger claim than the method makes.

## 8. Why the direct-SG network needed a special initialisation

- **The problem:** with PyTorch's default random weights and our sparse event
  inputs (about 0.1 events per pixel per bin), the deeper LIF layers never reach
  the threshold. They emit zero spikes, so the layers after them receive no
  gradient, and training barely starts (13 % validation accuracy after 63 updates).
- **The fix:** before training, each layer's weights are multiplied by one number,
  chosen so that the layer fires at a target rate on a few training batches. This is
  the SNN analogue of data-driven ANN initialisations such as LSUV.
- **How the target rate is chosen:** it is a hyperparameter, so it was **selected on
  validation data** by a pre-specified sweep over {1, 2, 5, 10, 20} %. 5 % scored
  best (74.1 % mean validation accuracy), and anything from 1 % to 10 % is within
  about 2 pp, so the exact value matters little. Details are in
  `IMPLEMENTATION_DECISIONS.md` 4.2 and
  `results/conversion_init_check/report_F_sweep.json`.

## 9. The training budget: counted in time-steps

- **One unit** = one training sample passed through the network (forward + backward)
  for one simulated time-step.
- An ANN sample costs 1 unit; an SNN sample at T_train = 24 costs 24.
- This shows what epoch counts hide: the hybrid method's ANN stage is cheap, and its
  SNN fine-tuning is expensive.
- **No cap (team decision):** both arms train to early stopping, and the budget each
  actually used is reported next to its minimum T.

## 10. Where each number comes from

Every number in the configuration falls into one of four kinds. The report should
say which kind each one is.

| Kind | Meaning | Examples |
|---|---|---|
| **Measured from data** | Read off the dataset or a simulation | 315-ms window (27 of 250 M events fall after it); LIF map c = 0.83, k = 1.36 (fitted to a simulated neuron); per-layer λ (99th percentile of activations) |
| **Field standard / literature** | Taken from established practice | Splitting the whole recording into T frames (SpikingJelly); layer-wise normalisation (Rueckauer 2017); λ at the 99th percentile; Adam, cosine schedule; ON/OFF polarity bit (dataset spec) |
| **Selected by a procedure** | A pre-specified candidate set and rule, decided on validation data | init target rate (sweep); early-stopping epoch; per-layer (c, k) refinement |
| **Policy / design decision** | A threshold that defines the question. It *cannot* be optimised, because optimising it would make the result circular; it is fixed before the experiment and justified by its meaning | 2-pp accuracy margin; gate thresholds (2 pp, 90 %, half of above-chance accuracy); T grid {3, …, 48}; T_train = 24 |
