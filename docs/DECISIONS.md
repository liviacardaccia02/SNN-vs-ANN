# Methodological decision log

Project: *Time-Step Efficiency of Hybrid Conversion-and-Fine-Tuning versus
Direct Surrogate-Gradient Training for SNNs on Event-Based Data* (KTH II2202).
Protocol version after these decisions: **v2.1-draft** (`src/config.py`).

Each entry: decision · who decided · rationale · consequences for training and
interpretation · evidence. "Decided by team" = decided by Tytus Felbor
(2026-11-22); to be confirmed with co-author and mentor.

---

## D-0 · Genuine temporal event input (replaces v1.0 static frame)

- **Decision:** SNN input = original (x, y, t, p) events binned by timestamp into
  chronologically ordered count frames `[T, 2, 34, 34]`; fixed window
  [0, 315 000) µs; events after 315 ms dropped *and counted*. No per-bin or
  per-sample normalisation (raw counts).
- **Decided by:** required by the research question (implementation milestone 1).
- **Why:** v1.0 accumulated each recording into one frame and repeated it at every
  time-step, which removed all timing information and invalidated the
  "event-based" premise.
- **Evidence:** train split: 27 of 250 316 228 events after 315 ms; test split: 155 of
  42 036 067. Smoke test checks that every event is either binned or counted as dropped.
- **Consequences:** all v1.0 results (`results/pilot_smoke/`) are invalid.

## D-G · Time-step grid {3, 6, 12, 24, 48}, T_train = 24 (was {2, 4, 8, 16, 32}, 16)

- **Decision:** inference grid T ∈ {3, 6, 12, 24, 48}; training horizon and training
  bin count T_train = n_time_bins = 24 (13.125 ms per bin).
- **Decided by:** team asked for a fix to the "bins straddle saccades" issue (G); the
  grid values were chosen by the implementer.
- **Why:** N-MNIST consists of three 105-ms saccades. With the old grid, *no* T
  placed bin edges at the saccade boundaries (105, 210 ms). For example, at T = 16
  the boundaries fell inside bins 5 and 10, which mixed two saccade directions in
  one bin. With multiples of 3, every T puts both boundaries exactly on bin edges
  (unit-tested for every T), and each bin lies inside one saccade. The factor-2
  spacing is kept. T_train = 24 keeps the old structure: 16 was the 4th of 5 grid
  values, and so is 24 (three smaller, one larger).
- **Consequences:**
  - The primary outcome ("minimum qualifying T") is now on the {3, …, 48} scale.
    This is not comparable to numbers stated on the old grid; the proposal text
    must be updated.
  - SNN training cost is 1.5× that of T = 16 (24 unrolled steps). Evaluation at
    T = 48 costs 1.5× the old maximum.
  - Alternative not chosen: T_train = 12 (cheaper, 26.25-ms bins). That would put
    T_train in the middle of the grid.

## D-A · Evaluation at T: re-bin the same window into T bins

- **Decision:** to evaluate at inference T, the same [0, 315 ms) window of the same
  recordings is re-binned into T equal bins (`make_eval_loaders_by_T`). Counts are
  multiplied by T / T_train (`t_eval_input_scaling = "match_training_rate"`).
- **Decided by:** team (followed the recommendation).
- **Why:**
  - Every T sees all events of the recording. T changes only the temporal
    resolution and the number of simulation steps, not how much of the recording
    is observed. The alternative (using the first T bins) would make low T see
    only the first saccade.
  - The scaling keeps the expected per-step input equal to training. A T = 3 bin
    contains 8× the events of a T = 24 bin, so unscaled input would drive the
    T_train-trained network far outside its training regime.
- **Sub-decision taken by the implementer (please confirm):** the scaling rule
  itself. With `"none"`, low-T inputs would be 8× (T = 3) to 0.5× (T = 48) the
  training scale.
- **Evidence:**
  - Per-step mean input is identical across T (0.0762 on the smoke subset).
  - Total events are conserved for every T.
  - T = 12 bins equal the pairwise sums of T = 24 bins exactly (tested).
- **Consequences:**
  - A model trained at T_train = 24 is evaluated on inputs with coarser (T < 24)
    or finer (T = 48) temporal resolution. Accuracy differences across T therefore
    reflect both the number of steps and the temporal resolution.
  - The pilot script is unblocked.

- **Addendum (2026-11-22): field standard, and investigating the effect of T.**
  - **What the field does.** SpikingJelly's N-MNIST loader (`data_type='frame'`,
    `frames_number=T`, `split_by='time'`) splits the *whole recording* into T
    equal-time frames of event counts. Re-binning the full window is that same
    convention, so the frame-forming method is a standard, not a guess.
  - **Where we deviate.** Standard practice, and Rathi et al., train or fine-tune
    *at each target T*. The plan trains once (T_train = 24) and only evaluates at
    the other T. That design is called **P1**; training per T is **P2**. P1 is
    cheaper, but at T ≠ 24 its accuracy mixes "fewer steps" with "inputs unlike
    training".
  - **How the effect is investigated rather than assumed:**
    - `run_pilot.py` (P1) reports two sensitivity variants next to the primary
      `rebin_scaled`:
      - `rebin_raw` (no T/24 scaling);
      - `prefix` (the first T training-resolution bins, i.e. same resolution but
        less of the recording).
    - `run_pilot.py --t-train T` runs P2 for one T.
    - The pilot compares P1 and P2 at T = 3 (`docs/COLAB_PILOT_WALKTHROUGH.md` §6).
  - **Decision rule, fixed before the pilot:**
    - If P2 is within ~1–2 pp of P1 at T = 3 and the sensitivity variants agree,
      P1 stays primary.
    - Otherwise P2 becomes the primary protocol for the main experiment (~3.9×
      the SNN training cost).
    - The outcome and the choice are recorded here before the full experiment.

## D-B · Conversion: layer-wise normalisation with LIF-fitted input map + fidelity gate

- **Decision:**
  1. The ANN input is the **mean** over the 24 bins (was the sum). That is the
     average per-step input the SNN receives; the information content is the same.
  2. Conversion uses layer-wise weight normalisation (Rueckauer et al. 2017) with
     λ_l = 99th percentile of the ANN's post-ReLU activations on training data,
     plus an affine LIF input map X = θ(c + k·u). (c, k) is fitted (data-free) to
     the simulated rate curve of the configured LIF (τ = 2, soft reset, θ = 1,
     T = 24): c = 0.83, k = 1.36. Thresholds stay θ = 1, so the converted network
     has exactly the SG arm's neuron hyper-parameters.
  3. **Gate (staged, see "Gate decision" below):** before fine-tuning, the converted
     SNN is compared with the ANN on the **validation** split at T_train.
     - Strict criteria (always checked and recorded): acc_SNN ≥ acc_ANN − 0.02
       **and** prediction agreement ≥ 0.90.
     - Sanity criterion (used if the strict check fails):
       acc_SNN ≥ chance + 0.5 · (acc_ANN − chance).
     - If both fail, `train_snn_hybrid` raises `ConversionGateError`.
- **Decided by:** team asked for a check that the converted SNN matches the ANN;
  the method and thresholds were chosen by the implementer (please confirm the
  thresholds).
- **Why the old conversion was replaced:**
  - It set θ_l = λ_l without rescaling the next layer's weights. Spikes have
    amplitude 1, so layer l+1 received its input scaled by 1/λ_l.
  - It ignored the LIF leak. SpikingJelly's LIF with `decay_input` is silent for
    constant X ≤ θ, and its rate rises steeply and concavely above θ.
  - It calibrated on whole-recording sums while the SNN receives per-bin counts.
- **Evidence (synthetic, ideal constant input, ANN 100 % accurate):**

  | conversion | SNN acc | agreement | per-layer rate corr. |
  |---|---|---|---|
  | plain normalisation (θ = 1, no offset) | 10 % (chance; deep layers silent) | 10 % | n/a |
  | +θ bias offset | 40 % | 40 % | 0.90–0.94 |
  | **fitted map (c = 0.83, k = 1.36)** | **100 %** | **100 %** | 0.97–0.98 |

  The rate-curve RMSE falls from 0.118 to 0.061. Results are the same at T = 200,
  so the remaining error comes from the shape of the rate curve, not from too few
  time-steps.
- **Evidence (real N-MNIST).**
  - Setup: ANN trained on 4 000 training recordings for 5 epochs (val 91.3 %). Gate run on
    1 000 validation recordings at T = 24 (`results/conversion_init_check/report_B.json`,
    `B_gate_stdout.txt`).

  | conversion | converted SNN val | agreement | rate corr. lif1/2/3 |
  |---|---|---|---|
  | plain normalisation (v1.0-like) | 10.0 % | 10.5 % | 0.91 / 0.42 / n/a (silent) |
  | +θ offset | 23.1 % | 25.0 % | 0.85 / 0.74 / 0.67 |
  | fitted map (constant input) | 69.1 % | 71.8 % | 0.94 / 0.92 / 0.88 |
  | **data-refined per layer (default)** | **73.7 %** | **76.8 %** | 0.94 / 0.93 / 0.91 |
  | data-refined + **1 epoch fine-tuning** (125 updates) | **89.4 %** | — | — |

  - Data refinement chose (c, k) = (0.65, 2.4), (0.75, 2.0), (0.83, 1.4) for layers 1–3,
    taking 343 s on CPU.
  - **The strict gate fails on real data**: it requires ≥ 89.3 % and ≥ 90 % agreement.
    The remaining gap comes from bursty event input over 24 steps through leaky neurons.
    The mean rates match their targets (0.125 vs 0.132, 0.094 vs 0.087, 0.101 vs 0.096);
    the per-neuron error is what fails the gate.
  - One epoch of fine-tuning brings the converted network within 1.9 pp of the ANN. So
    conversion transfers useful knowledge, consistent with the hybrid literature: lossy
    conversion at low T, recovered by fine-tuning.
- **Gate decision ("recheck, then relax"), decided by the team, 2026-11-22:**
  - *Recheck.* Every run evaluates the strict criteria and records the result
    (`conversion_fidelity.passed`). The first such run is the Colab pilot with the
    fully trained ANN (48 000 recordings, up to 30 epochs). Its strict result is
    reported in the paper whether it passes or not.
  - *Relax.* If the strict check fails, the gate falls back to a **sanity
    criterion**: the converted SNN must keep at least half of the ANN's
    above-chance accuracy, acc_SNN ≥ 0.1 + 0.5 · (acc_ANN − 0.1).
    - For the diagnostic ANN (91.3 %) this means ≥ 50.7 %.
    - The data-refined conversion (73.7 %) passes.
    - The plain (10.0 %) and offset (23.1 %) conversions fail, so the
      criterion still rejects a conversion that did not transfer knowledge.
  - If both fail, the hybrid arm is refused (`ConversionGateError`).
  - The outcome is stored as `gate_outcome` ∈ {`strict_pass`, `sanity_pass`,
    `fail`} in the checkpoint and the pilot `results.json`.
- **Why relaxing is not lowering the study's bar:**
  - The gate checks one *intermediate step* of the hybrid method (conversion).
    The research criterion (D-C: final SNN within 2 pp of the ANN on test) is
    applied unchanged *after* fine-tuning.
  - Hybrid training is defined as lossy conversion followed by fine-tuning
    (Rathi et al. 2020). Requiring equivalence before fine-tuning would test a
    stronger claim than the method makes.
  - Remaining infidelity comes from bursty event input through leaky neurons
    (a finding to report), not from a defect that architecture tuning should
    remove. Changing the architecture for one arm would also break the paired
    comparison.
- **Implementation choice to confirm:** the sanity fraction 0.5
  (`gate_sanity_fraction`).
- **Consequences:**
  - Method B is only meaningful when its gate passes. If it fails in the main
    experiment, that is a reportable result about conversion on leaky, temporally
    varying input, not something to tune away silently.
  - The gate uses validation data only, so there is no test leakage.

## D-C · Accuracy threshold — kept as submitted

- **Decision:** threshold_s = ANN *test* accuracy − 2 pp (`threshold_set = "test"`),
  unchanged.
- **Decided by:** team.
- **Consequence:** the success criterion depends on the test set. No model
  selection uses the test set, but report this as a limitation.

## D-E · Fairness and the training-compute budget: counted in time-steps

- **Decision:**
  - The fairness analysis is accepted as documented (README, issue E).
  - The training-compute budget is measured in **sample-time-steps**: one unit is
    one training sample passed forward and backward through the network for one
    simulated time-step.
    - ANN training costs 1 unit per sample per epoch (it has no time dimension).
    - SNN training costs T_train = 24 units per sample per epoch.
  - Method B's budget is ANN pre-training plus SNN fine-tuning. Method A's budget
    is SG training.
  - Forward-only work is counted but reported **separately**, never added to the
    training budget. This covers threshold/λ statistics, the per-layer conversion
    refinement, the D-F init calibration and validation.
- **Decided by:** team. The axis is time-steps, counted per sample.
- **Two "time-step" quantities, kept separate:**
  - *Inference T* (the **primary outcome**): how many simulation steps the
    finished SNN needs at test time.
  - *Training sample-time-steps* (the **budget**): how much simulated time was
    spent producing the SNN.

  The research question asks for the first, measured under a clearly defined
  second.
- **Why time-steps rather than epochs or updates:** one update costs very
  different amounts in the two arms. An ANN update unrolls 1 step; an SNN update
  unrolls 24 steps of backpropagation through time. Counting time-steps makes
  the hybrid arm's cheap ANN phase and expensive fine-tuning phase visible, which
  the research plan (Sec. 2.1) says update counts alone cannot do.
- **Implementation:**
  - `TrainingResult.train_sample_timesteps`, with
    `HybridTrainingResult.train_sample_timesteps = ANN + FT`.
  - Forward-only costs are in `conversion_info['refine_forward_sample_timesteps']`
    and `extra['init_forward_sample_timesteps']`.
  - `run_pilot.py` prints and saves all of these, plus wall-clock time and peak
    GPU memory per method.
- **Cap or no cap, decided by the team (2026-11-22): no cap; measure and report.**
  - Both arms run the same training procedure: maximum epochs, early stopping on
    validation accuracy (patience 7), and the same optimiser settings.
  - Neither arm is cut off at a shared budget. Each run records the
    sample-time-steps it actually used, so the report can show which method took
    more compute to build its SNN, alongside the minimum T it reached.
  - Why: a cap would make the outcome depend on where the cap is set. Measuring
    the budget answers "how expensive was each method?" without changing what
    each method can achieve.
  - Consequence for the analysis: the primary outcome (minimum T) is compared at
    whatever budget each arm needed; the budget is reported next to it. A
    statement such as "hybrid reached T = 6 using 13 M sample-time-steps, direct
    SG reached T = 12 using 30 M" must report both numbers.
- **Fairness note added by D-B/D-F:** the two arms use different, method-specific
  initialisation procedures (conversion vs firing-rate calibration). Both must be
  described in the paper.

## D-F · Direct-SG arm: firing-rate-calibrated initialisation

- **Decision:** after PyTorch default initialisation, each Conv/Linear layer's
  weights and bias are multiplied by a scalar gain. The gain is found by bisection
  on 4 fixed training batches, so that the following LIF layer fires at 5 %
  (`init_target_rate = 0.05`). This is applied to the direct-SG arm only.
- **Decided by:** team asked to fix silent layers; method and target rate chosen by
  the implementer (please confirm the 5 % target).
- **Why:** with default init, θ = 1 and τ = 2, the real-data firing rates were
  lif1 = 0.003, lif2 = 0, lif3 = 0. FC1/FC2 got zero gradient. The neuron model,
  threshold, τ and architecture stay unchanged; only per-layer scale changes (in the
  spirit of LSUV, Mishkin & Matas 2016).
- **Evidence (real N-MNIST):**
  - Smoke test gains were 4.5 / 8.7 / 7.6 and rates became 0.050 / 0.050 / 0.049.
    All weights now receive gradient.
  - Short training run (2 000 train / 500 val recordings, 1 epoch = 63 updates,
    same seed): default init — loss 2.307 → 2.262, val acc **13.2 %**; calibrated
    init — loss 2.265 → 0.999, val acc **73.4 %**
    (`results/conversion_init_check/report_F.json`, `F_init_comparison_stdout.txt`).
- **How the 5 % was chosen.** It was first set as a heuristic, then **selected by
  a pre-specified validation sweep**:
  - candidates {1, 2, 5, 10, 20} %, 2 seeds, 2 000 / 500 recordings, 1 epoch each;
  - rule: best mean validation accuracy, ties within 1 pp → lower rate;
  - result: 71.8 / 72.4 / **74.1** / 73.2 / 69.6 % → 5 %
    (`results/conversion_init_check/report_F_sweep.json`).

  The response is flat from 1 to 10 %, so the choice is not fragile.
- **Consequence:** the target rate is a hyper-parameter of Method A, selected on
  training and validation data only. Report it and the sweep.

## D-P · Polarity naming: p = 1 is ON, p = 0 is OFF

- **Decision:** follow the dataset authors' definition. In the original N-MNIST
  40-bit event format, bit 23 is polarity, 0 = OFF and 1 = ON. Tonic decodes
  exactly this bit (`(byte[2] & 128) >> 7`). Channel 0 = OFF, channel 1 = ON.
- **Decided by:** the team followed a recommendation from the dataset documentation.
- **Note:** the smoke test's leading-edge diagnostic (p = 0 events lead the assumed
  saccade motion) disagrees with this. It relies on an unverified assumption about
  display contrast and apparent-motion direction, so it is kept only as a recorded
  observation.
- **Consequence:** naming only; both models see the same channel order.
