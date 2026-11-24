# Implementation-level decisions (not yet reviewed by the team)

The team-level decisions are in [`DECISIONS.md`](DECISIONS.md). This file lists
every **other** choice made during implementation that could affect the
results, together with why it was made, the alternatives, and where it lives in
the code. The team should review each item and confirm it or change it before
the main experiment.

Legend:

- **Impact H** — can change the primary outcome or the comparison.
- **Impact M** — can change numbers, but should not bias the comparison between arms.
- **Impact L** — engineering only.

---

## 0. Origin of every number

Each value is one of four kinds (explained in [`PRIMER.md`](PRIMER.md) §10):
- **M**: measured from data or simulation;
- **S**: field standard or literature;
- **P**: selected by a pre-specified procedure on training/validation data;
- **D**: a policy/design threshold. It defines the question, so it cannot be
  optimised without circularity; it is fixed in advance and justified by its
  meaning.

| Value | Kind | Where it comes from / evidence |
|---|---|---|
| Window 0–315 ms | M | Three 105-ms saccades (Orchard 2015). Full train-split scan: only 27 of 250 316 228 events fall after 315 ms |
| Event bins = counts per pixel and polarity | S | Standard event-to-frame integration (SpikingJelly `integrate_events_segment_to_frame`) |
| Whole recording split into T equal-time bins | S | SpikingJelly N-MNIST `frames_number=T, split_by='time'` |
| T grid {3, 6, 12, 24, 48} | D | Smallest change to the plan's grid that aligns every bin edge with the saccade boundaries (tested) |
| T_train = 24 | D | Same position in the grid as the plan's 16. **P1 vs P2 is decided by the pilot** (DECISIONS D-A addendum) |
| T/T_train input scaling (P1 only) | D, + sensitivity | Keeps the mean input per step equal to training (measured 0.0762 at every T). Its effect is checked by the `rebin_raw` variant |
| ANN input = mean over bins | S | The ANN must see the per-step input that the SNN averages over time (rate-coding conversion, Rueckauer 2017) |
| λ = 99th percentile of activations | S | Rueckauer et al. 2017 ("robust normalisation") |
| LIF map c = 0.83, k = 1.36 | M | Least-squares fit to a simulated LIF neuron (τ = 2, θ = 1, T = 24). Rate RMSE 0.061 vs 0.118 for the naive map |
| Per-layer (c_l, k_l) | P | Coordinate search minimising rate error vs ANN targets on 4 training batches |
| Strict gate: 2 pp, 90 % agreement | D | 2 pp = the study's own margin; 90 % agreement = conversion must mostly reproduce the ANN's decisions |
| Sanity gate: ≥ ½ of above-chance accuracy | D | "Conversion transferred most of the knowledge." The observed separation is wide (failed conversions 10 % / 23 % vs working ones 69 % / 74 % against a 50.7 % line), so any fraction from 0.25 to 0.7 gives the same decisions on current evidence |
| Init target firing rate | **P** | Validation sweep over {1, 2, 5, 10, 20} %, 2 seeds; rule: best mean validation accuracy, ties within 1 pp → lower rate (§4.2) |
| Optimiser, learning rates, epochs, patience, clipping | S / inherited | Kept from protocol v1.0 (§6); standard Adam + cosine settings. Not tuned per arm, for fairness |
| 2-pp accuracy criterion, test-based | D | Research plan (D-C) |
| Op counts: SynOps = spikes × fan-out; input MACs = non-zero events × fan-out; padding excluded | S | SNN literature convention; see DECISIONS D-H. Energy constants MAC 4.6 pJ / AC 0.9 pJ from Horowitz (ISSCC 2014), 45 nm |
| Seed count | P (pending) | To be chosen after the pilot from variance and time per seed (plan Sec. 2.1) |

## 1. Data and preprocessing

| # | Choice | Why | Alternatives | Impact | Where |
|---|---|---|---|---|---|
| 1.1 | **Fixed** window [0, 315 000) µs for every recording, not [first, last event] per recording | Bin k is then the same physical time and saccade phase in every sample. 315 ms = 3 × 105-ms saccades (Orchard et al. 2015; Tonic constants). Verified on the full train split. | `per_recording` window (implemented, unused); a 336-ms window covering every event (the last bin would be almost empty) | H | `PreprocessConfig.window_*`, `binning_strategy` |
| 1.2 | Events after 315 ms are **dropped and counted**, not raised as an error | 27 of 250 M train events; dropping them changes nothing measurable, and every drop is counted per recording | `out_of_window_policy="error"` | L | `out_of_window_policy` |
| 1.3 | Integer bin rule `floor((t − t_start)·n / D)`, D = number of µs ticks | Exact with no floating-point rounding; the last tick lands in the last bin without clipping; bin widths differ by at most 1 µs | float division + clipping | L | `transforms.bin_indices` |
| 1.4 | Bin value = **raw event count**, no normalisation | Keeps relative intensity across time and samples; uses no data-derived constants | one global scale constant (implemented as `global_scale`); per-bin or per-sample normalisation (rejected: it erases temporal intensity) | M | `normalization` |
| 1.5 | Channel index = polarity value p (0 = OFF, 1 = ON) | Direct mapping of the dataset's bit 23 | — | L | `transforms.events_to_temporal_frames` |
| 1.6 | Binning happens on the fly; **no disk cache** | No stale v1.0 single-frame caches can be reused | a cache keyed by `preprocessing_signature` (helper exists) | L (speed) | `nmnist.py` |
| 1.7 | Train/validation split: stratified per class, 80/20, fixed split seed 42, identical for every experiment seed and both representations | Same recordings for both arms; class balance in validation | random unstratified split | M | `make_train_val_split` |
| 1.8 | Tonic 1.7.0 as the decoder of the original `.bin` files | Maintained reference implementation; its output format was verified on real files | own decoder | L | `requirements.txt` |

## 2. Time-step grid and evaluation (supporting D-G, D-A)

| # | Choice | Why | Alternatives | Impact | Where |
|---|---|---|---|---|---|
| 2.1 | Grid **{3, 6, 12, 24, 48}** | Smallest change to the old powers-of-2 grid that aligns every T with the saccades: it keeps factor-2 spacing and 5 points | {3, 6, 12, 24}; {6, 12, 24, 48}; a denser grid | H | `SNNConfig.t_eval_grid` |
| 2.2 | **T_train = 24** | Same position in the grid as the old 16 (3 values below, 1 above) | 12 (cheaper; T_train then sits mid-grid) | H | `t_train`, `n_time_bins` |
| 2.3 | Evaluation input scaled by **T / T_train** (`match_training_rate`) | Keeps the per-step input in the range the model was trained on; without it, T = 3 inputs are 8× the training scale | `"none"` (raw counts per coarser bin) | H | `t_eval_input_scaling` |
| 2.4 | **P1:** one model trained at T_train, evaluated at every grid T | Matches the plan's "fixed training horizon" design; ~3.9× cheaper than training per T | **P2:** train per T, the field standard (`run_pilot.py --t-train T`). **Decided by the pilot** using a pre-set rule (DECISIONS D-A addendum) | H | `run_pilot.py` |
| 2.5 | Sensitivity variants `rebin_raw` and `prefix` reported next to the primary `rebin_scaled` | Show whether the accuracy-vs-T curve depends on scaling or on observation length rather than on step count | none (would leave the T effect unexamined) | M | `make_eval_loaders_by_T(variant=…)` |

## 3. Conversion and gate (supporting D-B)

| # | Choice | Why | Alternatives | Impact | Where |
|---|---|---|---|---|---|
| 3.1 | ANN input = **mean** over bins (was sum) | It is the per-step input the SNN receives, so the ANN activation scale matches; the information is identical | sum (needs a 1/T correction elsewhere) | M | `ann_representation` |
| 3.2 | λ_l = **99th percentile** of post-ReLU activations, from 10 training batches | Standard robust normalisation (Rueckauer et al. 2017) | 99.9 / 100 (tested on synthetic data: 99.9 similar, 100 slightly worse) | M | `calib_percentile`, `calib_batches` |
| 3.3 | Thresholds kept at **θ = 1**; scaling moved into the weights | The converted SNN then has exactly the SG arm's neuron hyper-parameters | per-layer thresholds (the v1.0 approach, flawed) | H (fairness) | `normalize_for_lif` |
| 3.4 | Affine LIF input map X = θ(c + k·u), first fitted to the simulated LIF rate curve (grid search: c ∈ [0.5, 1.2], k ∈ [0.5, 3.0]) | The LIF used here is silent below θ and has a concave rate curve; a plain IF-style mapping gives chance accuracy (measured) | IF neurons for the converted model (changes the neuron model between arms); larger τ for both arms (changes a locked setting) | H | `fit_lif_input_map` |
| 3.5 | Per-layer **data refinement** of (c, k) on 4 training batches; coordinate search, 3 rounds; c ∈ [0.40, 1.10] step 0.05, k ∈ [0.6, 4.0] step 0.2; objective = rate MSE vs ANN target | Event input is bursty, so constant-input theory under-estimates the rate error; refinement improved real-data accuracy from 69.1 % to 73.7 % | no refinement (`"fitted"`); more batches; directly optimising agreement (closer to fine-tuning) | H | `refine_input_map_on_data`, `conversion_refine_batches` |
| 3.6 | Strict gate thresholds: accuracy drop ≤ **2 pp** and agreement ≥ **90 %**, on the full validation split at T_train | 2 pp mirrors the study's own accuracy margin; 90 % agreement makes sure the match is not just accuracy by coincidence | other margins | M (now recorded, not blocking) | `gate_max_acc_drop`, `gate_min_agreement` |
| 3.7 | Gate mode **staged** by default (team decision); **report** in pilot smoke mode | Implements "recheck, then relax"; smoke mode trains for 4 updates and could never pass | `enforce` (strict only) | M | `conversion_gate`, `run_pilot.py` |
| 3.9 | Sanity criterion: converted SNN keeps **≥ 50 %** of the ANN's above-chance accuracy | Scale-free (adapts to the ANN accuracy reached); clearly separates conversions that transfer knowledge (73.7 % passes) from failed ones (10 %, 23 % fail) | fixed floor (e.g. ≥ 50 %); "better than untrained SNN + x pp"; 0.25 / 0.75 fraction | H | `gate_sanity_fraction` |
| 3.8 | Gate diagnostics: per-layer Pearson correlation between ANN target rate and SNN rate | Shows *where* fidelity is lost; not a pass/fail criterion | — | L | `conversion_fidelity` |

## 4. Direct-SG initialisation (supporting D-F)

| # | Choice | Why | Alternatives | Impact | Where |
|---|---|---|---|---|---|
| 4.1 | Scale each Conv/Linear layer (weights and bias) by a scalar gain; leave the random pattern unchanged | Smallest intervention that makes the layers fire; no change to the neuron model or architecture | change θ or τ (locked); LSUV variance matching; bias-only shift | H | `init_calibration.py` |
| 4.2 | Target firing rate **5 %** per LIF layer, **selected by validation sweep** | Candidates {1, 2, 5, 10, 20} %; 2 seeds; 2 000 training / 500 validation recordings; 1 epoch each. Rule fixed before running: best mean validation accuracy, ties within 1 pp → lower rate. Mean val acc: 71.8 / 72.4 / **74.1** / 73.2 / 69.6 %. The curve is flat from 1–10 % (seed variation 5–7 pp exceeds the differences between rates) and drops at 20 % | — | M (low sensitivity) | `init_target_rate`; `results/conversion_init_check/report_F_sweep.json` |
| 4.3 | Gains found by bisection in log space on the **first 4 unshuffled training batches** | Deterministic; uses training data only; does not consume the shuffle RNG | random batches | L | `init_calib_batches` |

## 5. Budget accounting (supporting D-E)

| # | Choice | Why | Impact |
|---|---|---|---|
| 5.1 | Training budget counts forward+backward sample-time-steps over all epochs actually run, including the epochs after the best one up to early stopping | That is the compute actually spent | M |
| 5.2 | Forward-only costs (λ statistics, conversion refinement, init calibration, validation) are reported separately | A forward pass costs roughly 1/3 of forward+backward; adding them would mix units | M |
| 5.3 | Peak GPU memory is recorded per method in `run_pilot.py` | Required by Research Plan Step 6 / Risk 2 | L |

## 6. Training defaults inherited from v1.0 (unchanged, listed for completeness)

Adam; learning rates ANN 1e-3, SG 1e-3, fine-tuning 1e-4; weight decay 1e-4;
cosine LR schedule; batch size 128; early stopping on validation accuracy with
patience 7; gradient clipping at 1.0 for SNN training; maximum epochs ANN 30,
fine-tuning 10, SG 30; output readout = time-averaged FC2 logits.

## 7. Engineering and test choices

| # | Choice | Why |
|---|---|---|
| 7.1 | SpikingJelly pinned to 0.0.0.0.14 | `reset_net` semantics and `LIFNode` defaults are relied upon |
| 7.2 | DataLoader `worker_init_fn` is a `functools.partial` of a module-level function | The closure version crashed under the `spawn` start method (macOS, Python ≥ 3.14 on Linux) |
| 7.3 | Mechanism unit tests use θ = 0.2 | At default θ with random init the deep layers are silent, which would make the state tests vacuous; the experiment config is unaffected |
| 7.4 | Diagnostics (`check_conversion_and_init.py`) use subsets (2 000 / 4 000 training and 500 / 1 000 validation recordings), seed 0, CPU | Engineering evidence only; **not** experimental results |

## 8. Things deliberately NOT decided

- ~~Equal-budget cap vs measure-and-report~~: decided (D-E: no cap, measure and report).
- ~~Gate semantics~~: decided (staged gate, D-B). The sanity fraction is still to confirm.
- Seed count and statistical test (Research Plan Sec. 2.1, after the pilot).
