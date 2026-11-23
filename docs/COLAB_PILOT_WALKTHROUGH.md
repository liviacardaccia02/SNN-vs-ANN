# Pilot run on Google Colab: walkthrough

This is the research plan's **Step 6, the pilot**: one seed, full data, both arms.
It answers four questions before the main experiment:
1. Does everything run on a GPU, and how long and how much memory does one seed take?
2. **Recheck of the conversion gate** with a fully trained ANN (decision D-B).
3. How much the accuracy-vs-T curve depends on how a T-step input is formed
   (protocol P1 vs P2, see `PRIMER.md` §5).
4. Do both arms learn, and is the T grid able to discriminate between them?

A pilot with one seed **is not a result** about the research question. It decides
the final protocol (plan Step 7). Do not run the full experiment until the
decisions in step 7 below are made.

Prerequisite: the branch `feat/event-based-temporal-pipeline` is pushed (see the
commit/PR commands handed over with this document).

**Ready-made notebook:** [`notebooks/02_colab_pilot.ipynb`](../notebooks/02_colab_pilot.ipynb)
contains every step below as cells, plus helper cells that:
- estimate the full-pilot time;
- summarise `results.json`;
- apply the P1/P2 rule.

Open it in Colab via *File → Open notebook → GitHub*: paste the repository URL and
choose the branch. Its clone cell pulls the latest commit when re-run and prints the
commit hash, so you always know which code ran.

**History.** The first Colab attempt (2026-11-23, commit `f344818`) crashed in two
places, both fixed in the following commit:
1. Conversion calibration: a float32 index overflow above 2²⁴ activations. The
   percentile is now exact, computed with NumPy.
2. The smoke test's final step, when `--out-dir` was on Drive (outside the repo).

The gate's correlation diagnostics also now stream instead of holding about 3.5 GB in
RAM.

---

## 1. Open Colab with a GPU

New notebook → *Runtime → Change runtime type → T4 GPU* (any GPU works). Each cell
below is one notebook cell. A `!` runs a shell command.

```python
!nvidia-smi
```
Expected: a table naming the GPU (for example "Tesla T4", 15 GB). If you see
"command not found", the runtime has no GPU; change the runtime type.

## 2. Code, dependencies, storage

```python
from google.colab import drive
drive.mount('/content/drive')          # results survive a disconnect only on Drive
!mkdir -p /content/drive/MyDrive/snn_results
```
```python
!git clone -b feat/event-based-temporal-pipeline https://github.com/liviacardaccia02/SNN-vs-ANN.git /content/SNN-vs-ANN
%cd /content/SNN-vs-ANN
!pip install -q tonic==1.7.0 spikingjelly==0.0.0.0.14 pytest
```
Colab already ships PyTorch with CUDA. Do **not** reinstall torch.

```python
!python -m pytest tests/test_temporal_transforms.py tests/test_models.py tests/test_conversion.py -q -p no:warnings
```
Expected: all tests pass (about 1–3 min). These tests need no dataset. If any fail,
stop: the environment differs from the tested one. Record the error.

## 3. Dataset (≈ 1.1 GB, about 5–10 min)

Keep the data on Colab's local disk (`/content/data`), not on Drive. Reading 70 000
small files from Drive is very slow.

```python
!python -c "import tonic; tonic.datasets.NMNIST('/content/data', train=False); tonic.datasets.NMNIST('/content/data', train=True)"
```
If the Mendeley download fails, run `!python scripts/fix_nmnist_testset.py`. The data
must be downloaded again in every new Colab session.

Pipeline check on real data:
```python
!python scripts/test_event_pipeline.py --data-root /content/data --device cuda --out-dir /content/drive/MyDrive/snn_results/smoke
```
Expected: the last line reads `STATUS: PASS (28 checks, 0 critical failures, …)` and
`device_used: cuda` appears in the environment block.

## 4. Timing probe (≈ 15–40 min)

One epoch of every stage, full data. This measures how long a real run takes before
you commit hours to it.

```python
!python scripts/run_pilot.py --seed 0 --data-root /content/data \
    --ann-epochs 1 --ft-epochs 1 --sg-epochs 1 --no-sensitivity \
    --output-dir /content/drive/MyDrive/snn_results/timing_probe 2>&1 | tee /content/drive/MyDrive/snn_results/timing_probe.log
```

What to read from the output:
- `[ANN seed=00] Epoch 1/1 … time=XXs`, `[SNN-FT …] time=XXs`, `[SNN-SG …] time=XXs`:
  the seconds per epoch for each stage.
- The lines `[Hybrid seed=00] lambdas: …` and `conversion gate (staged): …`. After
  one ANN epoch the gate result means nothing; here we only want the timing. If it
  reports `FAIL`, the run stops before the SG arm. In that case rerun the probe with
  `--gate-mode report` to get the SG timings.
- `Peak GPU memory (Method A/B): … MB`.

Estimate the full pilot as:

> ANN s/epoch × 30 + FT s/epoch × 10 + SG s/epoch × 30 + evaluation time

That is an upper bound, because early stopping usually ends sooner. If the estimate
exceeds about 8 h, a free Colab session may disconnect first. Then either run the
arms in separate sessions, or reduce the maximum epochs for the pilot only, and
**record that in the log**.

If you see `CUDA out of memory`, add `--batch-size 64`. Record it: it is a protocol
deviation (plan, Risk 2).

## 5. Full pilot, protocol P1 (train at T = 24, evaluate all T)

```python
!python scripts/run_pilot.py --seed 0 --data-root /content/data \
    --output-dir /content/drive/MyDrive/snn_results/pilot_seed00_Ttrain24 2>&1 | tee /content/drive/MyDrive/snn_results/pilot_seed00_Ttrain24.log
```

### How to read the output, section by section

| Output section | What it means | What to expect / red flags |
|---|---|---|
| `METHOD B … [ANN seed=00] Epoch …` | ANN training on the mean-over-bins input. `val acc` is validation accuracy. | It should climb fast. Small CNNs on N-MNIST typically exceed 95 %. **Red flag:** below 90 % after several epochs. |
| `lambdas: conv1=…` | The 99th-percentile ANN activation per layer, used to normalise conversion. | Positive, finite numbers. |
| `conversion gate (staged): STRICT_PASS / SANITY_PASS / FAIL …` | **The recheck (D-B).** Strict = converted SNN within 2 pp of the ANN with ≥ 90 % agreement; sanity = keeps ≥ 50 % of above-chance accuracy. | Write down all of: `strict PASS/FAIL`, `sanity …`, SNN val accuracy, ANN val accuracy, agreement. `FAIL` stops the run with `ConversionGateError`; that is a result to report (see §7). |
| `[SNN-FT seed=00] Epoch …` | Hybrid fine-tuning, starting from the converted weights. | Val accuracy should start near the converted accuracy and rise towards the ANN within a few epochs. |
| `Method B summary`, `training budget (D-E)` | Epochs, updates, wall-clock time and **sample-time-steps** used: ANN + FT. | Note all of them; they are the training-cost secondary outcomes. |
| `ANN test evaluation -> threshold_s` | ANN **test** accuracy, and the target = test accuracy − 0.02 (D-C). | This is the bar every SNN must reach. |
| `init calibration: lif1: g=… rate a->b` | D-F: the gains that bring each layer to its target firing rate. | The "after" rates close to the target (within 10 %). |
| `[SNN-SG seed=00] Epoch …` | Direct SG training. | Val accuracy rising over epochs. **Red flag:** stuck near 10–20 %. |
| `Method A summary`, budget, `Peak GPU memory` | Same as for Method B. | SG usually costs more sample-time-steps than hybrid (Fig. 6 shows the maxima). |
| `Results table` | Test accuracy of each SNN at T = 3 … 48 (primary variant `rebin_scaled`). `*` = reaches the target. `Min qualifying T` per arm. | Accuracy normally rises with T. **The headline pilot number is the min qualifying T of each arm.** |
| `Sensitivity: evaluation variant "rebin_raw"` / `"prefix"` | The same models, but T-step inputs formed differently (see `PRIMER.md` §5). | Compare with the main table, as in §6. |
| `PILOT SANITY CHECKS` | Automatic plausibility checks: both arms learn, ANN > 80 %, some T qualifies, accuracy varies across T, small degradation at T = 24. | `[FAIL]` lines are things to discuss, not crashes. The final "PILOT VERDICT" summarises them. |
| `TIMING SUMMARY` | Wall-clock time per arm, and an estimate for 5 seeds. | Feeds the seed-count decision. |

Everything printed is also saved in `results.json` in the output directory. The
fields you will cite:

```
ann_baseline.test_acc, ann_baseline.threshold_s
method_b_hybrid.training.conversion_fidelity.{ann_acc, snn_acc, agreement, passed, sanity_passed, gate_outcome}
method_b_hybrid.training.{train_sample_timesteps, ann_sample_timesteps, ft_sample_timesteps, peak_gpu_mb, total_wall_s}
method_a_sg.training.{train_sample_timesteps, init_calibration, peak_gpu_mb, total_wall_s}
method_a_sg.evaluation.snn_evals[*].{T, test_acc, qualifies}, method_a_sg.evaluation.min_qualifying_T
method_b_hybrid.evaluation.(same)
evaluation_sensitivity.{rebin_raw, prefix}.{snn_sg, snn_ft}.min_qualifying_T
hardware.*  (GPU name, versions)
```

## 6. Optional P2 probe (one extra T, ≈ the cost of one short run)

This checks whether "train once at 24, test at 3" understates what a 3-step SNN can
do:

```python
!python scripts/run_pilot.py --seed 0 --data-root /content/data --t-train 3 --no-sensitivity \
    --output-dir /content/drive/MyDrive/snn_results/pilot_seed00_Ttrain3 2>&1 | tee /content/drive/MyDrive/snn_results/pilot_seed00_Ttrain3.log
```

Compare **accuracy at T = 3** between:
- P2: this run's results table, row T = 3 (each model trained at 3);
- P1: the §5 run's results table, row T = 3 (models trained at 24).

| Observation | Interpretation | Consequence |
|---|---|---|
| P2 ≈ P1 (within ~1–2 pp) and the §5 sensitivity variants agree with each other | Evaluation at other T is not distorted by train/test mismatch | P1 is acceptable as the primary protocol, which is cheaper |
| P2 clearly higher than P1, or the sensitivity variants disagree | P1 understates low-T ability | Use **P2** (one run per T) for the main experiment, with about 3.9× the SNN training cost; check feasibility with the §4 timings |

## 7. After the pilot: decisions to record (plan Step 7)

1. **Gate recheck outcome.**
   - `strict_pass`: report that conversion is near-lossless with a fully trained ANN.
   - `sanity_pass`: report the fidelity numbers; this is the expected "lossy
     conversion, fine-tuning recovers" case.
   - `fail`: the hybrid arm could not be run. That is itself a finding; discuss it
     with the mentor before changing anything.
2. **Protocol P1 or P2** (from §6).
3. **Seed count and statistical test** (plan Sec. 2.1). Use the timing summary: 5
   seeds × (time per seed). Remember that 5 paired seeds cannot reach p < 0.05 with
   a Wilcoxon test.
4. Any deviations: batch size, reduced epochs, interrupted sessions.

Add each decision to `docs/DECISIONS.md` *before* starting the full experiment, so
nothing is chosen after seeing main-experiment results.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: N-MNIST … not found` | Wrong `--data-root`, or the download didn't finish; rerun §3. |
| `ConversionGateError` | The gate refused the hybrid arm (see §7.1). The SG arm did not run in that invocation (Method B runs first). For diagnosis, rerun with `--gate-mode report`, which records the gate result but does not block, and note in the log that the hybrid result was obtained *despite* a failed gate. |
| `CUDA out of memory` | Use `--batch-size 64` (or 32) and record it. |
| Session disconnected | Results only persist on Drive. Rerun the interrupted step; every run has its own output directory and run ID. |
| Very slow epochs, GPU idle (`!nvidia-smi` shows low utilisation) | Data loading is the bottleneck (events are binned on the fly). Expected to some degree; note the time. |
