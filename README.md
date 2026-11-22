# SNN-vs-ANN

**Time-Step Efficiency of Hybrid Conversion-and-Fine-Tuning versus Direct
Surrogate-Gradient Training for Spiking Neural Networks on Event-Based Data**
— KTH II2202, Livia Cardaccia & Tytus Felbor.

> **Status (protocol v2.1-draft, 2026-11-22).** The event-based pipeline,
> evaluation-time re-binning, conversion with a fidelity gate, and the
> direct-SG initialisation are implemented and tested. **No experiment has been
> run.** All methodological decisions, with rationale and evidence, are in
> [`docs/DECISIONS.md`](docs/DECISIONS.md).
> Concepts explained for newcomers: [`docs/PRIMER.md`](docs/PRIMER.md). Running
> the pilot: [`docs/COLAB_PILOT_WALKTHROUGH.md`](docs/COLAB_PILOT_WALKTHROUGH.md). Open items are listed at the end of
> this README.

## Data representations

| Level | Shape | Defined in | Temporal order |
|---|---|---|---|
| Original event stream `e_i = (x_i, y_i, t_i, p_i)` | `[N]` structured array | Tonic decoding of the original `.bin` files | yes (µs timestamps) |
| Binned sequence — **SNN input** | `[T, 2, 34, 34]` (batched `[T, B, 2, 34, 34]`); T_train = 24 | `src/data/transforms.py::events_to_temporal_frames` | yes |
| Aggregate — **ANN input** | `[2, 34, 34]` = **mean** over the 24 bins | `temporal_to_aggregate(..., "mean")` | no (removed on purpose) |

Both model inputs come from the same recordings, split and binning (tested).
Channel 0 = OFF (p = 0), channel 1 = ON (p = 1), following the dataset's
bit-23 polarity definition (decision D-P).

### Verified properties of N-MNIST (Tonic 1.7.0)

- dtype `[('x','<i8'),('y','<i8'),('t','<i8'),('p','<i8')]`; x, y ∈ [0, 33]; p ∈ {0, 1}.
- `t` is in integer **microseconds** and non-decreasing within each file.
  Recordings last ≈ 297–336 ms (three ≈ 105-ms saccades).
- Train split: 27 of 250 316 228 events (in 24 of 60 000 recordings) occur at t ≥ 315 ms.
  Test split: 155 of 42 036 067.
- Event rate rises in bursts mid-saccade and drops in lulls at the turn-arounds
  (`results/event_pipeline_smoke/temporal_bins_sample.png`).

### Temporal binning (deterministic, integer arithmetic)

- **Window.** Every recording uses the same fixed window `[0, 315 000)` µs. Training uses
  24 bins of 13 125 µs: `bin(t) = floor(t · 24 / 315 000)`. The 105-ms and 210-ms saccade
  boundaries fall exactly on bin edges (D-G).
- **Out-of-window events.** Events at t ≥ 315 ms are **dropped and counted**. For every
  recording, binned + dropped = original is checked.
- **Bin values.** Each bin holds raw event counts. There is no per-bin or per-sample
  normalisation. Empty bins stay zero and are reported.
- **Validation.** Invalid coordinates, polarities or timestamps raise an error. Binning
  uses timestamps only, so the order of events in the input array does not matter.
- **No disk cache.** Any future cache must be keyed by `preprocessing_signature(cfg)`.

### Evaluation at inference T (D-A)

- **Grid.** T ∈ {3, 6, 12, 24, 48} (D-G).
- **Re-binning.** For each T, the **same 315-ms window** of the same recordings is re-binned
  into T bins. Counts are multiplied by T / 24, so the per-step input matches training.
  Loaders come from `make_eval_loaders_by_T`.
- **What is guaranteed.** Every T sees all events. The coarse bins equal sums of the fine
  bins exactly (tested).

## Models

- **Architecture.** Unchanged: Conv(2→32)–LIF–AvgPool–Conv(32→64)–LIF–AvgPool–FC(4096→256)–LIF–FC(256→10).
- **Neuron model.** LIF, τ = 2, θ = 1, soft reset, ATan surrogate gradient, SpikingJelly multi-step mode.
- **Readout.** Output = mean of FC2 over T.
- **State handling.** Membrane state carries across the bins of one recording.
  `functional.reset_net` clears it before every batch (tested).
- **Method A (direct SG).** Uses firing-rate-calibrated initialisation (D-F). Per-layer
  gains bring every LIF layer to a 5 % firing rate at initialisation.
- **Method B (hybrid).** Steps:
  1. Train the ANN on the mean-over-bins input.
  2. Convert with layer-wise normalisation and an LIF input map, refined per layer on
     training data (D-B).
  3. Run the **conversion-fidelity gate** on the validation split.
  4. Fine-tune with SG.

  The gate is staged (default `conversion_gate="staged"`):
  - the strict criteria (within 2 pp, ≥ 90 % agreement) are always checked and recorded;
  - if they fail, fine-tuning proceeds only if the converted SNN keeps at least half of
    the ANN's above-chance accuracy;
  - otherwise the hybrid arm is refused.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -c "import tonic; tonic.datasets.NMNIST('./data', train=False); tonic.datasets.NMNIST('./data', train=True)"
python -m pytest tests/ -q                                  # unit + real-data tests
python scripts/test_event_pipeline.py                       # pipeline smoke test
python scripts/check_conversion_and_init.py --skip-f        # conversion gate diagnostic (B)
python scripts/check_conversion_and_init.py                 # + init comparison (F); slow on CPU
python scripts/run_pilot.py --seed 0                        # pilot (GPU recommended)
```

## Open items (need a team / mentor decision)

1. **Conversion gate: decided as "recheck, then relax"** (D-B).
   - With a briefly trained ANN (4 000 recordings, 5 epochs, 91.3 % validation), the best
     conversion reaches 73.7 %. That fails the strict check (≥ 89.3 %) but passes the
     sanity check (≥ 50.7 %).
   - The Colab pilot with the fully trained ANN is the recheck.
   - Still to confirm: the sanity fraction (0.5).
2. ~~Training-compute budget~~: decided (D-E). Counted in sample-time-steps, no cap,
   measured and reported per run.
3. **Implementer-chosen sub-parameters to confirm.** The rate-matching scaling in D-A, the
   gate thresholds in D-B, and the 5 % init target in D-F.
4. **Proposal text.** It must be updated for the new T grid (D-G) and the mean ANN input (D-B).

## Repository map

```
docs/DECISIONS.md          decision log (rationale, evidence, consequences)
docs/PRIMER.md             concepts and vocabulary, where each number comes from
docs/COLAB_PILOT_WALKTHROUGH.md  how to run and read the pilot
src/config.py              all settings; validate_config(), preprocessing_signature()
src/data/transforms.py     event validation, timestamp binning, aggregate
src/data/nmnist.py         NMNISTEventDataset, split, make_dataloaders, make_eval_loaders_by_T
src/models/{ann,snn}.py    matched ANN / SNN
src/conversion.py          conversion (layer-wise norm + LIF input map) and fidelity gate
src/init_calibration.py    firing-rate calibrated initialisation (Method A)
src/trainer.py, src/train_ann.py, src/train_snn.py, src/evaluate.py
scripts/test_event_pipeline.py      real-data pipeline smoke test
scripts/check_conversion_and_init.py  diagnostics for D-B and D-F
scripts/run_pilot.py                pilot runner
tests/                              unit + real-data tests
results/pilot_smoke/                v1.0 output, INVALID (see LEGACY_INVALID.md)
```
