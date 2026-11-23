# Legacy v1.0 pilot output — methodologically invalid

`results.json` in this folder was produced by protocol v1.0, which fed the SNN
one accumulated event frame repeated over all time-steps (no event timing).
It must not be used as evidence for any research question.

Additional defects visible in the file:

- `ann_baseline.test_acc = 0.0` and `threshold_s = -0.02`, so every SNN
  "qualified" at every T. The smoke mode took `range(256)` of class-ordered
  N-MNIST files, i.e. a single-class test subset (fixed in `scripts/run_pilot.py`).
- The 3 checkpoints (≈36 MB) were removed from version control on
  branch `feat/event-based-temporal-pipeline`; they remain in git history.

`../stage1_sample_visualization.png` likewise shows the v1.0 per-sample
max-normalised aggregate frames, not the v2.0 temporal representation
(see `../event_pipeline_smoke/temporal_bins_sample.png`).
