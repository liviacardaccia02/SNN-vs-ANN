# %% [markdown]
# # Stage 1 — Environment and Dataset Verification
# **Project:** SNN-vs-ANN (II2202, KTH, Autumn 2026)
# **Protocol version:** v1.0
#
# Run each cell in order. Do not skip cells.
# All cells are idempotent (safe to re-run).

# %%
# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for box-drawing chars).
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# %% [markdown]
# ## Cell 1 — Install dependencies
#
# Run once per Colab session. Takes ~2 min.

# %%
# Install all required packages. Pinned to known-compatible versions.
# If you get a version conflict, note it and tell your implementation assistant.

import subprocess, sys

packages = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "tonic>=1.4.0",
    "spikingjelly>=0.0.0.0.14",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "pytest>=7.4.0",

]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Installation complete.")

# %% [markdown]
# ## Cell 2 — Version check
#
# Paste the output of this cell into any bug reports or handover summaries.

# %%
import sys, platform, datetime

print(f"Run timestamp  : {datetime.datetime.now().isoformat()}")
print(f"Python         : {sys.version}")
print(f"Platform       : {platform.platform()}")
print()

import torch
print(f"PyTorch        : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"GPU name       : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory     : {props.total_memory / 1e9:.1f} GB")

import tonic
print(f"\nTonic          : {tonic.__version__}")

try:
    import spikingjelly
    # SpikingJelly does not always expose __version__; fall back to package metadata.
    try:
        sj_version = spikingjelly.__version__
    except AttributeError:
        import importlib.metadata
        sj_version = importlib.metadata.version("spikingjelly")
    print(f"SpikingJelly   : {sj_version}")
except ImportError:
    print("SpikingJelly   : NOT FOUND — check installation")

import sklearn
print(f"scikit-learn   : {sklearn.__version__}")

import numpy as np
print(f"NumPy          : {np.__version__}")

import matplotlib
print(f"Matplotlib     : {matplotlib.__version__}")

# %% [markdown]
# ## Cell 3 — Environment detection and project root setup
#
# This cell auto-detects whether you are running in Google Colab or locally.
# - **Colab:** optionally mounts Google Drive for persistence; clones the repo.
# - **Local:** uses the repo directory you are already running from.

# %%
import os, sys

# ---- Detect environment ----
try:
    import google.colab          # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Colab: {IN_COLAB}")

if IN_COLAB:
    # ---- Colab: mount Drive (optional) ----
    USE_DRIVE = True             # Set False to skip Drive; data resets each session
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        PROJECT_ROOT = '/content/drive/MyDrive/SNN-vs-ANN'
    else:
        PROJECT_ROOT = '/content/SNN-vs-ANN'
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    # ---- Colab: clone / pull repo ----
    import subprocess
    REPO_URL   = "https://github.com/liviacardaccia02/SNN-vs-ANN.git"
    CLONE_DIR  = os.path.join(PROJECT_ROOT, "repo")
    if not os.path.exists(os.path.join(CLONE_DIR, ".git")):
        subprocess.check_call(["git", "clone", REPO_URL, CLONE_DIR])
        print(f"Cloned to {CLONE_DIR}")
    else:
        subprocess.check_call(["git", "-C", CLONE_DIR, "pull"])
        print(f"Pulled latest into {CLONE_DIR}")

    if CLONE_DIR not in sys.path:
        sys.path.insert(0, CLONE_DIR)
    os.chdir(CLONE_DIR)

else:
    # ---- Local: the project root is the parent of the notebooks/ folder ----
    # __file__ is  <project_root>/notebooks/01_env_and_dataset.py
    CLONE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = CLONE_DIR
    if CLONE_DIR not in sys.path:
        sys.path.insert(0, CLONE_DIR)
    os.chdir(CLONE_DIR)
    print(f"Local project root: {CLONE_DIR}")

# %% [markdown]
# ## Cell 4 — (Colab only: handled in Cell 3)
# Nothing to do here when running locally.

# %% [markdown]
# ## Cell 5 — Load configuration

# %%
from src.config import get_config

cfg = get_config()

# Override all paths relative to the project root (works for both local and Colab)
cfg.data.data_root            = os.path.join(PROJECT_ROOT, "data")
cfg.data.cache_root           = os.path.join(PROJECT_ROOT, "cache")
cfg.experiment.output_dir     = os.path.join(PROJECT_ROOT, "results")
cfg.experiment.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
cfg.experiment.log_dir        = os.path.join(PROJECT_ROOT, "logs")

for d in [cfg.data.data_root, cfg.data.cache_root,
          cfg.experiment.output_dir, cfg.experiment.checkpoint_dir,
          cfg.experiment.log_dir]:
    os.makedirs(d, exist_ok=True)

print("Configuration loaded:")
print(f"  data_root      : {cfg.data.data_root}")
print(f"  cache_root     : {cfg.data.cache_root}")
print(f"  T_train        : {cfg.snn.t_train}")
print(f"  T_eval_grid    : {cfg.snn.t_eval_grid}")
print(f"  accuracy_margin: {cfg.experiment.accuracy_margin:.0%}")
print(f"  threshold_set  : {cfg.experiment.threshold_set}")
print(f"  seeds          : {cfg.experiment.seeds}")
print(f"  protocol_ver   : {cfg.experiment.protocol_version}")

# %% [markdown]
# ## Cell 6 — Download and inspect N-MNIST
#
# First run: Tonic downloads ~180 MB to `cfg.data.data_root`.
# Subsequent runs: skips download, loads from disk.
# Expected output: "Training samples: 60000, Test samples: 10000"

# %%
from src.data.nmnist import NMNISTEventDataset, SENSOR_SIZE, binning_spec_from_config

print(f"Tonic sensor_size for N-MNIST: {SENSOR_SIZE}")
assert SENSOR_SIZE == (34, 34, 2), f"Unexpected sensor_size: {SENSOR_SIZE}"

# Protocol v2.0: events are binned by timestamp into n_time_bins ordered frames.
spec = binning_spec_from_config(cfg)
print(f"Binning spec: {spec}")

print("\nLoading training set (downloading if needed)...")
train_ds = NMNISTEventDataset(cfg.data.data_root, True, spec, representation="snn")
print("Loading test set...")
test_ds = NMNISTEventDataset(cfg.data.data_root, False, spec, representation="snn")

print(f"\nTraining samples : {len(train_ds):,}")
print(f"Test samples     : {len(test_ds):,}")

assert len(train_ds) == 60_000, f"Expected 60000 train samples, got {len(train_ds)}"
assert len(test_ds) == 10_000,  f"Expected 10000 test samples, got {len(test_ds)}"
print("\nDataset sizes: OK")

# %% [markdown]
# ## Cell 7 — Inspect a single sample (original events -> temporal bins -> aggregate)

# %%
events, label = train_ds.get_events(0)
frames, info, _ = train_ds.get_binned(0)
x_snn, _ = train_ds[0]
x_ann = x_snn.sum(dim=0)

print(f"Sample 0: label={label}")
print(f"  original events : {events.size}  dtype={events.dtype}")
print(f"  t range (µs)    : [{events['t'].min()}, {events['t'].max()}]")
print(f"  polarity counts : p=0 {(events['p']==0).sum()}, p=1 {(events['p']==1).sum()}")
print(f"  SNN tensor      : {tuple(x_snn.shape)}  (T, C, H, W) raw counts")
print(f"  events per bin  : {info.events_per_bin}")
print(f"  ANN tensor      : {tuple(x_ann.shape)}  (sum over T)")

assert x_snn.shape == (cfg.preprocess.n_time_bins, 2, 34, 34)
assert int(x_snn.sum()) + info.n_events_outside_window == events.size, "event conservation"
assert torch.isfinite(x_snn).all(), "Tensor contains NaN or Inf"
print("\nSingle-sample checks: OK")

# %% [markdown]
# ## Cell 8 — Label distribution check

# %%
import numpy as np
from tqdm import tqdm

print("Counting labels in training set (sampling 2000 items for speed)...")
sample_indices = np.linspace(0, len(train_ds) - 1, 2000, dtype=int)
labels_sample = np.array([train_ds[int(i)][1] for i in tqdm(sample_indices)])

counts = np.bincount(labels_sample, minlength=10)
print("\nLabel counts (2000-sample estimate):")
for cls, cnt in enumerate(counts):
    bar = "#" * int(cnt / counts.max() * 30)
    print(f"  Class {cls}: {cnt:4d}  {bar}")


assert set(labels_sample) == set(range(10)), "Not all classes present in sample!"
print("\nAll 10 classes present: OK")

# %% [markdown]
# ## Cell 9 — Verify train/val split

# %%
from src.data.nmnist import make_train_val_split

val_frac = 1.0 - cfg.data.train_val_split
train_sub, val_sub = make_train_val_split(
    train_ds,
    val_fraction=val_frac,
    seed=cfg.data.seed_split,
)

print(f"Split seed     : {cfg.data.seed_split} (fixed, NOT experiment seed)")
print(f"Train subset   : {len(train_sub):,} samples")
print(f"Val subset     : {len(val_sub):,} samples")
print(f"Total          : {len(train_sub) + len(val_sub):,} samples")

# No overlap
train_idx_set = set(train_sub.indices)
val_idx_set   = set(val_sub.indices)
overlap       = train_idx_set & val_idx_set
print(f"Overlap between train and val: {len(overlap)} indices  (must be 0)")
assert len(overlap) == 0, "DATA LEAKAGE: train and val sets overlap!"

# Determinism
t2, v2 = make_train_val_split(train_ds, val_frac, cfg.data.seed_split)
assert train_sub.indices == t2.indices, "Split is not deterministic!"
print("\nSplit checks: OK")

# %% [markdown]
# ## Cell 10 — Preprocessing determinism check

# %%
import torch

for idx in [0, 500, 1234]:
    f1, l1 = train_ds[idx]
    f2, l2 = train_ds[idx]
    assert torch.allclose(f1, f2), f"Frame {idx} is not deterministic!"
    assert l1 == l2, f"Label {idx} changed between calls!"

print("Preprocessing determinism: OK (same index → same tensor, 3 samples checked)")

# %% [markdown]
# ## Cell 11 — Temporal visualisation + end-to-end smoke test
#
# The v1.0 aggregate-image visualisation is removed: an aggregate alone cannot
# show that temporal structure is preserved. The smoke-test script writes
# results/event_pipeline_smoke/temporal_bins_sample.png (16 bins, both
# polarities, per-bin counts and time ranges) and smoke_report.json.

# %%
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/test_event_pipeline.py"], cwd=CLONE_DIR,
                   capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise RuntimeError("Event pipeline smoke test FAILED")

# %% [markdown]
# ## Cell 12 — Run unit tests

# %%
# pytest is run as a subprocess so its output is visible in the cell
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
    cwd=CLONE_DIR,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    raise RuntimeError("Unit tests FAILED. See output above before proceeding.")
else:
    print("All unit tests passed.")

# %% [markdown]
# ## Cell 13 — Device and DataLoader smoke test
#
# Verifies that a DataLoader can produce a batch and that tensors can be
# moved to the GPU (if available). Does not run any training.

# %%
import torch
from src.data.nmnist import make_dataloaders
from src.utils.reproducibility import set_all_seeds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

set_all_seeds(seed=0)
for rep, shape in [("snn", (cfg.preprocess.n_time_bins, 2, 34, 34)), ("ann", (2, 34, 34))]:
    loaders = make_dataloaders(cfg, experiment_seed=0, representation=rep, batch_size=32)
    for split_name, loader in loaders.items():
        xb, yb = next(iter(loader))
        xb = xb.to(DEVICE)
        assert tuple(xb.shape[1:]) == shape, f"{rep}/{split_name} batch shape {tuple(xb.shape)}"
        assert xb.min() >= 0.0 and torch.isfinite(xb).all()
        print(f"{rep}/{split_name:5s}: batch {tuple(xb.shape)} labels[0:5]={yb[:5].tolist()} device={xb.device}")

print("\nDataLoader smoke test: OK")
print("\n" + "="*60)
print("Stage 1 complete. All checks passed.")
print("Send the full output of this notebook to your implementation assistant.")
print("="*60)
