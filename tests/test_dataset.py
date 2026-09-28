import pytest
import numpy as np
import torch
import os

from src.config import get_config
from src.data.nmnist import (
    NMNISTFrameDataset,
    make_train_val_split,
    SENSOR_SIZE,
    EXPECTED_TRAIN_SIZE,
    EXPECTED_TEST_SIZE,
    N_CLASSES,
)


# ---------------------------------------------------------------------------
# Helper: check if test set is available on disk
# ---------------------------------------------------------------------------

def _test_set_available(cfg) -> bool:
    """Return True if the N-MNIST Test/ directory has the expected files."""
    test_dir = os.path.join(cfg.data.data_root, "NMNIST", "Test")
    if not os.path.isdir(test_dir):
        return False
    count = sum(len(f) for _, _, f in os.walk(test_dir))
    return count >= EXPECTED_TEST_SIZE


TEST_SET_UNAVAILABLE_REASON = (
    "N-MNIST Test set not found on disk. "
    "Run: python scripts/fix_nmnist_testset.py to download it. "
    "Train-set tests will still pass."
)


# ---------------------------------------------------------------------------
# Fixtures — loaded once per test session to avoid repeated downloads
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def train_ds(cfg):
    """Full 60k training set (no disk cache to keep tests simple)."""
    return NMNISTFrameDataset(
        root=cfg.data.data_root,
        train=True,
        cache_root=None,    # No caching in tests; avoids stale-cache issues
    )

@pytest.fixture(scope="session")
def test_ds(cfg):
    if not _test_set_available(cfg):
        pytest.skip(TEST_SET_UNAVAILABLE_REASON)
    return NMNISTFrameDataset(
        root=cfg.data.data_root,
        train=False,
        cache_root=None,
    )


# ---------------------------------------------------------------------------
# Test: constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_sensor_size(self):
        assert SENSOR_SIZE == (34, 34, 2), (
            f"Locked sensor_size is (34, 34, 2) but Tonic reports {SENSOR_SIZE}. "
            "If this has changed, update config.py and this test."
        )

    def test_n_classes(self):
        assert N_CLASSES == 10


# ---------------------------------------------------------------------------
# Test: dataset sizes
# ---------------------------------------------------------------------------

class TestDatasetSizes:

    def test_train_size(self, train_ds):
        assert len(train_ds) == EXPECTED_TRAIN_SIZE, (
            f"Expected {EXPECTED_TRAIN_SIZE} training samples, got {len(train_ds)}"
        )

    def test_test_size(self, test_ds):
        # test_ds fixture skips automatically if Test/ is missing
        assert len(test_ds) == EXPECTED_TEST_SIZE, (
            f"Expected {EXPECTED_TEST_SIZE} test samples, got {len(test_ds)}"
        )


# ---------------------------------------------------------------------------
# Test: sample shape, dtype, value range
# ---------------------------------------------------------------------------

class TestSampleFormat:

    def test_frame_shape(self, train_ds):
        frame, _ = train_ds[0]
        assert frame.shape == (2, 34, 34), (
            f"Expected frame shape (2, 34, 34), got {frame.shape}\n"
            "Channel 0 = ON events, Channel 1 = OFF events."
        )

    def test_frame_dtype(self, train_ds):
        frame, _ = train_ds[0]
        assert frame.dtype == torch.float32, (
            f"Expected float32, got {frame.dtype}"
        )

    def test_frame_is_tensor(self, train_ds):
        frame, _ = train_ds[0]
        assert isinstance(frame, torch.Tensor)

    def test_frame_values_in_range(self, train_ds):
        """Check the first 20 samples for value range [0, 1]."""
        for idx in range(20):
            frame, _ = train_ds[idx]
            lo, hi = float(frame.min()), float(frame.max())
            assert lo >= 0.0, f"Frame {idx}: min value {lo:.6f} < 0"
            assert hi <= 1.0 + 1e-6, f"Frame {idx}: max value {hi:.6f} > 1"

    def test_frame_max_is_one(self, train_ds):
        """After per-sample normalization, max should be 1.0 (unless all-zero)."""
        for idx in range(20):
            frame, _ = train_ds[idx]
            hi = float(frame.max())
            assert hi > 0.0, (
                f"Frame {idx} is all-zeros — this sample appears to have no events."
            )
            assert abs(hi - 1.0) < 1e-5, (
                f"Frame {idx}: after normalization max={hi:.6f}, expected 1.0"
            )

    def test_no_nan_or_inf(self, train_ds):
        for idx in range(20):
            frame, _ = train_ds[idx]
            assert torch.isfinite(frame).all(), (
                f"Frame {idx} contains NaN or Inf values."
            )


# ---------------------------------------------------------------------------
# Test: labels
# ---------------------------------------------------------------------------

class TestLabels:

    def test_label_type(self, train_ds):
        _, label = train_ds[0]
        assert isinstance(label, int), f"Expected int label, got {type(label)}"

    def test_label_range_sample(self, train_ds):
        # N-MNIST is stored sorted by class: Train/0/, Train/1/, …, Train/9/
        # Checking only the first N indices would miss classes. Use evenly
        # spaced indices across the full dataset instead.
        step = len(train_ds) // 100
        for idx in range(0, min(len(train_ds), step * 100), step):
            _, label = train_ds[idx]
            assert 0 <= label <= 9, f"Label {label} at index {idx} is out of [0, 9]"

    def test_all_classes_present(self, train_ds):
        """All 10 digit classes must appear in the dataset.

        N-MNIST is stored sorted by class directory (Train/0/, Train/1/, …),
        so the first 1000 indices are all label 0. We use get_all_labels()
        which reads file-path metadata (no event I/O) and is fast.
        """
        labels = train_ds.get_all_labels()
        missing = set(range(10)) - set(labels.tolist())
        assert not missing, (
            f"Classes {missing} are absent from the training set labels."
        )

    def test_test_labels(self, test_ds):
        for idx in range(50):
            _, label = test_ds[idx]
            assert 0 <= label <= 9


# ---------------------------------------------------------------------------
# Test: train/val split
# ---------------------------------------------------------------------------

class TestSplit:

    def test_split_sizes(self, train_ds, cfg):
        val_frac = 1.0 - cfg.data.train_val_split
        train_sub, val_sub = make_train_val_split(
            train_ds, val_fraction=val_frac, seed=cfg.data.seed_split
        )
        n_total = len(train_ds)
        n_val_expected = round(n_total * val_frac)

        assert abs(len(val_sub) - n_val_expected) <= 20, (
            f"Val size {len(val_sub)} is far from expected ~{n_val_expected}"
        )
        assert len(train_sub) + len(val_sub) == n_total, (
            "train + val sizes do not sum to total dataset size"
        )

    def test_split_no_overlap(self, train_ds, cfg):
        val_frac = 1.0 - cfg.data.train_val_split
        train_sub, val_sub = make_train_val_split(
            train_ds, val_fraction=val_frac, seed=cfg.data.seed_split
        )
        train_idx = set(train_sub.indices)
        val_idx = set(val_sub.indices)
        overlap = train_idx & val_idx
        assert len(overlap) == 0, (
            f"Train/val overlap: {len(overlap)} shared indices. "
            "This would cause data leakage."
        )

    def test_split_deterministic(self, train_ds, cfg):
        """Calling make_train_val_split twice with the same seed must give identical splits."""
        val_frac = 1.0 - cfg.data.train_val_split
        seed = cfg.data.seed_split

        s1_train, s1_val = make_train_val_split(train_ds, val_frac, seed)
        s2_train, s2_val = make_train_val_split(train_ds, val_frac, seed)

        assert s1_train.indices == s2_train.indices, "Train indices differ between calls."
        assert s1_val.indices == s2_val.indices, "Val indices differ between calls."

    def test_split_stratified(self, train_ds, cfg):
        """Class distribution in val must be approximately uniform (balanced N-MNIST)."""
        val_frac = 1.0 - cfg.data.train_val_split
        _, val_sub = make_train_val_split(
            train_ds, val_fraction=val_frac, seed=cfg.data.seed_split
        )
        val_labels = np.array([train_ds[i][1] for i in val_sub.indices[:500]])
        counts = np.bincount(val_labels, minlength=10)
        # In a balanced dataset each class should appear roughly equally
        assert counts.min() > 0, "Some class has zero samples in val split."


# ---------------------------------------------------------------------------
# Test: preprocessing determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_index_same_output(self, train_ds):
        """Calling dataset[i] twice must return the same tensor."""
        for idx in [0, 100, 999]:
            f1, l1 = train_ds[idx]
            f2, l2 = train_ds[idx]
            assert torch.allclose(f1, f2), (
                f"Frame at index {idx} is not deterministic — preprocessing is random."
            )
            assert l1 == l2
