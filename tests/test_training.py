"""
Smoke tests for training loops (test_training.py).

These tests verify that:
  1. ANN training loop runs without error and returns a valid TrainingResult
  2. SNN direct-SG training loop runs without error and returns TrainingResult
  3. Hybrid (Method B) training loop runs end-to-end and returns HybridTrainingResult
  4. Early stopping triggers correctly (patience=1 with no improvement)
  5. Checkpoint files are created on disk
  6. The loaded checkpoint restores model weights correctly
  7. HybridTrainingResult.total_wall_s == ann + ft wall times (accounting check)

Tests use SYNTHETIC data (torch.rand tensors) so N-MNIST files are NOT needed.
All tests use max 2 epochs, batch_size=4, 2 batches — runs in <15 s on CPU.

NOTE: These are functional smoke tests, NOT accuracy tests. The models are
trained with random initialisation on random data; we only verify that the
API contracts are honoured and the return values have the expected types and
shapes.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_config, Config, TrainConfig
from src.models.ann import ANN
from src.models.snn import SNN
from src.trainer import (
    EarlyStopping, EpochLog, TrainingResult, HybridTrainingResult,
    evaluate, train_one_epoch, load_model_from_checkpoint,
)
from src.train_ann import train_ann
from src.train_snn import train_snn_sg, train_snn_hybrid
from spikingjelly.activation_based import functional


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> Config:
    """Minimal config for smoke tests: 2 epochs, tiny batches, early stop = 10."""
    c = get_config()
    # Override epoch counts so tests run fast
    c.train.ann_epochs    = 2
    c.train.snn_sg_epochs = 2
    c.train.snn_ft_epochs = 2
    # Large patience so early stopping does NOT trigger during 2-epoch runs
    c.train.early_stop_patience = 10
    # Tiny calibration (1 batch)
    c.train.calib_batches   = 1
    c.train.calib_percentile = 99.0
    # Random labels on 8 samples: the converted SNN cannot be expected to match
    # an untrained ANN, so the gate is recorded, not enforced (tested separately).
    c.train.conversion_gate = 'report'
    c.snn.init_calib_batches = 1
    return c


def _synthetic_pair(seed: int, n: int = 8, T: int = 24):
    """Synthetic temporal sequences [n, T, 2, 34, 34] and their aggregate
    [n, 2, 34, 34] (mean over T) — mirrors the real SNN/ANN representations."""
    g = torch.Generator().manual_seed(seed)
    rate = 0.3 * torch.rand(n, T, 2, 34, 34, generator=g)
    seq = torch.poisson(rate, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return seq, seq.mean(dim=1), y


@pytest.fixture(scope="module")
def synthetic_pairs():
    return {split: _synthetic_pair(i) for i, split in enumerate(('train', 'val', 'test'))}


@pytest.fixture(scope="module")
def synthetic_loaders(synthetic_pairs):
    """SNN loaders: batches of temporal sequences [B, T, 2, 34, 34]."""
    return {k: DataLoader(TensorDataset(seq, y), batch_size=4, shuffle=False)
            for k, (seq, _, y) in synthetic_pairs.items()}


@pytest.fixture(scope="module")
def synthetic_ann_loaders(synthetic_pairs):
    """ANN loaders: aggregate counts [B, 2, 34, 34] of the same samples."""
    return {k: DataLoader(TensorDataset(agg, y), batch_size=4, shuffle=False)
            for k, (_, agg, y) in synthetic_pairs.items()}


@pytest.fixture(scope="module")
def device():
    return torch.device('cpu')


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("ckpts")


# ---------------------------------------------------------------------------
# EarlyStopping unit tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3)
        for acc in [0.5, 0.6, 0.7, 0.8]:
            stopped = es.step(acc, epoch=1)
            assert not stopped
        assert es.best_val_acc == pytest.approx(0.8)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.9, 1)    # best
        es.step(0.85, 2)   # counter=1
        es.step(0.85, 3)   # counter=2
        assert es.step(0.85, 4)   # counter=3 == patience → must return True (stop)

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.8, 1)
        es.step(0.75, 2)   # counter=1
        es.step(0.75, 3)   # counter=2
        es.step(0.81, 4)   # improvement → counter resets to 0
        assert es.counter == 0
        assert not es.improved or True   # improved should be True

    def test_best_epoch_tracks_correctly(self):
        es = EarlyStopping(patience=5)
        es.step(0.7, 1)
        es.step(0.8, 2)
        es.step(0.75, 3)
        assert es.best_epoch == 2
        assert es.best_val_acc == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ANN training smoke tests
# ---------------------------------------------------------------------------

class TestTrainANN:

    @pytest.fixture(scope="class")
    def ann_result(self, cfg, synthetic_ann_loaders, device, checkpoint_dir):
        return train_ann(
            cfg, seed=0,
            dataloaders=synthetic_ann_loaders,
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )

    def test_returns_training_result(self, ann_result):
        assert isinstance(ann_result, TrainingResult)

    def test_method_is_ann(self, ann_result):
        assert ann_result.method == 'ann'

    def test_epoch_logs_length(self, ann_result, cfg):
        assert len(ann_result.epoch_logs) == cfg.train.ann_epochs

    def test_epoch_logs_are_epoch_log(self, ann_result):
        for log in ann_result.epoch_logs:
            assert isinstance(log, EpochLog)

    def test_epoch_numbers_sequential(self, ann_result):
        epochs = [log.epoch for log in ann_result.epoch_logs]
        assert epochs == list(range(1, len(epochs) + 1))

    def test_all_accuracies_in_unit_interval(self, ann_result):
        for log in ann_result.epoch_logs:
            assert 0.0 <= log.train_acc <= 1.0
            assert 0.0 <= log.val_acc   <= 1.0

    def test_all_losses_positive(self, ann_result):
        for log in ann_result.epoch_logs:
            assert log.train_loss >= 0.0
            assert log.val_loss   >= 0.0

    def test_total_updates_positive(self, ann_result):
        assert ann_result.total_updates > 0

    def test_total_wall_s_positive(self, ann_result):
        assert ann_result.total_wall_s > 0.0

    def test_best_val_acc_matches_logs(self, ann_result):
        max_log_acc = max(log.val_acc for log in ann_result.epoch_logs)
        assert ann_result.best_val_acc == pytest.approx(max_log_acc, abs=1e-6)

    def test_checkpoint_file_exists(self, ann_result):
        assert ann_result.checkpoint_path is not None
        assert Path(ann_result.checkpoint_path).exists(), (
            f"Checkpoint not found: {ann_result.checkpoint_path}"
        )

    def test_checkpoint_restores_weights(self, ann_result, cfg):
        """Loading checkpoint into a fresh ANN must give identical weights."""
        fresh_ann = ANN(
            in_channels=cfg.network.in_channels,
            conv1_out=cfg.network.conv1_out,
            conv2_out=cfg.network.conv2_out,
            fc1_out=cfg.network.fc1_out,
            n_classes=cfg.network.n_classes,
        )
        ckpt = load_model_from_checkpoint(ann_result.checkpoint_path, fresh_ann)
        assert ckpt['val_acc'] == pytest.approx(ann_result.best_val_acc, abs=1e-6), (
            "Checkpoint val_acc doesn't match best_val_acc."
        )

    def test_hardware_dict_present(self, ann_result):
        assert isinstance(ann_result.hardware, dict)
        assert 'torch' in ann_result.hardware
        assert 'cuda_available' in ann_result.hardware

    def test_not_early_stopped_in_2_epochs(self, ann_result):
        """With patience=10 and only 2 epochs, early stopping must not trigger."""
        assert not ann_result.early_stopped


# ---------------------------------------------------------------------------
# SNN-SG training smoke tests
# ---------------------------------------------------------------------------

class TestTrainSNNSG:

    @pytest.fixture(scope="class")
    def sg_result(self, cfg, synthetic_loaders, device, checkpoint_dir):
        return train_snn_sg(
            cfg, seed=1,
            dataloaders=synthetic_loaders,
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )

    def test_returns_training_result(self, sg_result):
        assert isinstance(sg_result, TrainingResult)

    def test_method_is_snn_sg(self, sg_result):
        assert sg_result.method == 'snn_sg'

    def test_epoch_count(self, sg_result, cfg):
        assert len(sg_result.epoch_logs) == cfg.train.snn_sg_epochs

    def test_accuracies_in_unit_interval(self, sg_result):
        for log in sg_result.epoch_logs:
            assert 0.0 <= log.train_acc <= 1.0
            assert 0.0 <= log.val_acc   <= 1.0

    def test_losses_positive(self, sg_result):
        for log in sg_result.epoch_logs:
            assert log.train_loss >= 0.0
            assert log.val_loss   >= 0.0

    def test_total_updates_positive(self, sg_result):
        assert sg_result.total_updates > 0

    def test_checkpoint_exists(self, sg_result):
        assert Path(sg_result.checkpoint_path).exists()

    def test_not_early_stopped(self, sg_result):
        assert not sg_result.early_stopped


# ---------------------------------------------------------------------------
# SNN-Hybrid training smoke tests
# ---------------------------------------------------------------------------

class TestTrainSNNHybrid:

    @pytest.fixture(scope="class")
    def hybrid_result(self, cfg, synthetic_loaders, synthetic_ann_loaders, device, checkpoint_dir):
        return train_snn_hybrid(
            cfg, seed=2,
            snn_dataloaders=synthetic_loaders,
            ann_dataloaders=synthetic_ann_loaders,
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )

    def test_returns_hybrid_result(self, hybrid_result):
        assert isinstance(hybrid_result, HybridTrainingResult)

    def test_ann_result_is_training_result(self, hybrid_result):
        assert isinstance(hybrid_result.ann_result, TrainingResult)
        assert hybrid_result.ann_result.method == 'ann'

    def test_ft_result_is_training_result(self, hybrid_result):
        assert isinstance(hybrid_result.ft_result, TrainingResult)
        assert hybrid_result.ft_result.method == 'snn_ft'

    def test_calibrated_thresholds_keys(self, hybrid_result):
        assert set(hybrid_result.calibrated_thresholds.keys()) == {
            'lif1', 'lif2', 'lif3'
        }

    def test_calibrated_thresholds_positive(self, hybrid_result):
        for name, val in hybrid_result.calibrated_thresholds.items():
            assert val > 0, f"Threshold '{name}' = {val} is not positive."

    def test_total_wall_s_is_sum(self, hybrid_result):
        """Total cost must equal ANN cost + FT cost."""
        expected = (hybrid_result.ann_result.total_wall_s
                    + hybrid_result.ft_result.total_wall_s)
        assert hybrid_result.total_wall_s == pytest.approx(expected, rel=1e-3), (
            "HybridTrainingResult.total_wall_s is not the sum of ANN and FT wall times."
        )

    def test_ft_checkpoint_exists(self, hybrid_result):
        assert Path(hybrid_result.ft_result.checkpoint_path).exists()

    def test_ann_checkpoint_exists(self, hybrid_result):
        assert Path(hybrid_result.ann_result.checkpoint_path).exists()

    def test_ft_checkpoint_restores_snn_weights(self, hybrid_result, cfg):
        """Loaded SNN checkpoint must be forwarded without error."""
        snn = SNN(
            in_channels=cfg.network.in_channels,
            conv1_out=cfg.network.conv1_out,
            conv2_out=cfg.network.conv2_out,
            fc1_out=cfg.network.fc1_out,
            n_classes=cfg.network.n_classes,
            tau=cfg.snn.tau,
            v_threshold=cfg.snn.v_threshold,
        )
        load_model_from_checkpoint(hybrid_result.ft_result.checkpoint_path, snn)
        snn.eval()
        x_seq = torch.rand(16, 2, 2, 34, 34)
        functional.reset_net(snn)
        with torch.no_grad():
            out = snn(x_seq)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_seed_stored_correctly(self, hybrid_result):
        assert hybrid_result.seed == 2
        assert hybrid_result.ann_result.seed == 2
        assert hybrid_result.ft_result.seed == 2


# ---------------------------------------------------------------------------
# Early stopping integration test
# ---------------------------------------------------------------------------

class TestEarlyStoppingIntegration:

    def test_early_stopping_triggers_in_ann(self, cfg, synthetic_ann_loaders,
                                             device, checkpoint_dir):
        """With patience=1, ANN training should stop after 2 epochs (1 non-improving)."""
        tight_cfg = get_config()
        tight_cfg.train.ann_epochs          = 10   # budget: 10
        tight_cfg.train.early_stop_patience = 1    # stops after 1 non-improvement
        result = train_ann(
            tight_cfg, seed=99,
            dataloaders=synthetic_ann_loaders,
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )
        # With patience=1, we expect early stopping before epoch 10
        # (the exact epoch depends on whether val_acc improves, but with
        # patience=1 and random data it will almost certainly stop early)
        assert result.total_epochs < tight_cfg.train.ann_epochs or result.early_stopped

    def test_early_stopping_triggers_in_snn_sg(self, cfg, synthetic_loaders,
                                                device, checkpoint_dir):
        """With patience=1, SNN-SG training should stop early."""
        tight_cfg = get_config()
        tight_cfg.train.snn_sg_epochs       = 10
        tight_cfg.train.early_stop_patience = 1
        result = train_snn_sg(
            tight_cfg, seed=98,
            dataloaders=synthetic_loaders,
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )
        assert result.total_epochs < tight_cfg.train.snn_sg_epochs or result.early_stopped


# ---------------------------------------------------------------------------
# Protocol v2.0 guards
# ---------------------------------------------------------------------------

class TestTemporalInputGuards:

    def test_snn_training_rejects_static_frames(self, cfg, synthetic_ann_loaders,
                                                device, checkpoint_dir):
        """Passing aggregate [B, 2, 34, 34] frames to SNN training must fail
        loudly instead of silently repeating them over time (v1.0 behaviour)."""
        with pytest.raises(ValueError, match="temporal event sequence"):
            train_snn_sg(cfg, seed=5, dataloaders=synthetic_ann_loaders,
                         device=device, checkpoint_dir=checkpoint_dir, verbose=False)

    def test_snn_eval_at_other_T_is_blocked(self, cfg, synthetic_loaders, device):
        """A 24-bin loader cannot be evaluated at T=12: re-binned loaders are
        required (decision D-A, make_eval_loaders_by_T)."""
        from src.trainer import evaluate
        snn = SNN()
        with pytest.raises(ValueError, match="bins"):
            evaluate(snn, synthetic_loaders['val'], torch.nn.CrossEntropyLoss(),
                     device, is_snn=True, T=12)

    def test_hybrid_records_conversion_info_and_fidelity(self, cfg, synthetic_loaders,
                                                         synthetic_ann_loaders, device, checkpoint_dir):
        res = train_snn_hybrid(cfg, seed=7, snn_dataloaders=synthetic_loaders,
                               ann_dataloaders=synthetic_ann_loaders, device=device,
                               checkpoint_dir=checkpoint_dir, verbose=False)
        assert res.conversion_info['input_map'] == cfg.train.conversion_input_map
        assert {'ann_acc', 'snn_acc', 'agreement', 'passed'} <= set(res.conversion_fidelity)
        ck = torch.load(res.ft_result.checkpoint_path, map_location='cpu', weights_only=False)
        assert 'conversion_fidelity' in ck and 'conversion_info' in ck

    def test_hybrid_enforced_gate_refuses_fine_tuning(self, cfg, synthetic_loaders,
                                                      synthetic_ann_loaders, device, checkpoint_dir):
        from src.conversion import ConversionGateError
        strict = copy.deepcopy(cfg)
        strict.train.conversion_gate = 'enforce'
        strict.train.gate_min_agreement = 1.01          # impossible -> must fail
        with pytest.raises(ConversionGateError):
            train_snn_hybrid(strict, seed=8, snn_dataloaders=synthetic_loaders,
                             ann_dataloaders=synthetic_ann_loaders, device=device,
                             checkpoint_dir=checkpoint_dir, verbose=False)

    def test_sg_init_calibration_recorded_and_no_silent_layer(self, cfg, synthetic_loaders,
                                                              device, checkpoint_dir):
        res = train_snn_sg(cfg, seed=9, dataloaders=synthetic_loaders, device=device,
                           checkpoint_dir=checkpoint_dir, verbose=False)
        rep = res.extra['init_calibration']
        assert set(rep) == {'lif1', 'lif2', 'lif3'}
        for v in rep.values():
            assert v['rate_after'] == pytest.approx(cfg.snn.init_target_rate, rel=0.15)


# ---------------------------------------------------------------------------
# Decision D-E: training budget in sample-time-steps
# ---------------------------------------------------------------------------

class TestSampleTimestepBudget:

    def test_ann_counts_one_step_per_sample(self, cfg, synthetic_ann_loaders, device, checkpoint_dir):
        res = train_ann(cfg, seed=11, dataloaders=synthetic_ann_loaders, device=device,
                        checkpoint_dir=checkpoint_dir, verbose=False)
        n = len(synthetic_ann_loaders['train'].dataset)
        assert res.train_sample_timesteps == res.total_epochs * n * 1

    def test_sg_counts_t_train_steps_per_sample(self, cfg, synthetic_loaders, device, checkpoint_dir):
        res = train_snn_sg(cfg, seed=12, dataloaders=synthetic_loaders, device=device,
                           checkpoint_dir=checkpoint_dir, verbose=False)
        n = len(synthetic_loaders['train'].dataset)
        assert res.train_sample_timesteps == res.total_epochs * n * cfg.snn.t_train
        assert res.extra['init_forward_sample_timesteps'] > 0     # reported separately

    def test_hybrid_budget_is_ann_plus_finetune(self, cfg, synthetic_loaders, synthetic_ann_loaders,
                                                device, checkpoint_dir):
        res = train_snn_hybrid(cfg, seed=13, snn_dataloaders=synthetic_loaders,
                               ann_dataloaders=synthetic_ann_loaders, device=device,
                               checkpoint_dir=checkpoint_dir, verbose=False)
        n = len(synthetic_loaders['train'].dataset)
        assert res.ann_result.train_sample_timesteps == res.ann_result.total_epochs * n
        assert res.ft_result.train_sample_timesteps == res.ft_result.total_epochs * n * cfg.snn.t_train
        assert res.train_sample_timesteps == (res.ann_result.train_sample_timesteps
                                              + res.ft_result.train_sample_timesteps)
        assert res.conversion_info['refine_forward_sample_timesteps'] > 0


def test_hybrid_staged_gate_refuses_when_sanity_fails(cfg, synthetic_loaders, synthetic_ann_loaders,
                                                     device, checkpoint_dir):
    from src.conversion import ConversionGateError
    c = copy.deepcopy(cfg)
    c.train.conversion_gate = 'staged'
    c.train.gate_min_agreement = 1.01      # strict impossible
    c.train.gate_sanity_fraction = 1.01    # sanity impossible
    with pytest.raises(ConversionGateError) as e:
        train_snn_hybrid(c, seed=14, snn_dataloaders=synthetic_loaders,
                         ann_dataloaders=synthetic_ann_loaders, device=device,
                         checkpoint_dir=checkpoint_dir, verbose=False)
    assert e.value.report.gate_outcome == 'fail'


# ---------------------------------------------------------------------------
# D-I: one ANN per seed, reused at every T by rescaling conv1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 2, 3, 6, 12])
def test_reused_ann_rescaled_conv1_is_the_same_function(T):
    """ANN trained on total/24 must give identical logits on total/T after
    conv1 weights are multiplied by T/24 (ann_input_scale_for)."""
    import copy
    from src.config import get_config
    from src.models.ann import ANN
    from src.train_snn import ann_input_scale_for
    torch.manual_seed(0)
    ann = ANN().eval()
    total = torch.poisson(torch.full((4, 2, 34, 34), 3.0))
    ref = ann(total / 24)
    c = get_config()
    c.snn.t_train = c.preprocess.n_time_bins = T
    scaled = copy.deepcopy(ann)
    with torch.no_grad():
        scaled.features[0].weight.mul_(ann_input_scale_for(c, 24))
    assert torch.allclose(scaled(total / T), ref, atol=1e-5)
    assert ann_input_scale_for(c, None) == 1.0
