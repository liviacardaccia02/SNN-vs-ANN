#!/usr/bin/env python3
"""
Diagnostic for decisions D-B (conversion + fidelity gate) and D-F (init).

NOT an experiment: small subsets, short training, one seed. It answers two
engineering questions on real N-MNIST before the pilot:

  F) Does the direct-SG SNN start learning with D-F initialisation, compared
     with PyTorch default initialisation (same seed, same data, same steps)?
  B) After a briefly trained ANN is converted with the D-B method, does the
     converted SNN (no fine-tuning) reproduce the ANN on validation data,
     i.e. does it pass the gate? Ablations: input_map "fitted" (constant-input
     curve fit only), "offset" (X = 1 + u), "none" (plain normalisation).

Usage:
    python scripts/check_conversion_and_init.py --n-train 6000 --n-val 1000
Outputs results/conversion_init_check/report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

from src.config import get_config, preprocessing_signature  # noqa: E402
from src.conversion import convert, conversion_fidelity  # noqa: E402
from src.data.nmnist import _make_ds, make_train_val_split  # noqa: E402
from src.init_calibration import calibrate_initial_firing, calibration_batches  # noqa: E402
from src.models.ann import ANN  # noqa: E402
from src.models.snn import SNN  # noqa: E402
from src.trainer import get_hardware_info, to_time_major  # noqa: E402
from src.utils.reproducibility import make_run_id, set_all_seeds  # noqa: E402


def stratified(indices: list[int], labels: np.ndarray, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices)
    out = []
    for c in range(10):
        pool = idx[labels[idx] == c]
        out.extend(rng.choice(pool, n // 10, replace=False).tolist())
    return sorted(out)


def snn_eval(snn: SNN, loader: DataLoader, device) -> float:
    snn.eval(); correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            functional.reset_net(snn)
            correct += int((snn(to_time_major(x.to(device))).argmax(1) == y.to(device)).sum()); n += len(y)
    functional.reset_net(snn)
    return correct / n


def train_snn(snn: SNN, train_dl, val_dl, epochs: int, lr: float, device, clip: float) -> dict:
    opt = torch.optim.Adam(snn.parameters(), lr=lr)
    losses, val = [], []
    for _ in range(epochs):
        snn.train()
        for x, y in train_dl:
            functional.reset_net(snn)
            loss = nn.functional.cross_entropy(snn(to_time_major(x.to(device))), y.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(snn.parameters(), clip); opt.step()
            losses.append(float(loss))
        val.append(snn_eval(snn, val_dl, device))
    k = max(1, len(losses) // 10)
    return {"loss_first_10pct": float(np.mean(losses[:k])), "loss_last_10pct": float(np.mean(losses[-k:])),
            "val_acc_per_epoch": val, "n_updates": len(losses)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--snn-epochs", type=int, default=1)
    ap.add_argument("--ann-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-f", action="store_true", help="only run the conversion (B) part")
    ap.add_argument("--skip-b", action="store_true", help="only run the initialisation (F) part")
    ap.add_argument("--init-rates", default=None,
                    help="comma-separated target rates for a D-F selection sweep, e.g. 0.01,0.02,0.05,0.1,0.2 "
                         "(runs only the sweep; selection on VALIDATION data)")
    ap.add_argument("--sweep-seeds", default="0,1")
    ap.add_argument("--variants", default="data_fitted,fitted,offset,none")
    ap.add_argument("--ft-epochs", type=int, default=0,
                    help="fine-tune the data_fitted conversion for this many epochs (diagnostic)")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "conversion_init_check"))
    args = ap.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {"run_id": make_run_id("conv_init_check", args.seed), "args": vars(args),
              "protocol_version": cfg.experiment.protocol_version,
              "preprocessing_signature": preprocessing_signature(cfg)}
    t0 = time.time()

    # Same recordings for every representation: indices into the official train split.
    full_snn = _make_ds(cfg, True, "snn", download=False)
    full_ann = _make_ds(cfg, True, "ann", download=False)
    labels = np.asarray(full_snn.targets)
    tr, va = make_train_val_split(full_snn, 1.0 - cfg.data.train_val_split, cfg.data.seed_split)
    tr_idx = stratified(tr.indices, labels, args.n_train, args.seed)
    va_idx = stratified(va.indices, labels, args.n_val, args.seed + 1)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers)
    g = lambda: torch.Generator().manual_seed(args.seed)  # noqa: E731
    snn_train = DataLoader(Subset(full_snn, tr_idx), shuffle=True, generator=g(), **kw)
    snn_val = DataLoader(Subset(full_snn, va_idx), shuffle=False, **kw)
    ann_train = DataLoader(Subset(full_ann, tr_idx), shuffle=True, generator=g(), **kw)
    ann_val = DataLoader(Subset(full_ann, va_idx), shuffle=False, **kw)
    print(f"train subset {len(tr_idx)}, val subset {len(va_idx)} (validation split only), device {device}")

    def new_snn() -> SNN:
        torch.manual_seed(args.seed)
        return SNN(tau=cfg.snn.tau, v_threshold=cfg.snn.v_threshold).to(device)

    # ---------------- F-sweep: select the init target rate ----------------
    if args.init_rates:
        rates = [float(r) for r in args.init_rates.split(",")]
        seeds = [int(x) for x in args.sweep_seeds.split(",")]
        print(f"\n[F-sweep] target rates {rates}, seeds {seeds}; selection rule: highest mean "
              f"validation accuracy, ties within 1 pp -> lower rate")
        sweep = {}
        for r in rates:
            accs, finals = [], []
            for sd in seeds:
                set_all_seeds(sd)
                torch.manual_seed(sd)
                snn = SNN(tau=cfg.snn.tau, v_threshold=cfg.snn.v_threshold).to(device)
                seqs = calibration_batches(snn_train, cfg.snn.init_calib_batches, device)
                cal = calibrate_initial_firing(snn, seqs, r)
                g = torch.Generator().manual_seed(sd)
                dl = DataLoader(snn_train.dataset, shuffle=True, generator=g, batch_size=args.batch_size,
                                num_workers=args.workers)
                res = train_snn(snn, dl, snn_val, args.snn_epochs, cfg.train.snn_sg_lr, device,
                                cfg.train.max_grad_norm)
                accs.append(res["val_acc_per_epoch"][-1]); finals.append(res["loss_last_10pct"])
            sweep[r] = {"val_acc_per_seed": accs, "mean_val_acc": float(np.mean(accs)),
                        "loss_last_10pct_per_seed": finals}
            print(f"  rate {r:5.2f}: val acc per seed {[round(a, 4) for a in accs]} -> mean {np.mean(accs):.4f}")
        best = max(sweep.values(), key=lambda d: d["mean_val_acc"])["mean_val_acc"]
        chosen = min(r for r, d in sweep.items() if d["mean_val_acc"] >= best - 0.01)
        print(f"  selected target rate: {chosen} (best mean {best:.4f}; rule: within 1 pp -> lowest rate)")
        report["F_sweep"] = {"rates": {str(k): v for k, v in sweep.items()}, "seeds": seeds,
                             "selected_rate": chosen, "rule": "max mean val acc; ties within 1 pp -> lower rate",
                             "n_updates_per_run": len(snn_train), "epochs": args.snn_epochs}
        report["environment"] = get_hardware_info()
        report["wall_s"] = round(time.time() - t0, 1)
        (out_dir / "report_F_sweep.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport: {(out_dir / 'report_F_sweep.json').resolve()}")
        return 0

    # ---------------- F: init comparison ----------------
    report["F"] = {}
    if not args.skip_f:
        print("\n[F] direct-SG: default init vs D-F firing-rate init")
    for name in (() if args.skip_f else ("default_init", "firing_rate_init")):
        set_all_seeds(args.seed)
        snn = new_snn()
        if name == "firing_rate_init":
            seqs = calibration_batches(snn_train, cfg.snn.init_calib_batches, device)
            report["F"]["calibration"] = calibrate_initial_firing(snn, seqs, cfg.snn.init_target_rate)
        t = time.time()
        res = train_snn(snn, snn_train, snn_val, args.snn_epochs, cfg.train.snn_sg_lr, device,
                        cfg.train.max_grad_norm)
        res["wall_s"] = round(time.time() - t, 1)
        report["F"][name] = res
        print(f"  {name:17s}: loss {res['loss_first_10pct']:.3f} -> {res['loss_last_10pct']:.3f}, "
              f"val acc {res['val_acc_per_epoch']} ({res['wall_s']} s)")

    # ---------------- B: ANN -> conversion -> gate ----------------
    if args.skip_b:
        report["environment"] = get_hardware_info()
        report["wall_s"] = round(time.time() - t0, 1)
        (out_dir / "report_F.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport: {(out_dir / 'report_F.json').resolve()}")
        return 0
    print("\n[B] ANN training on mean-over-bins input, conversion, fidelity gate")
    set_all_seeds(args.seed)
    ann = ANN().to(device)
    opt = torch.optim.Adam(ann.parameters(), lr=cfg.train.ann_lr, weight_decay=cfg.train.ann_weight_decay)
    for ep in range(args.ann_epochs):
        ann.train()
        for x, y in ann_train:
            loss = nn.functional.cross_entropy(ann(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    ann.eval()
    report["B"] = {"ann_epochs": args.ann_epochs, "variants": {}}
    variants = args.variants.split(",")
    for variant in variants:
        snn = new_snn()
        t = time.time()
        snn, info = convert(ann, snn, ann_train, cfg.train.calib_percentile, device,
                            n_batches=cfg.train.calib_batches, v_threshold=cfg.snn.v_threshold,
                            input_map=variant, T=cfg.snn.t_train, snn_calibration_loader=snn_train,
                            refine_batches=cfg.train.conversion_refine_batches)
        info["conversion_wall_s"] = round(time.time() - t, 1)
        fid = conversion_fidelity(ann, snn, ann_val, snn_val, info["lambdas"], device,
                                  max_acc_drop=cfg.train.gate_max_acc_drop,
                                  min_agreement=cfg.train.gate_min_agreement,
                                  sanity_fraction=cfg.train.gate_sanity_fraction)
        from src.conversion import gate_decision
        gate_decision(fid, "staged")
        report["B"]["variants"][variant] = {"conversion_info": info, "fidelity": fid.to_dict()}
        if variant == "data_fitted" and args.ft_epochs > 0:
            t = time.time()
            ft = train_snn(snn, snn_train, snn_val, args.ft_epochs, cfg.train.snn_ft_lr, device,
                           cfg.train.max_grad_norm)
            ft["wall_s"] = round(time.time() - t, 1)
            report["B"]["variants"][variant]["fine_tune"] = ft
            print(f"  data_fitted + fine-tune {args.ft_epochs} ep (lr {cfg.train.snn_ft_lr}): "
                  f"loss {ft['loss_first_10pct']:.3f} -> {ft['loss_last_10pct']:.3f}, val acc {ft['val_acc_per_epoch']}")
        print(f"  input_map={variant:11s}: staged gate {fid.gate_outcome} (strict {'PASS' if fid.passed else 'FAIL'}, "
              f"sanity >= {fid.sanity_threshold:.3f} {'PASS' if fid.sanity_passed else 'FAIL'}) | ANN val {fid.ann_acc:.4f} | "
              f"SNN(T={fid.T}) val {fid.snn_acc:.4f} | agreement {fid.agreement:.4f} | "
              f"rate corr {({k: round(v, 3) for k, v in fid.layer_rate_correlation.items()})} | "
              f"rates {({k: round(v, 3) for k, v in fid.layer_mean_rate.items()})} vs target "
              f"{({k: round(v, 3) for k, v in fid.layer_mean_target_rate.items()})}")

    report["environment"] = get_hardware_info()
    report["wall_s"] = round(time.time() - t0, 1)
    name = "report_B.json" if args.skip_f else "report.json"
    (out_dir / name).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport: {(out_dir / name).resolve()}  ({report['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
