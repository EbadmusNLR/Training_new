#!/usr/bin/env python3
"""Evaluate one checkpoint with strict split-level percentage metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import physics
from gridfm.model import EdgeStateGridFM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    help="dataset/config override; checkpoint config is authoritative by default")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--baseline", choices=("v_init",),
                    help="evaluate a non-learned baseline instead of a checkpoint")
    ap.add_argument("--split", choices=("seen", "unseen", "test"), default="unseen")
    ap.add_argument("--device")
    ap.add_argument("--kcl-vsource", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not args.ckpt and not args.baseline:
        ap.error("provide --ckpt or --baseline")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else None
    cfg = yaml.safe_load(args.config.read_text()) if args.config else ck["cfg"]
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    if ck is not None:
        model = EdgeStateGridFM(**cfg["model"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
    sums: dict[str, float] = {}
    metric_rows: dict[str, list[float]] = {}
    batches = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                         num_workers=int(cfg["data"].get("num_workers", 0)))
    clamp = float(cfg["loss"]["feat_clamp"])
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            if args.baseline == "v_init":
                preds = {"node": torch.zeros_like(batch["node"].dv)}
                preds.update({s: torch.zeros_like(batch[s].x_true) for s in physics.SPECS})
            else:
                preds = {k: v.float() for k, v in model(batch).items()}
            preds = physics.clamp_structural_zeros(batch, preds)
            if args.kcl_vsource:
                preds = physics.kcl_decode_vsource(batch, preds, clamp)
            for key, value in physics.percentage_error_sums(batch, preds, clamp).items():
                sums[key] = sums.get(key, 0.0) + value
            xbar, vr, vi = physics.completed(batch, preds)
            scaler = json.loads((Path(cfg["data"]["root"]) / "feature_scaler.json").read_text())
            skcl = statistics.median(v["I_scale"] for v in scaler["current"].values())
            _, _, pm = physics.physics_losses(batch, xbar, vr, vi, clamp, skcl)
            for key, value in pm.items():
                metric_rows.setdefault(key, []).append(float(value))
    report = {
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "baseline": args.baseline, "split": args.split,
        "kcl_vsource": args.kcl_vsource, "n_samples": len(dataset),
    }
    for key in {k[:-4] for k in sums if k.endswith("_num")}:
        den = sums.get(f"{key}_den", 0.0)
        if den > 0:
            report[f"{key}_wape_pct"] = 100.0 * sums[f"{key}_num"] / den
    report.update({k: statistics.fmean(v) for k, v in metric_rows.items()})
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
