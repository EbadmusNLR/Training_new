#!/usr/bin/env python3
"""Attribute current WAPE numerator and denominator to truth-magnitude bins."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import SPECS, i_offset, physics, store_width
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.tree_current import decode_tree_line_currents


BOUNDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def empty_bins():
    return [{"count": 0, "error_sum": 0.0, "truth_abs_sum": 0.0}
            for _ in range(len(BOUNDS) + 1)]


def labels():
    edges = (0.0,) + BOUNDS + (math.inf,)
    return [f"[{edges[i]:.0e},{edges[i + 1]:.0e})" for i in range(len(edges) - 1)]


def add_bins(rows, truth, pred):
    mag = truth.abs()
    err = (pred - truth).abs()
    idx = torch.bucketize(mag, mag.new_tensor(BOUNDS))
    for bucket in range(len(rows)):
        take = idx == bucket
        if take.any():
            rows[bucket]["count"] += int(take.sum())
            rows[bucket]["error_sum"] += float(err[take].sum())
            rows[bucket]["truth_abs_sum"] += float(mag[take].sum())


def summarize(rows):
    total_err = sum(row["error_sum"] for row in rows)
    total_truth = sum(row["truth_abs_sum"] for row in rows)
    out = []
    for label, row in zip(labels(), rows):
        item = {"truth_range_pu": label, **row}
        item["error_numerator_fraction"] = row["error_sum"] / max(total_err, 1e-30)
        item["truth_denominator_fraction"] = row["truth_abs_sum"] / max(total_truth, 1e-30)
        item["bin_wape_pct"] = 100 * row["error_sum"] / max(row["truth_abs_sum"], 1e-30)
        out.append(item)
    return {"wape_pct": 100 * total_err / max(total_truth, 1e-30), "bins": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("seen", "unseen", "test"), default="unseen")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = dict(cfg["model"])
    dtype = model_cfg.pop("dtype", "float32")
    if "condition_on_scale" not in model_cfg:
        width = ck["model"]["comp_encoder.line.0.weight"].shape[1]
        model_cfg["condition_on_scale"] = width == 4 * store_width("line")
    model = EdgeStateGridFM(**model_cfg).to(device)
    if dtype == "float64":
        model = model.double()
    load_compatible_state(model, ck["model"])
    model.eval()
    clamp = float(cfg["loss"]["feat_clamp"])
    workers = int(cfg["data"].get("num_workers", 0))
    loader = DataLoader(
        dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
        num_workers=workers, multiprocessing_context="fork" if workers else None,
    )
    rows = {"aggregate": empty_bins(), **{store: empty_bins() for store in SPECS}}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = physics.clamp_structural_zeros(batch, model(batch))
            preds = decode_tree_line_currents(batch, preds, clamp)
            preds = physics.kcl_decode_vsource(batch, preds, clamp)
            for store in SPECS:
                st = batch[store]
                if st.num_nodes == 0:
                    continue
                ni = i_offset(store)
                mask = st.msk[:, ni:]
                truth = physics.decode_truth(st.x_true[:, ni:], st.scale[:, ni:])[mask]
                pred = physics.decode(preds[store][:, ni:], st.scale[:, ni:], clamp)[mask]
                add_bins(rows[store], truth, pred)
                add_bins(rows["aggregate"], truth, pred)
    report = {
        "checkpoint": str(args.ckpt), "split": args.split, "n_samples": len(dataset),
        "metrics": {name: summarize(value) for name, value in rows.items()},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"]["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
