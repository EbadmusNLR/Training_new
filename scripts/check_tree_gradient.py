#!/usr/bin/env python3
"""Verify that structural line-current loss backpropagates into device heads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import physics, store_width
from gridfm.losses import physical_ibus_wape_loss
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.tree_current import decode_tree_line_currents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    batch = Batch.from_data_list([
        bundle.train[i] for i in range(min(args.samples, len(bundle.train)))
    ]).to(device)
    model_cfg = dict(cfg["model"])
    model_dtype = model_cfg.pop("dtype", "float32")
    in_features = ck["model"]["comp_encoder.line.0.weight"].shape[1]
    model_cfg["condition_on_scale"] = in_features == 4 * store_width("line")
    model = EdgeStateGridFM(**model_cfg).to(device)
    if model_dtype == "float64":
        model = model.double()
    load_compatible_state(model, ck["model"])
    raw = model(batch)
    preds = physics.clamp_structural_zeros(batch, raw)
    clamp = float(cfg["loss"]["feat_clamp"])
    tree = decode_tree_line_currents(batch, preds, clamp)
    loss = physical_ibus_wape_loss(batch, tree, clamp, ("line",))
    loss.backward()
    norms = {}
    for store in ("load", "transformer", "generator", "pvsystem", "storage"):
        grads = [p.grad for p in model.field_head[store].parameters() if p.grad is not None]
        norms[store] = sum(float(g.abs().sum()) for g in grads)
    print(f"tree_line_wape={float(loss.detach()):.8f}")
    print("device_head_gradient_l1=" + ",".join(f"{k}:{v:.6e}" for k, v in norms.items()))
    if norms["load"] <= 0 or norms["transformer"] <= 0:
        raise SystemExit("structural loss did not reach required device heads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
