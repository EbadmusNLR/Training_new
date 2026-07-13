#!/usr/bin/env python3
"""Attribute the unseen PF current (Ibus) error of a trained checkpoint.

The promoted E51 model reaches V~1.7% and shunt-device current ~2% (hybrid
Ibus=YV-Icomp), but line/reactor/transformer/vsource series currents stay at
~4-7%. Series line current is reconstructed by the KCL tree sweep, which treats
every NON-line terminal current (loads, but also transformer/reactor/vsource
direct-head currents) as a KNOWN injection. This script measures, on the real
model, how much each error source contributes to the series-current WAPE by
substituting ground truth for one source at a time before the standard
hybrid-device + tree-line decode.

Variants (all use hybrid device decode + tree line w/ physics shunt):
  model            : model predictions as-is (reproduces the scorecard)
  truthV           : node voltage set to truth (=> hybrid shunt currents exact)
  truth_shuntI     : shunt-device Ibus set to truth (load/cap/gen/pv/storage)
  truth_seriesI    : transformer + reactor Ibus set to truth
  truth_xfmrI      : transformer Ibus set to truth only
  truth_reactorI   : reactor Ibus set to truth only
  truthV_seriesI   : both truth V and truth transformer/reactor Ibus

If truth_seriesI collapses line/vsource WAPE, the stiff transformer/reactor
direct-head currents polluting the sweep are the dominant cause and the fix is
to reconstruct them by KCL too. If truthV does it, voltage is the ceiling.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.config import load_config
from gridfm.legacy import physics, store_width, i_offset, SPECS
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.hybrid_current import decode_hybrid_device_currents
from gridfm.tree_current import decode_tree_line_currents

SHUNT = ("capacitor", "generator", "load", "pvsystem", "storage")
SERIES = ("transformer", "reactor")


def sub_truth_current(preds, batch, stores):
    out = dict(preds)
    for s in stores:
        st = batch[s]
        if st.num_nodes == 0:
            continue
        ni = i_offset(s)
        p = out[s].clone()
        p[:, ni:] = st.x_true[:, ni:]
        out[s] = p
    return out


def sub_truth_voltage(preds, batch):
    out = dict(preds)
    out["node"] = batch["node"].dv.clone()
    return out


def decode(preds, batch, clamp):
    preds = decode_hybrid_device_currents(batch, preds, clamp)
    preds = decode_tree_line_currents(batch, preds, clamp, physics_shunt=True)
    return preds


def family_wape(batch, preds, clamp):
    return physics.percentage_error_sums(batch, preds, clamp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="unseen")
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["mask"]["mixture"] = {"pf": 1.0}
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = dict(cfg["model"])
    model_cfg.pop("dtype", "float32")
    if "condition_on_scale" not in model_cfg:
        in_features = ck["model"]["comp_encoder.line.0.weight"].shape[1]
        model_cfg["condition_on_scale"] = in_features == 4 * store_width("line")
    model = EdgeStateGridFM(**model_cfg).to(device)
    load_compatible_state(model, ck["model"])
    model.eval()

    clamp = float(cfg["loss"]["feat_clamp"])
    batches = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]),
                         shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)),
                         multiprocessing_context="fork" if cfg["data"].get("num_workers") else None)

    variants = ("model", "truthV", "truth_shuntI", "truth_seriesI",
                "truth_xfmrI", "truth_reactorI", "truthV_seriesI")
    sums = {v: {} for v in variants}

    with torch.no_grad():
        for bi, batch in enumerate(batches):
            if args.limit_batches and bi >= args.limit_batches:
                break
            batch = batch.to(device)
            base = physics.clamp_structural_zeros(batch, model(batch))
            builders = {
                "model": base,
                "truthV": sub_truth_voltage(base, batch),
                "truth_shuntI": sub_truth_current(base, batch, SHUNT),
                "truth_seriesI": sub_truth_current(base, batch, SERIES),
                "truth_xfmrI": sub_truth_current(base, batch, ("transformer",)),
                "truth_reactorI": sub_truth_current(base, batch, ("reactor",)),
                "truthV_seriesI": sub_truth_current(
                    sub_truth_voltage(base, batch), batch, SERIES),
            }
            for name, p in builders.items():
                dec = decode(p, batch, clamp)
                for k, val in family_wape(batch, dec, clamp).items():
                    sums[name][k] = sums[name].get(k, 0.0) + val

    def wape(d, key):
        num, den = d.get(f"{key}_num", 0.0), d.get(f"{key}_den", 0.0)
        return 100.0 * num / den if den > 0 else None

    keys = ["V", "Ibus", "Ibus_line", "Ibus_transformer", "Ibus_reactor",
            "Ibus_vsource", "Ibus_load", "Ibus_capacitor"]
    report = {v: {k: wape(sums[v], k) for k in keys} for v in variants}
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n")

    print("\n=== series-current WAPE by attribution (unseen PF) ===")
    hdr = f"{'variant':16s} " + " ".join(f"{k.replace('Ibus_',''):>9s}" for k in keys)
    print(hdr)
    for v in variants:
        row = " ".join(
            (f"{report[v][k]:9.3f}" if report[v][k] is not None else f"{'--':>9s}")
            for k in keys)
        print(f"{v:16s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
