#!/usr/bin/env python3
"""Decompose current error into feature, voltage, and topology contributions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import SPECS, i_offset, physics, store_width
from gridfm.model import EdgeStateGridFM


def add(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def wapes(sums: dict[str, float]) -> dict[str, float]:
    out = {}
    for key in {k[:-4] for k in sums if k.endswith("_num")}:
        den = sums.get(f"{key}_den", 0.0)
        if den > 0:
            out[f"{key}_wape_pct"] = 100.0 * sums[f"{key}_num"] / den
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("seen", "unseen", "test"), default="unseen")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    model_cfg = dict(cfg["model"])
    model_dtype = model_cfg.pop("dtype", "float32")
    in_features = ck["model"]["comp_encoder.line.0.weight"].shape[1]
    model_cfg["condition_on_scale"] = in_features == 4 * store_width("line")
    model = EdgeStateGridFM(**model_cfg).to(device)
    if model_dtype == "float64":
        model = model.double()
    model.load_state_dict(ck["model"])
    model.eval()
    clamp = float(cfg["loss"]["feat_clamp"])

    modes = {name: {} for name in (
        "direct", "direct_kcl", "physics_pred_v", "physics_pred_v_kcl", "physics_truth_v"
    )}
    family = {s: {
        "entries": 0, "pu_abs_sum": 0.0, "feat_abs_sum": 0.0,
        "scale_min": math.inf, "scale_max": 0.0, "near_zero": 0,
    } for s in SPECS}
    worst = []
    with torch.no_grad():
        for idx, (fi, _) in enumerate(dataset.items):
            batch = Batch.from_data_list([dataset[idx]]).to(device)
            raw = model(batch)
            direct = physics.clamp_structural_zeros(batch, raw)
            direct_kcl = physics.kcl_decode_vsource(batch, direct, clamp)
            predv = physics.decode_currents(batch, direct, clamp)
            predv_kcl = physics.kcl_decode_vsource(batch, predv, clamp)
            truth = {"node": batch["node"].dv}
            truth.update({s: batch[s].x_true for s in SPECS})
            truthv = physics.decode_currents(batch, truth, clamp)
            variants = {
                "direct": direct, "direct_kcl": direct_kcl,
                "physics_pred_v": predv, "physics_pred_v_kcl": predv_kcl,
                "physics_truth_v": truthv,
            }
            per = {}
            for name, preds in variants.items():
                terms = physics.percentage_error_sums(batch, preds, clamp)
                add(modes[name], terms)
                per[name] = wapes(terms)
            worst.append({
                "feeder": dataset.caches[fi].name,
                "Ibus_wape_pct": per["direct_kcl"].get("Ibus_wape_pct"),
                "line_wape_pct": per["direct_kcl"].get("Ibus_line_wape_pct"),
            })
            for store in SPECS:
                st = batch[store]
                if st.num_nodes == 0:
                    continue
                ni = i_offset(store)
                mask = st.msk[:, ni:]
                if not mask.any():
                    continue
                pu = physics.decode_truth(st.x_true[:, ni:], st.scale[:, ni:])
                vals, feat, scale = pu[mask], st.x_true[:, ni:][mask], st.scale[:, ni:][mask]
                row = family[store]
                row["entries"] += int(mask.sum())
                row["pu_abs_sum"] += vals.abs().sum().item()
                row["feat_abs_sum"] += feat.abs().sum().item()
                row["scale_min"] = min(row["scale_min"], scale.min().item())
                row["scale_max"] = max(row["scale_max"], scale.max().item())
                row["near_zero"] += int((vals.abs() < 1e-8).sum())
    for row in family.values():
        n = max(1, row["entries"])
        row["pu_abs_mean"] = row.pop("pu_abs_sum") / n
        row["feat_abs_mean"] = row.pop("feat_abs_sum") / n
        row["near_zero_fraction"] = row.pop("near_zero") / n
        if not math.isfinite(row["scale_min"]):
            row["scale_min"] = None
    report = {
        "checkpoint": str(args.ckpt), "split": args.split,
        "n_samples": len(dataset), "scale_conditioned": model_cfg["condition_on_scale"],
        "modes": {name: wapes(terms) for name, terms in modes.items()},
        "families": family,
        "worst_feeders": sorted(
            worst, key=lambda x: x["Ibus_wape_pct"] or -1, reverse=True
        )[:20],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

