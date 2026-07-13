#!/usr/bin/env python3
"""Evaluate one checkpoint with strict split-level percentage metrics."""
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
from gridfm.legacy import physics, store_width
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.current_projection import project_kcl
from gridfm.hybrid_current import decode_hybrid_device_currents
from gridfm.tree_current import decode_tree_line_currents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    help="dataset/config override; checkpoint config is authoritative by default")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--ensemble-ckpt", type=Path, action="append", default=[])
    ap.add_argument("--baseline", choices=("v_init",),
                    help="evaluate a non-learned baseline instead of a checkpoint")
    ap.add_argument("--split", choices=("seen", "unseen", "test"), default="unseen")
    ap.add_argument(
        "--task", choices=(
            "pf", "se", "se_known", "param", "param_one", "injection",
            "random", "ctrl", "topo", "sysid",
        ),
        help="replace the checkpoint mask mixture with one deterministic task family",
    )
    ap.add_argument("--device")
    ap.add_argument("--kcl-vsource", action="store_true")
    ap.add_argument("--kcl-project", choices=("equal", "series", "line"))
    ap.add_argument("--tree-line", action="store_true",
                    help="reconstruct paired radial line-series currents from KCL")
    ap.add_argument("--hybrid-device", action="store_true",
                    help="decode non-stiff device currents from local complex physics")
    ap.add_argument(
        "--exact-pf-ceiling", action="store_true",
        help=(
            "diagnostic only: apply the validated dense KCL solve when Y and Icomp "
            "are fully observed; this is not a learned-model result"
        ),
    )
    ap.add_argument(
        "--physics-current", action="store_true",
        help="diagnostic only: decode Ibus=YV-Icomp after voltage completion",
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not args.ckpt and not args.baseline:
        ap.error("provide --ckpt or --baseline")
    if args.exact_pf_ceiling and args.task != "pf":
        ap.error("--exact-pf-ceiling requires --task pf")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else None
    ensemble_cks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.ensemble_ckpt
    ]
    if ensemble_cks and ck is None:
        ap.error("--ensemble-ckpt requires --ckpt")
    cfg = load_config(args.config) if args.config else ck["cfg"]
    if args.task:
        cfg["mask"]["mixture"] = {args.task: 1.0}
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    models = []
    if ck is not None:
        model_cfg = dict(cfg["model"])
        model_dtype = model_cfg.pop("dtype", "float32")
        if "condition_on_scale" not in model_cfg:
            in_features = ck["model"]["comp_encoder.line.0.weight"].shape[1]
            model_cfg["condition_on_scale"] = in_features == 4 * store_width("line")
        model = EdgeStateGridFM(**model_cfg).to(device)
        if model_dtype == "float64":
            model = model.double()
        load_compatible_state(model, ck["model"])
        model.eval()
        models.append(model)
        for other in ensemble_cks:
            other_cfg = dict(other["cfg"]["model"])
            other_dtype = other_cfg.pop("dtype", "float32")
            if "condition_on_scale" not in other_cfg:
                other_in = other["model"]["comp_encoder.line.0.weight"].shape[1]
                other_cfg["condition_on_scale"] = other_in == 4 * store_width("line")
            if other_cfg != model_cfg or other_dtype != model_dtype:
                raise SystemExit("ensemble checkpoints must use the same model architecture")
            member = EdgeStateGridFM(**other_cfg).to(device)
            if other_dtype == "float64":
                member = member.double()
            load_compatible_state(member, other["model"])
            member.eval()
            models.append(member)
    sums: dict[str, float] = {}
    metric_rows: dict[str, list[float]] = {}
    feasibility = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    workers = int(cfg["data"].get("num_workers", 0))
    batches = DataLoader(
        dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
        num_workers=workers, multiprocessing_context="fork" if workers else None,
    )
    clamp = float(cfg["loss"]["feat_clamp"])
    scaler = json.loads((Path(cfg["data"]["root"]) / "feature_scaler.json").read_text())
    skcl = statistics.median(v["I_scale"] for v in scaler["current"].values())
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            if args.baseline == "v_init":
                preds = {"node": torch.zeros_like(batch["node"].dv)}
                preds.update({s: torch.zeros_like(batch[s].x_true) for s in physics.SPECS})
            else:
                members = [member(batch) for member in models]
                preds = {"node": torch.stack([row["node"] for row in members]).mean(0)}
                for store in physics.SPECS:
                    st = batch[store]
                    ni = physics.i_offset(store)
                    value = torch.stack([row[store] for row in members]).mean(0)
                    if len(members) > 1 and st.num_nodes:
                        currents = [
                            physics.decode(row[store][:, ni:], st.scale[:, ni:], clamp)
                            for row in members
                        ]
                        mean_current = torch.stack(currents).mean(0)
                        value[:, ni:] = torch.asinh(
                            mean_current / (st.scale[:, ni:] + physics.EPS)
                        )
                    preds[store] = value
                preds = {
                    k: v.float() if v.dtype in (torch.float16, torch.bfloat16) else v
                    for k, v in preds.items()
                }
            preds = physics.clamp_structural_zeros(batch, preds)
            if args.exact_pf_ceiling:
                preds = physics.exact_pf_solve(batch, preds, clamp)
            if args.physics_current:
                preds = physics.decode_currents(batch, preds, clamp)
            if args.hybrid_device:
                preds = decode_hybrid_device_currents(batch, preds, clamp)
            if args.tree_line:
                preds = decode_tree_line_currents(batch, preds, clamp)
            if args.kcl_project:
                preds = project_kcl(batch, preds, clamp, args.kcl_project)
            if args.kcl_vsource:
                preds = physics.kcl_decode_vsource(batch, preds, clamp)
            for key, value in physics.percentage_error_sums(batch, preds, clamp).items():
                sums[key] = sums.get(key, 0.0) + value
            nd = batch["node"]
            score_v = nd.msk_v
            if score_v.any():
                vhat = nd.v_init + torch.where(
                    nd.msk_v.unsqueeze(1), preds["node"].to(nd.dv.dtype), nd.dv
                )
                vtrue = nd.v_init + nd.dv
                pred_bad = (vhat[score_v].norm(dim=1) < 0.95) | (
                    vhat[score_v].norm(dim=1) > 1.05
                )
                true_bad = (vtrue[score_v].norm(dim=1) < 0.95) | (
                    vtrue[score_v].norm(dim=1) > 1.05
                )
                feasibility["tp"] += int((pred_bad & true_bad).sum())
                feasibility["fp"] += int((pred_bad & ~true_bad).sum())
                feasibility["tn"] += int((~pred_bad & ~true_bad).sum())
                feasibility["fn"] += int((~pred_bad & true_bad).sum())
            xbar, vr, vi = physics.completed(batch, preds)
            _, _, pm = physics.physics_losses(batch, xbar, vr, vi, clamp, skcl)
            for key, value in pm.items():
                metric_rows.setdefault(key, []).append(float(value))
    report = {
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "ensemble_checkpoints": [str(path) for path in args.ensemble_ckpt],
        "baseline": args.baseline, "split": args.split,
        "task": args.task,
        "kcl_vsource": args.kcl_vsource, "n_samples": len(dataset),
        "kcl_project": args.kcl_project,
        "exact_pf_ceiling": args.exact_pf_ceiling,
        "physics_current": args.physics_current,
        "tree_line": args.tree_line,
        "hybrid_device": args.hybrid_device,
    }
    for key in {k[:-4] for k in sums if k.endswith("_num")}:
        den = sums.get(f"{key}_den", 0.0)
        if den > 0:
            report[f"{key}_wape_pct"] = 100.0 * sums[f"{key}_num"] / den
    report.update({k: statistics.fmean(v) for k, v in metric_rows.items()})
    tp, fp, tn, fn = (feasibility[k] for k in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    if total:
        report.update({
            "feasibility_accuracy": (tp + tn) / total,
            "feasibility_precision": tp / max(1, tp + fp),
            "feasibility_recall": tp / max(1, tp + fn),
            "feasibility_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "feasibility_violation_rate": (tp + fn) / total,
        })
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
