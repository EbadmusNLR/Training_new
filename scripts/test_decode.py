#!/usr/bin/env python3
"""Compare current-decode pipelines on a trained checkpoint (unseen PF)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import physics, store_width
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.hybrid_current import decode_hybrid_device_currents, SAFE_PHYSICS_STORES
from gridfm.tree_current import decode_tree_line_currents

REACT_SAFE = SAFE_PHYSICS_STORES + ("reactor",)


def pipe(name, base, batch, clamp):
    p = {k: v.clone() for k, v in base.items()}
    if name == "baseline":
        p = decode_hybrid_device_currents(batch, p, clamp)
        p = decode_tree_line_currents(batch, p, clamp, physics_shunt=True)
        p = physics.kcl_decode_vsource(batch, p, clamp)
    elif name == "react_yv":
        p = decode_hybrid_device_currents(batch, p, clamp, stores=REACT_SAFE)
        p = decode_tree_line_currents(batch, p, clamp, physics_shunt=True)
        p = physics.kcl_decode_vsource(batch, p, clamp)
    elif name == "react_zero":
        from gridfm.legacy import i_offset
        p = decode_hybrid_device_currents(batch, p, clamp)
        ni = i_offset("reactor")
        if batch["reactor"].num_nodes:
            p["reactor"][:, ni:] = 0.0
        p = decode_tree_line_currents(batch, p, clamp, physics_shunt=True)
        p = physics.kcl_decode_vsource(batch, p, clamp)
    elif name == "react_truth":
        from gridfm.legacy import i_offset
        p = decode_hybrid_device_currents(batch, p, clamp)
        ni = i_offset("reactor")
        if batch["reactor"].num_nodes:
            p["reactor"][:, ni:] = batch["reactor"].x_true[:, ni:]
        p = decode_tree_line_currents(batch, p, clamp, physics_shunt=True)
        p = physics.kcl_decode_vsource(batch, p, clamp)
    return p


def line_conductor_wape(batch, preds, clamp):
    """WAPE of line Ibus split into phase conductors (0,1,2) vs neutral (3)."""
    from gridfm.legacy import FC, i_offset
    st = batch["line"]
    ni = i_offset("line")
    pu_p = physics.decode(preds["line"][:, ni:].float(), st.scale[:, ni:], clamp)
    pu_t = physics.decode_truth(st.x_true[:, ni:], st.scale[:, ni:])
    msk = st.msk[:, ni:]
    # columns per terminal t: real t*2FC+cond, imag +FC. conductors 0..3.
    phase_cols, neut_cols = [], []
    for t in range(2):
        for cond in range(FC):
            base = t * 2 * FC + cond
            (neut_cols if cond == FC - 1 else phase_cols).extend([base, base + FC])
    out = {}
    for tag, cols in (("phase", phase_cols), ("neutral", neut_cols)):
        c = torch.tensor(cols, device=pu_p.device)
        m = msk[:, c]
        num = (pu_p[:, c] - pu_t[:, c]).abs()[m].sum().item()
        den = pu_t[:, c].abs()[m].sum().item()
        out[tag] = (num, den)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="unseen")
    ap.add_argument("--limit-batches", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["mask"]["mixture"] = {"pf": 1.0}
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = dict(cfg["model"])
    mcfg.pop("dtype", "float32")
    if "condition_on_scale" not in mcfg:
        mcfg["condition_on_scale"] = (
            ck["model"]["comp_encoder.line.0.weight"].shape[1] == 4 * store_width("line"))
    model = EdgeStateGridFM(**mcfg).to(device)
    load_compatible_state(model, ck["model"])
    model.eval()
    clamp = float(cfg["loss"]["feat_clamp"])
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)),
                        multiprocessing_context="fork" if cfg["data"].get("num_workers") else None)

    names = ["baseline", "react_zero", "react_truth", "react_yv"]
    sums = {n: {} for n in names}
    cond = {n: {"phase": [0.0, 0.0], "neutral": [0.0, 0.0]} for n in names}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            batch = batch.to(device)
            base = physics.clamp_structural_zeros(batch, model(batch))
            for n in names:
                dec = pipe(n, base, batch, clamp)
                for k, v in physics.percentage_error_sums(batch, dec, clamp).items():
                    sums[n][k] = sums[n].get(k, 0.0) + v
                lc = line_conductor_wape(batch, dec, clamp)
                for tag in ("phase", "neutral"):
                    cond[n][tag][0] += lc[tag][0]
                    cond[n][tag][1] += lc[tag][1]

    def w(d, key):
        num, den = d.get(f"{key}_num", 0.0), d.get(f"{key}_den", 0.0)
        return 100.0 * num / den if den > 0 else float("nan")

    keys = ["Ibus", "Ibus_line", "Ibus_reactor", "Ibus_transformer", "Ibus_vsource",
            "Ibus_load", "V"]
    hdr = f"{'pipeline':16s} " + " ".join(f"{k.replace('Ibus_',''):>11s}" for k in keys)
    print(hdr)
    for n in names:
        print(f"{n:16s} " + " ".join(f"{w(sums[n], k):11.3f}" for k in keys))
    print("\nline conductor split (WAPE%):")
    for n in names:
        ph = 100 * cond[n]["phase"][0] / max(cond[n]["phase"][1], 1e-12)
        nt = 100 * cond[n]["neutral"][0] / max(cond[n]["neutral"][1], 1e-12)
        print(f"  {n:16s} phase={ph:8.3f}  neutral={nt:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
