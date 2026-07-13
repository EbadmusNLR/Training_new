#!/usr/bin/env python3
"""Current error vs voltage error for the physics current decode (truth-based).

react_yv_truthV = 0.16% established that hybrid(shunts incl. reactor Y*V) +
tree_line + kcl_vsource maps truth V -> currents essentially exactly. This maps
how that current error grows as V degrades, to decide whether "improve V + decode
currents from physics" can reach <1% currents, and at what V accuracy.

Perturb truth node voltage (masked nodes) by relative Gaussian noise of size eps,
then run the full physics decode and report aggregate/line/reactor/vsource WAPE
and the induced V WAPE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.config import load_config
from gridfm.legacy import physics, SPECS
from gridfm.hybrid_current import decode_hybrid_device_currents, SAFE_PHYSICS_STORES

REACT_SAFE = SAFE_PHYSICS_STORES + ("reactor",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default="unseen")
    ap.add_argument("--limit-batches", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["mask"]["mixture"] = {"pf": 1.0}
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], int(cfg["train"]["seed"]))
    dataset = getattr(bundle, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp = float(cfg["loss"]["feat_clamp"])
    from gridfm.tree_current import decode_tree_line_currents

    epslist = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2]
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = {e: {} for e in epslist}
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.limit_batches:
                break
            batch = batch.to(device)
            nd = batch["node"]
            for eps in epslist:
                # truth preds, then perturb masked node voltages
                preds = {"node": nd.dv.clone()}
                preds.update({s: batch[s].x_true.clone() for s in SPECS})
                if eps > 0:
                    vmag = (nd.v_init + nd.dv).norm(dim=1, keepdim=True)
                    noise = torch.randn(nd.dv.shape, generator=g).to(device) * eps * vmag
                    preds["node"] = nd.dv + noise * nd.msk_v.unsqueeze(1)
                preds = physics.clamp_structural_zeros(batch, preds)
                p = decode_hybrid_device_currents(batch, preds, clamp, stores=REACT_SAFE)
                p = decode_tree_line_currents(batch, p, clamp, physics_shunt=True)
                p = physics.kcl_decode_vsource(batch, p, clamp)
                for k, v in physics.percentage_error_sums(batch, p, clamp).items():
                    rows[eps][k] = rows[eps].get(k, 0.0) + v

    def w(d, key):
        n, dd = d.get(f"{key}_num", 0.0), d.get(f"{key}_den", 0.0)
        return 100.0 * n / dd if dd > 0 else float("nan")

    keys = ["V", "Ibus", "Ibus_line", "Ibus_reactor", "Ibus_transformer", "Ibus_vsource", "Ibus_load"]
    print(f"{'eps':>8s} " + " ".join(f"{k.replace('Ibus_',''):>10s}" for k in keys))
    for eps in epslist:
        print(f"{eps:8.1e} " + " ".join(f"{w(rows[eps], k):10.4f}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
