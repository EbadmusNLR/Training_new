"""Y-codebook survey: does the corpus's Y hypothesis space collapse into a small
GLOBAL codebook of normalized families x a scalar?

WHY: the Y head regresses dim*dim*2 free entries and diverges at generalization
scale (T24: 388k-3.2M% wape) while micro-overfitting cleanly (1.3%) -- the codec
is sound, the hypothesis space is too big. T22 says single-snapshot algebra
cannot pin Y, so the head needs the STRUCTURAL prior (conductor family, symmetry,
per-length scaling). This survey measures whether that prior exists in the data:
per store, normalize each component's Y by its dominant magnitude and count
distinct families at a rounding tolerance, plus how much of the corpus the top-K
families cover and how the scale factor spreads within a family.

If (say) >=95% of line components fall into <100 normalized families, the right
head is family-classification + log-scale regression (tens of DOF, codebook
shared corpus-wide) -- not free-entry regression.

Usage: y_family_survey.py --n-feeders 60 --subset-seed 601 [--variant 0]
"""
import argparse, os, sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from core.scenario_store import FeederScenarios
from gridfm.dk_physics import STORES
from gridfm.dk_data import discover_feeders, split_feeders

ROOTS = ["/kfs2/projects/gogpt/Ebadmus/training_data/" + c for c in
         ("SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feeders", type=int, default=60)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=601)
    ap.add_argument("--round", type=int, default=3,
                    help="decimals kept on the normalized matrix (family key)")
    a = ap.parse_args()
    from itertools import zip_longest
    srng = np.random.default_rng(a.subset_seed)
    per = [split_feeders(discover_feeders(r), seed=42) for r in ROOTS]
    pools = [list(c["train"]) + list(c["unseen"]) for c in per]
    for p in pools:
        srng.shuffle(p)
    feeders = [d for tup in zip_longest(*pools) for d in tup if d][: a.n_feeders]

    # store -> family_key -> [count, scales list, one exemplar]
    fam = defaultdict(lambda: defaultdict(lambda: [0, []]))
    totals = defaultdict(int)
    nf = 0
    for fdir in feeders:
        try:
            d = FeederScenarios(fdir)[a.variant]
        except Exception as e:
            print(f"{os.path.basename(fdir)[:44]:44s} SKIP {e}")
            continue
        nf += 1
        for s, (prefix, _, _) in STORES.items():
            fr, fi = f"{prefix}_r_pu", f"{prefix}_i_pu"
            if s not in d.node_types or fr not in d[s]:
                continue
            Y = (d[s][fr].double().numpy()
                 + 1j * d[s][fi].double().numpy())
            if Y.ndim < 3 or not Y.shape[0]:
                continue
            n = Y.shape[0]
            flat = Y.reshape(n, -1)
            scale = np.abs(flat).max(axis=1)
            scale[scale == 0] = 1.0
            norm = flat / scale[:, None]
            for k in range(n):
                key = (np.round(norm[k].real, a.round).tobytes()
                       + np.round(norm[k].imag, a.round).tobytes())
                fam[s][key][0] += 1
                fam[s][key][1].append(scale[k])
                totals[s] += 1

    print(f"\n=== Y family survey: {nf} feeders, variant {a.variant}, "
          f"round={a.round} (tol ~1e-{a.round}), subset-seed={a.subset_seed} ===")
    for s in sorted(totals, key=lambda x: -totals[x]):
        fams = fam[s]
        counts = np.array(sorted((v[0] for v in fams.values()), reverse=True))
        tot = totals[s]
        cum = np.cumsum(counts) / tot
        k95 = int(np.searchsorted(cum, 0.95) + 1) if tot else 0
        k99 = int(np.searchsorted(cum, 0.99) + 1) if tot else 0
        # scale spread within the top-5 families (log10 range)
        top = sorted(fams.values(), key=lambda v: -v[0])[:5]
        spreads = []
        for v in top:
            sc = np.array(v[1])
            sc = sc[sc > 0]
            if sc.size:
                spreads.append(float(np.log10(sc.max() / sc.min())))
        print(f"--- {s}: comps {tot} | families {counts.size} | "
              f"top1 {counts[0]/tot:.1%} top10 {cum[min(9, counts.size-1)]:.1%} | "
              f"95% needs {k95} fams, 99% needs {k99} | "
              f"top5 scale-spread log10 {['%.1f' % x for x in spreads]}")


if __name__ == "__main__":
    raise SystemExit(main())
