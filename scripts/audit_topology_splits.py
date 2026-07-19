#!/usr/bin/env python3
"""Audit train/unseen/test leakage by content-derived topology fingerprints.

The production split is feeder-name based. Vendored OpenDSS copies can therefore
put the same physical graph under different paths into different splits. We report
both an order-sensitive exact structural hash and a node-order-invariant
Weisfeiler-Lehman-style hash of the heterogeneous bus/component incidence graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "datakit"))
sys.path.insert(0, str(ROOT))

CORPORA = ("dss_data", "minimal_component", "new_dss_data", "SMART-DS_1000")


def _h(parts):
    z = hashlib.sha256()
    for p in parts:
        z.update(str(p).encode()); z.update(b"\0")
    return z.hexdigest()


def fingerprint(row):
    corpus, feeder, split = row
    from core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, node_count, store_size

    try:
        d = FeederScenarios(feeder)[0]
        nn = int(node_count(d))
        offsets = {}; nv = nn
        initial = ["bus"] * nn
        counts = {}
        for s in sorted(STORES):
            n = int(store_size(d, s)) if s in d.node_types else 0
            counts[s] = n
            offsets[s] = nv
            initial.extend([f"comp:{s}"] * n)
            nv += n

        adj = [[] for _ in range(nv)]
        exact_edges = []
        for rel in sorted(d.edge_types, key=str):
            src, role, dst = rel
            if src not in STORES or dst != "node" or src not in offsets:
                continue
            ei = d[rel].edge_index
            if not ei.numel():
                continue
            for c, n in zip(ei[0].tolist(), ei[1].tolist()):
                u, v = offsets[src] + int(c), int(n)
                lab = f"{src}:{role}"
                adj[u].append((lab + ">", v))
                adj[v].append(("<" + lab, u))
                exact_edges.append((src, role, int(c), int(n)))

        exact = _h([nn, *[f"{s}:{counts[s]}" for s in sorted(counts)],
                    *sorted(exact_edges)])
        colors = [_h([x]) for x in initial]
        for _ in range(6):
            colors = [_h([initial[i], colors[i],
                          *sorted(f"{lab}:{colors[j]}" for lab, j in adj[i])])
                      for i in range(nv)]
        wl = _h([nn, nv, *sorted(colors)])
        return {"corpus": corpus, "feeder": feeder, "split": split,
                "nodes": nn, "components": nv - nn, "exact": exact, "wl": wl}
    except Exception as exc:
        return {"corpus": corpus, "feeder": feeder, "split": split,
                "error": f"{type(exc).__name__}: {exc}"}


def summarize(rows, key):
    groups = defaultdict(list)
    for r in rows:
        if "error" not in r:
            groups[r[key]].append(r)
    cross = []
    leaked = Counter()
    for fp, members in groups.items():
        splits = {r["split"] for r in members}
        if len(splits) > 1:
            cross.append((fp, members))
        if "train" in splits:
            for r in members:
                if r["split"] != "train":
                    leaked[r["split"]] += 1
    return groups, cross, leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--json", default=None)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()

    from gridfm.dk_data import discover_feeders, split_feeders
    items = []
    for corpus in CORPORA:
        root = str(PROJECT / "training_data" / corpus)
        fs = discover_feeders(root)
        split = split_feeders(fs, seed=42)
        for name, dirs in split.items():
            items.extend((corpus, d, name) for d in dirs)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(fingerprint, items, chunksize=2))

    errors = [r for r in rows if "error" in r]
    print(f"feeders={len(rows)} errors={len(errors)} split={dict(Counter(r['split'] for r in rows))}")
    for key in ("exact", "wl"):
        groups, cross, leaked = summarize(rows, key)
        duplicates = sum(len(v) - 1 for v in groups.values())
        print(f"{key}: unique={len(groups)} redundant={duplicates} "
              f"cross_split_groups={len(cross)} train_leak={dict(leaked)}")
        for _, members in sorted(cross, key=lambda x: -len(x[1]))[:a.examples]:
            compact = [f"{r['split']}:{r['corpus']}:{os.path.basename(r['feeder'])}"
                       for r in members]
            print(f"  x{len(members)} " + " | ".join(compact))
    for r in errors[:a.examples]:
        print(f"ERROR {r['corpus']}:{os.path.basename(r['feeder'])}: {r['error']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
