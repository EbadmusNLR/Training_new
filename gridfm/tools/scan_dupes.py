"""Find DUPLICATE NETWORKS in a corpus by CONTENT, not by name.

new_dss_data is assembled from the same OpenDSS examples vendored into many upstream repos
(data-new / OpenDSS_Distrib / Parallel_Version / svn-mirror trunk|Version7|Version8 /
sourceforge ...). Each copy becomes its own feeder with its own 100 variants, so ONE network
can contribute several hundred highly-correlated samples and silently dominate the corpus.

Fingerprint = variant 0 (the UNPERTURBED baseline: make_training_pt uses base_text for k=1),
hashing the topology (per-relation edge_index) + every Y matrix + Icomp. Two feeders with the
same fingerprint are the SAME network, regardless of path.
"""
import argparse, glob, hashlib, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")


def fp(path):
    from Datakit.core.scenario_store import FeederScenarios
    from gridfm.dk_physics import STORES, store_size, node_count
    name = os.path.basename(os.path.dirname(path))
    try:
        d = FeederScenarios(os.path.dirname(path))[0]
        h = hashlib.sha1()
        h.update(str(node_count(d)).encode())
        for s in sorted(STORES):
            if s not in d.node_types or store_size(d, s) == 0:
                continue
            h.update(s.encode()); h.update(str(store_size(d, s)).encode())
            st = d[s]
            for k in sorted(st.keys()):
                v = st[k]
                if torch.is_tensor(v) and v.is_floating_point():
                    h.update(k.encode())
                    h.update(v.detach().double().numpy().tobytes())
        for r in sorted(d.edge_types, key=str):
            ei = d[r].edge_index
            if ei.numel():
                h.update(str(r).encode()); h.update(ei.numpy().tobytes())
        return (name, h.hexdigest(), len(FeederScenarios(os.path.dirname(path))))
    except Exception as e:
        return (name, f"ERR:{type(e).__name__}", 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="new_dss_data")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    root = f"/kfs2/projects/gogpt/Ebadmus/training_data/{a.corpus}"
    fs = sorted(glob.glob(os.path.join(root, "*", "static.pt")))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(fp, fs, chunksize=2))
    groups = defaultdict(list)
    for n, hh, nv in rows:
        groups[hh].append((n, nv))
    dupes = {h: v for h, v in groups.items() if len(v) > 1 and not h.startswith("ERR")}
    nfeed = len([r for r in rows if not r[1].startswith("ERR")])
    extra = sum(len(v) - 1 for v in dupes.values())
    print(f"=== {a.corpus}: {nfeed} feeders, {len(groups)} distinct networks")
    print(f"  duplicate GROUPS: {len(dupes)}   redundant feeders: {extra} "
          f"({100*extra/max(nfeed,1):.1f}% of the corpus)")
    print(f"  redundant SAMPLES: {sum((len(v)-1)*v[0][1] for v in dupes.values())}")
    for h, v in sorted(dupes.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"\n  x{len(v)} copies of one network ({v[0][1]} variants each):")
        for n, nv in sorted(v)[:4]:
            print(f"      {n[-72:]}")
        if len(v) > 4:
            print(f"      ... +{len(v)-4} more")
