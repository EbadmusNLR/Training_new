"""Remove DUPLICATE NETWORKS: keep ONE representative per distinct network, move the rest.

new_dss_data is 864 feeders but only 113 distinct networks -- the same OpenDSS examples
vendored into many upstream repos (ckt5 x59, IEEE123 x53, IEEE13 x33). Each copy carries its
own 100 variants, so one network contributes thousands of highly-correlated samples and
dominates training ~50x over every other network. Fingerprint = the UNPERTURBED variant 0
(topology + every Y + Icomp), so this is content-based, never name-based.

Keeps the SHORTEST name in each group (the least vendored-prefix noise) and moves the rest --
store AND its dss source, so a rebuild does not recreate them -- to data/excluded/duplicate_network/.
"""
import argparse, glob, hashlib, os, re, shutil, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/Training_new")
from gridfm.scan_dupes import fp

ROOT = "/kfs2/projects/gogpt/Ebadmus"
EXC = f"{ROOT}/data_excluded/duplicated"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    root = f"{ROOT}/training_data/{a.corpus}"
    fs = sorted(glob.glob(os.path.join(root, "*", "static.pt")))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(fp, fs, chunksize=2))
    groups = defaultdict(list)
    for n, h, nv in rows:
        if not h.startswith("ERR"):
            groups[h].append(n)
    moved = kept = 0
    for h, names in groups.items():
        if len(names) < 2:
            continue
        names = sorted(names, key=lambda x: (len(x), x))
        keep, drop = names[0], names[1:]
        kept += 1
        for nm in drop:
            src = os.path.join(root, nm)
            if not os.path.isdir(src):
                continue
            if not a.apply:
                moved += 1; continue
            dst = os.path.join(EXC, a.corpus, nm)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            # THE DSS SOURCE is what matters: leaving it in data/ means the next rebuild
            # recreates the duplicate. Move it first, then the built store.
            base = re.sub(r"__[0-9a-f]{12}$", "", nm)
            dsrc = f"{ROOT}/data/{a.corpus}/{base}"
            note = "no 1:1 source dir (nested tree) -- source left in place"
            if os.path.isdir(dsrc):
                os.makedirs(dst, exist_ok=True)
                shutil.move(dsrc, os.path.join(dst, "_dss_source"))
                note = f"dss source moved from data/{a.corpus}/{base}"
            shutil.move(src, os.path.join(dst, "store"))
            open(os.path.join(dst, "EXCLUDED_REASON.md"), "w").write(
                f"# Duplicate network (removed from training_data/{a.corpus})\n\n"
                f"Byte-identical network to the KEPT feeder:\n\n    {keep}\n\n"
                f"Fingerprint (unperturbed variant 0: topology + all Y + Icomp): {h}\n\n"
                "This is the same OpenDSS example vendored into another upstream repo. Its 100\n"
                "variants are perturbations of a network already represented, so keeping it only\n"
                "re-weights that topology in training -- it adds no new network.\n\n"
                f"DSS source: {note}\n")
            moved += 1
    tag = "MOVED" if a.apply else "would move (dry run)"
    print(f"{a.corpus}: {len(groups)} distinct networks | {kept} duplicate groups | {tag}: {moved}")
    print(f"  remaining feeders: {len(fs) - (moved if a.apply else 0)}")
