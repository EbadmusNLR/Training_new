"""Find UNUSED, HEALTHY SMART-DS feeders to replace excluded ones.

Compares by RELATIVE PATH under v1.0 (the same feeder name recurs across scenarios, so a
name-only match would wrongly rule out valid candidates). Health is VERIFIED by solving --
the two feeders just excluded were 95%+ de-energized despite compiling fine, so "it exists"
is not evidence it is usable.
"""
import os, sys, json
POOL = "/kfs2/projects/gogpt/Ebadmus/SMART-DS/v1.0/peak"
USED = "/kfs2/projects/gogpt/Ebadmus/data/SMART-DS_1000/v1.0/peak"
EXCL = "/kfs2/projects/gogpt/Ebadmus/data/excluded"

def feeders(root):
    out = {}
    for sub, dirs, files in os.walk(root):
        # a feeder dir is  .../opendss_no_loadshapes/<substation>/<substation>--<feeder>
        # so opendss_no_loadshapes is its GRANDparent, not its parent.
        if "--" in os.path.basename(sub) and os.sep + "opendss_no_loadshapes" + os.sep in sub:
            out[os.path.relpath(sub, root)] = sub
            dirs[:] = []
    return out

used = feeders(USED)
pool = feeders(POOL)
used_names = {os.path.basename(p) for p in used}
# names already excluded (never re-import a known-bad feeder)
excl_names = set()
for k in os.listdir(EXCL):
    d = os.path.join(EXCL, k)
    if os.path.isdir(d):
        for n in os.listdir(d):
            excl_names.add(n.rsplit("__", 1)[0])
cand = {r: p for r, p in pool.items()
        if r not in used and os.path.basename(p) not in used_names
        and os.path.basename(p) not in excl_names}
print(f"pool={len(pool)}  used={len(used)}  excluded_names={len(excl_names)}  candidates={len(cand)}")
# prefer feeders that LOOK loaded: a non-trivial Loads.dss
scored = []
for r, p in cand.items():
    lp = os.path.join(p, "Loads.dss")
    if not os.path.isfile(lp): continue
    try: nl = sum(1 for l in open(lp) if l.strip().lower().startswith("new load."))
    except Exception: continue
    if nl >= 5: scored.append((nl, r, p))
print(f"candidates with >=5 loads: {len(scored)}")
# DEDUPE BY FEEDER NAME: the same feeder recurs across ~10 scenario dirs, so the raw
# top-N is one feeder ten times. Two replacements must be two DIFFERENT feeders.
seen, uniq = set(), []
for nl, r, p in scored:
    nm = os.path.basename(p)
    if nm in seen: continue
    seen.add(nm); uniq.append((nl, r, p))
# prefer TYPICAL size (match the corpus profile, not the 11896-load extreme)
mid = [t for t in uniq if 40 <= t[0] <= 600]
mid.sort(key=lambda t: -t[0])
print(f"distinct feeders: {len(uniq)}   with 40-600 loads: {len(mid)}")
json.dump([[n, r, p] for n, r, p in mid[:40]], open(sys.argv[1], "w"))
for n, r, p in mid[:6]:
    print(f"  {n:5d} loads  {r}")
