"""Which COMPONENT/CONNECTION types does the decoder fail on? minimal_component
feeder names encode the case (t1_0221_k10_load_delta_2ph_ab_m4_...), so group the
per-feeder error by category instead of guessing."""
import glob, json, os, re, sys
from collections import defaultdict
D = "runs/valmc"
rows = []
for f in sorted(glob.glob(os.path.join(D, "shard_*.json"))):
    for r in json.load(open(f)):
        if r["err"]:
            continue
        per = {}
        fn = fd = 0.0
        for s, (num, den) in r["acc"].items():
            per[s] = num / (den + 1e-30)
            fn += num; fd += den
        rows.append((fn / (fd + 1e-30), r["name"], per))
# category = the descriptive middle of the name: t1_0221_k10_<CATEGORY>_m4_<hash>
def cat(name):
    m = re.match(r"t1_\d+_k\d+_(.+?)(?:_m\d+)?(?:_[0-9a-f]{4,})?$", name)
    c = m.group(1) if m else name
    return re.sub(r"_m\d+.*$", "", c)
g = defaultdict(list)
for w, name, per in rows:
    g[cat(name)].append((w, per))
print(f"{'category':44s} {'n':>4s} {'meanWAPE':>10s} {'worst':>10s}  worst-family")
out = []
for c, lst in g.items():
    ws = [w for w, _ in lst]
    mean = sum(ws) / len(ws)
    # which family dominates in the worst feeder of this category
    wf = max(lst, key=lambda t: t[0])
    fam = max(wf[1].items(), key=lambda kv: kv[1])[0] if wf[1] else "-"
    out.append((mean, c, len(lst), max(ws), fam))
out.sort(reverse=True)
for mean, c, n, wmax, fam in out[:28]:
    print(f"{c[:44]:44s} {n:4d} {mean:10.2e} {wmax:10.2e}  {fam}")
print(f"\ncategories clean (<1e-9): {sum(1 for m,_,_,_,_ in out if m < 1e-9)}/{len(out)}")
