import sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["/kfs2/projects/gogpt/Ebadmus/Training_new", "/kfs2/projects/gogpt/Ebadmus"]
from datakit.core.scenario_store import FeederScenarios
from collections import defaultdict
import itertools
stats = defaultdict(lambda: {"rows":0,"zero":0,"maxabs":0.0,"sumabs":0.0,"sumI":0.0})
root = Path("/kfs2/projects/gogpt/Ebadmus/training_data")
for corpus in ("minimal_component","SMART-DS_1000"):
    for p in itertools.islice((root/corpus).rglob("static.pt"), 120):
        try: d = FeederScenarios(p.parent)[0]
        except Exception: continue
        for fam in ("load","pvsystem","generator","storage","vsource"):
            if fam not in d.node_types: continue
            st = d[fam]
            if st.get("Icomp_r_pu") is None: continue
            ic = st["Icomp_r_pu"].numpy() + 1j*st["Icomp_i_pu"].numpy()
            it = st["I_r_bus1_pu"].numpy() + 1j*st["I_i_bus1_pu"].numpy()
            s = stats[(corpus,fam)]
            s["rows"] += ic.size
            s["zero"] += int((np.abs(ic) < 1e-12).sum())
            s["maxabs"] = max(s["maxabs"], float(np.abs(ic).max()))
            s["sumabs"] += float(np.abs(ic).sum())
            s["sumI"]   += float(np.abs(it).sum())
print(f"{'corpus/family':32s} {'rows':>9s} {'zero%':>7s} {'max|Icomp|':>12s} {'sum|Icomp|/sum|I|':>18s}")
for (c,f),s in sorted(stats.items()):
    z = 100*s["zero"]/max(s["rows"],1)
    ratio = s["sumabs"]/max(s["sumI"],1e-30)
    print(f"{c+'/'+f:32s} {s['rows']:9d} {z:7.2f} {s['maxabs']:12.3e} {ratio:18.3e}")
