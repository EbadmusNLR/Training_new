"""Does the IEEE 30 Bus source actually solve in OpenDSS?

It is the ONE feeder the decoder refuses (loop through a transformer, rank 47 < 56),
and the arranged-corpus validation reports a SEGFAULT (worker exit -11) for the
IEEE_30_Bus master -- but for a DIFFERENT copy of the network. Before excluding
anything, check the copy the corpus was actually built from.
"""
import sys
import numpy as np
import opendssdirect as dss

p = sys.argv[1] if len(sys.argv) > 1 else \
    "/kfs2/projects/gogpt/Ebadmus/data/dss_data/IEEE 30 Bus/network/Master.dss"
print("source:", p)
try:
    dss.Text.Command('Clear')
    dss.Text.Command('Redirect "%s"' % p)
    print("  compile: OK   converged=%s" % dss.Solution.Converged())
    dss.Text.Command("Set ControlMode=OFF")   # NOT MaxControlIter=1: that combination
    dss.Solution.Solve()                       # itself raises #485 (see memory note)
    print("  solve  : converged=%s  iterations=%s  nodes=%s"
          % (dss.Solution.Converged(), dss.Solution.Iterations(), dss.Circuit.NumNodes()))
    v = np.array(dss.Circuit.AllBusMagPu())
    print("  V pu   : min=%.4f max=%.4f  nan=%d  (a healthy feeder is ~0.9-1.1)"
          % (np.nanmin(v), np.nanmax(v), int(np.isnan(v).sum())))
    print("  elements: %d  | transformers: %d  | lines: %d"
          % (dss.Circuit.NumCktElements(), dss.Transformers.Count(), dss.Lines.Count()))
except Exception as e:
    print("  EXCEPTION:", type(e).__name__, str(e)[:300])
