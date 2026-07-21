#!/usr/bin/env python3
"""Gate the deployed, identifiable hybrid reconstruction path on unseen grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# The stored terminal target is I_bus + Icomp (= Y V). Older receipts named that
# sum "Ifeat", which read as a third physical quantity; the vocabulary is just
# I_bus and Icomp. Accept the legacy key so historical reports stay readable.
LEGACY_KEYS = {"Ibus_plus_Icomp_wape_pct": ("Ifeat_wape_pct", "Ibus_wape_pct")}


def metric(report: dict, key: str) -> float:
    value = report.get(key)
    if value is None:
        for legacy in LEGACY_KEYS.get(key, ()):
            if legacy in report:
                value = report[legacy]
                break
    if value is None:
        raise SystemExit(f"{report.get('task')}: missing {key}")
    return float(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--threshold-pct", type=float, default=1.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reports = {}
    for task in ("pf", "se_known", "param_one", "injection", "random_safe"):
        path = args.reports_dir / f"{task}.json"
        row = json.loads(path.read_text())
        if row.get("split") != "unseen" or not row.get("structural_safe"):
            raise SystemExit(f"{path}: requires unseen structural-safe report")
        reports[task] = row
    checks = {
        "pf_V": metric(reports["pf"], "V_wape_pct"),
        "pf_Ibus_plus_Icomp": metric(reports["pf"], "Ibus_plus_Icomp_wape_pct"),
        "se_V": metric(reports["se_known"], "V_wape_pct"),
        "se_Ibus_plus_Icomp": metric(reports["se_known"], "Ibus_plus_Icomp_wape_pct"),
        "param_Y": metric(reports["param_one"], "Y_wape_pct"),
        "injection_Icomp": metric(reports["injection"], "Icomp_wape_pct"),
        "random_safe_V": metric(reports["random_safe"], "V_wape_pct"),
        "random_safe_Ibus_plus_Icomp": metric(
            reports["random_safe"], "Ibus_plus_Icomp_wape_pct"),
        "random_safe_Icomp": metric(reports["random_safe"], "Icomp_wape_pct"),
        "random_safe_Y": metric(reports["random_safe"], "Y_wape_pct"),
    }
    payload = {
        "contract": "unseen identifiable hybrid; raw learned heads scored separately",
        "threshold_pct": args.threshold_pct,
        "checks_pct": checks,
        "max_pct": max(checks.values()),
        "mean_pct": sum(checks.values()) / len(checks),
        "pass": all(value <= args.threshold_pct for value in checks.values()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
