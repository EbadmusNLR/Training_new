#!/usr/bin/env python3
"""Build a fail-closed foundation acceptance receipt from task reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required report: {path}")
    return json.loads(path.read_text())


def worst_scale(report: dict, role: str) -> dict:
    rows = [
        {"field": key.removesuffix("_wape_pct"), "pct": float(value)}
        for key, value in report.items()
        if key.startswith("field_")
        and f"_{role}" in key
        and key.endswith("_scale_wape_pct")
    ]
    if not rows:
        raise SystemExit(f"report {report.get('task')} has no scale-normalized {role} fields")
    return max(rows, key=lambda row: row["pct"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--threshold-pct", type=float, default=1.0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.reports_dir
    pf = load(root / "pf.json")
    pf_tree = load(root / "pf_tree.json")
    se = load(root / "se_known.json")
    param = load(root / "param_one.json")
    injection = load(root / "injection.json")
    random = load(root / "random.json")
    checks = {
        "pf_voltage": float(pf["V_wape_pct"]),
        "pf_current_direct": float(pf["Ibus_wape_pct"]),
        "pf_current_structural": float(pf_tree["Ibus_wape_pct"]),
        "se_voltage": float(se["V_wape_pct"]),
        "se_current": float(se["Ibus_wape_pct"]),
        "parameter_Y": float(param["Y_wape_pct"]),
        "injection_Icomp": float(injection["Icomp_wape_pct"]),
        "worst_Y_scale_field": worst_scale(param, "Y")["pct"],
        "worst_Icomp_scale_field": worst_scale(injection, "Icomp")["pct"],
        "random_voltage": float(random["V_wape_pct"]),
        "random_current": float(random["Ibus_wape_pct"]),
        "random_Y": float(random["Y_wape_pct"]),
        "random_Icomp": float(random["Icomp_wape_pct"]),
    }
    threshold = float(args.threshold_pct)
    payload = {
        "reports_dir": str(root),
        "checkpoint": pf.get("checkpoint"),
        "threshold_pct": threshold,
        "checks_pct": checks,
        "feasibility": {
            key: pf.get(key) for key in (
                "feasibility_accuracy", "feasibility_precision", "feasibility_recall",
                "feasibility_f1", "feasibility_violation_rate",
            )
        },
        "passed": all(value <= threshold for value in checks.values()),
        "failures": sorted(key for key, value in checks.items() if value > threshold),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
