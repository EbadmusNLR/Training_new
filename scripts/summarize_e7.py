#!/usr/bin/env python3
"""Select on unseen det2f metrics and report the untouched test result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CANDIDATES = (
    "e7_det2f_norm400", "e7_det2f_norm1000", "e7_det2f_norm2000",
    "e7_det2f_wape400", "e7_det2f_wape1000", "e7_det2f_wape2000",
    "e7_det2f_h256_400", "e7_det2f_h256_1000", "e7_det2f_h256_2000",
)


def read(name: str, filename: str) -> dict | None:
    path = RUNS / name / filename
    return json.loads(path.read_text()) if path.is_file() else None


def score(row: dict) -> float:
    # Current is the binding target; retain a strong voltage penalty so a
    # current-only solution cannot win. Selection never reads the test split.
    return float(row["Ibus_wape_pct"]) + 5.0 * float(row["V_wape_pct"])


def winner() -> str:
    rows = [(name, read(name, "unseen_source_kcl.json")) for name in CANDIDATES]
    complete = [(name, row) for name, row in rows if row is not None]
    if len(complete) != len(CANDIDATES):
        missing = [name for name, row in rows if row is None]
        raise SystemExit(f"missing det2f unseen evaluations: {missing}")
    return min(complete, key=lambda item: score(item[1]))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    selected = winner()
    if args.select:
        print(selected)
        return 0
    print("run,unseen_V_WAPE,unseen_Ibus_WAPE,score")
    for name in CANDIDATES:
        row = read(name, "unseen_source_kcl.json")
        if row:
            print(f"{name},{row['V_wape_pct']:.6f},{row['Ibus_wape_pct']:.6f},{score(row):.6f}")
    print(f"selected,{selected}")
    test = read(selected, "test_source_kcl.json")
    if test:
        print(f"test_V_WAPE,{test['V_wape_pct']:.6f}")
        print(f"test_Ibus_WAPE,{test['Ibus_wape_pct']:.6f}")
    direct = read(selected, "test_direct.json")
    if direct:
        print(f"test_direct_V_WAPE,{direct['V_wape_pct']:.6f}")
        print(f"test_direct_Ibus_WAPE,{direct['Ibus_wape_pct']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
