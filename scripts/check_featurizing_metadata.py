#!/usr/bin/env python3
"""Verify pu->feature conversion preserves passive definitions exactly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT)]

from datakit.core.scenario_store import FeederScenarios  # noqa: E402
from gridfm.featurizing import (  # noqa: E402
    FEAT_EPS,
    PASSIVE_DEFINITION_FIELDS,
    SCENARIO_Y_FIELDS,
    _scenario_build_feat_sample,
)


FAMILY = {
    "line": "Line", "capacitor": "Capacitor", "reactor": "Reactor",
    "transformer": "Transformer", "vsource": "Vsource", "load": "Load",
    "generator": "Generator", "pvsystem": "PVSystem", "storage": "Storage",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feeder-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = FeederScenarios(args.feeder_dir)[0]
    families = {}
    all_families = set(FAMILY.values()) | {"TriplexLine", "TriplexLoad"}
    scaler = {
        "epsilon": 1e-12,
        "current": {
            family: {"I_scale": 1.0, "transform": "asinh"}
            for family in all_families
        },
        "admittance": {
            family: {
                "transform": "asinh", "Y_r_diag_scale": 1.0,
                "Y_r_offdiag_scale": 1.0, "Y_i_diag_scale": 1.0,
                "Y_i_offdiag_scale": 1.0,
            }
            for family in all_families
        },
    }
    for store, family in FAMILY.items():
        if store not in raw.node_types or not raw[store].keys():
            continue
        tensors = [value for value in raw[store].values() if torch.is_tensor(value)]
        if not tensors:
            continue
        n = int(tensors[0].shape[0])
        families[store] = [family] * n
    feat = _scenario_build_feat_sample(raw, scaler, families)
    for store, (dim, y_fields) in SCENARIO_Y_FIELDS.items():
        if store not in families:
            continue
        rows, cols = torch.tril_indices(dim, dim)
        for pu_field, feat_field, _part in y_fields:
            truth = getattr(raw[store], pu_field).reshape(-1, dim, dim)[:, rows, cols]
            decoded = torch.sinh(getattr(feat[store], feat_field)) * (1.0 + FEAT_EPS)
            if not torch.allclose(decoded, truth, rtol=1e-6, atol=1e-9):
                raise AssertionError(f"Y feature roundtrip failed: {store}.{feat_field}")
        spec_icomp = getattr(raw[store], "Icomp_r_pu", None)
        for term in range(1, 4):
            source = getattr(raw[store], f"I_r_bus{term}_pu", None)
            encoded = getattr(feat[store], f"I_r_bus{term}_feat", None)
            if not torch.is_tensor(source) or not torch.is_tensor(encoded):
                continue
            expected = source.clone()
            if torch.is_tensor(spec_icomp):
                start = (term - 1) * 4
                expected += spec_icomp[:, start:start + 4]
            decoded = torch.sinh(encoded) * (1.0 + FEAT_EPS)
            if not torch.allclose(decoded, expected, rtol=1e-6, atol=1e-9):
                raise AssertionError(f"I_bus + Icomp feature failed: {store}.bus{term}")
    checked = 0
    for store, fields in PASSIVE_DEFINITION_FIELDS.items():
        if store not in raw.node_types:
            continue
        for field in fields:
            source = getattr(raw[store], field, None)
            if not torch.is_tensor(source):
                continue
            target = getattr(feat[store], field, None)
            if not torch.is_tensor(target) or not torch.equal(source, target):
                raise AssertionError(f"definition field changed: {store}.{field}")
            checked += 1
    if checked == 0:
        raise AssertionError("no passive definition fields found")
    print(f"FEATURIZING_METADATA_OK fields={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
