"""Pinned access to the validated DG_FM_Training data/physics contract.

Only decoding, masking, and physical metric helpers are reused. The learned model and
training/evaluation policy are implemented in Training_new.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ROOT / "DG_FM_Training"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

data = importlib.import_module("data")
masking = importlib.import_module("masking")
physics = importlib.import_module("physics")

FC = data.FC
PE_DIM_EXT = data.PE_DIM_EXT
SPECS = data.SPECS
build_datasets = data.build_datasets
i_offset = data.i_offset
n_slots = data.n_slots
store_width = data.store_width
y_width = data.y_width
