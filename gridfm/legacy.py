"""Access to the validated corpus data/masking/physics contract.

These modules used to be imported out of the sibling ``DG_FM_Training`` tree via a
``sys.path`` hack. They now live in :mod:`gridfm.core` so Training_new is
self-contained and nothing references DG_FM_Training. The module name is kept so
the ~22 existing import sites (`from gridfm.legacy import ...`) keep working.

Only decoding, masking, and physical metric helpers live here. The learned model
and training/evaluation policy are implemented in Training_new.
"""
from __future__ import annotations

from .core import data, masking, physics

FC = data.FC
PE_DIM_EXT = data.PE_DIM_EXT
SPECS = data.SPECS
build_datasets = data.build_datasets
i_offset = data.i_offset
n_slots = data.n_slots
store_width = data.store_width
y_width = data.y_width

__all__ = [
    "data", "masking", "physics", "FC", "PE_DIM_EXT", "SPECS",
    "build_datasets", "i_offset", "n_slots", "store_width", "y_width",
]
