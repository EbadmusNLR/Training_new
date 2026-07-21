"""Core corpus schema, masking, and physics.

Migrated out of the old DG_FM_Training tree so Training_new is self-contained;
nothing references DG_FM_Training any more. Import sites use ``gridfm.legacy``,
which now re-exports from here.
"""
from . import data, masking, physics  # noqa: F401

__all__ = ["data", "masking", "physics"]
