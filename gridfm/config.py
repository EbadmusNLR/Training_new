"""Small YAML inheritance helper for reproducible ablation configs."""
from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge(load_config(parent_path), raw)

