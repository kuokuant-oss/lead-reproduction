"""Expose V5 frozen contexts through the proven balanced-context interface."""

from __future__ import annotations

from pathlib import Path

from m5_building_count_v5_protocol import FixedContext
from m5_building_count_v5_protocol import load_fixed_context as _load_fixed_context


def load_fixed_context(manifest_path: Path, budget: int) -> FixedContext:
    return _load_fixed_context(manifest_path, budget)
