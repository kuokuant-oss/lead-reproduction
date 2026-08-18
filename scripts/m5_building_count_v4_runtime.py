"""Runtime adapter exposing V4 contexts through the proven V3 cell interface."""

from __future__ import annotations

from pathlib import Path

from m5_building_count_v4_protocol import FixedContext
from m5_building_count_v4_protocol import load_fixed_context as _load_fixed_context


def load_fixed_context(manifest_path: Path, budget: int) -> FixedContext:
    context = _load_fixed_context(manifest_path, budget)
    # The inherited checkpointed cells call this field balance_seed. V4 also
    # records the canonical row_seed/row_draw_seed in the immutable manifest.
    context.manifest["balance_seed"] = context.row_seed
    return context
