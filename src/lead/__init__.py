"""Importable LEAD reproduction pipeline helpers.

The exported helpers preserve M3 reproduction semantics by default. In
particular, ``downsample_indices`` keeps the original positive-duplication
sampling shape, and the M3 scripts keep ``StandardScaler`` in their fit path for
numeric parity even though tree boosters such as LightGBM are scale-invariant.
"""

from .data import (
    BASELINE_FEATURE_COLS,
    BASELINE_FEATURE_COLS_WITH_BUILDING_ID,
    BUILDING_META_FEATURE_COLS,
    CYCLIC_FEATURE_COLS,
    DOWNSAMPLE_SEEDS,
    FUTURE_SHIFTS,
    M3,
    M3_3_EXTRA_FEATURE_COLS,
    MODEL_SEEDS,
    PAST_SHIFTS,
    PROC,
    RANDOM_STATE,
    ROOT,
    SHIFTS,
    SHUFFLE_SEEDS,
    WEATHER_LAG_BASE_COLS,
    WEATHER_WINDOWS,
    load_m3_frame,
)
from .bdg2 import load_bdg2_frame
from .evaluate import classification_metrics
from .features import add_value_change_features
from .io import write_json_with_provenance
from .m5_context import (
    M5_CONTEXT_ROOT,
    M5_CONTEXT_SCHEMA_VERSION,
    M5_HOLDOUT_RULE,
    M5_SENTINEL_SITES,
    M5_FIT_RULE,
    METER_NAME_TO_ID,
    array_sha256,
    build_context_manifest,
    build_query_artifact,
    context_indices,
    context_summary,
    feature_names,
    manifest_path,
    parse_context_tag,
    protocol_source,
    query_paths,
    validate_context_manifest,
)
from .sample import downsample_indices
from .split import assert_no_building_overlap, leave_site_out_mask, split_mask

__all__ = [
    "ROOT",
    "M3",
    "PROC",
    "RANDOM_STATE",
    "DOWNSAMPLE_SEEDS",
    "MODEL_SEEDS",
    "SHUFFLE_SEEDS",
    "BASELINE_FEATURE_COLS",
    "BASELINE_FEATURE_COLS_WITH_BUILDING_ID",
    "BUILDING_META_FEATURE_COLS",
    "CYCLIC_FEATURE_COLS",
    "M3_3_EXTRA_FEATURE_COLS",
    "WEATHER_LAG_BASE_COLS",
    "WEATHER_WINDOWS",
    "SHIFTS",
    "PAST_SHIFTS",
    "FUTURE_SHIFTS",
    "load_m3_frame",
    "load_bdg2_frame",
    "add_value_change_features",
    "split_mask",
    "assert_no_building_overlap",
    "leave_site_out_mask",
    "downsample_indices",
    "classification_metrics",
    "write_json_with_provenance",
    "M5_CONTEXT_ROOT",
    "M5_CONTEXT_SCHEMA_VERSION",
    "M5_FIT_RULE",
    "M5_HOLDOUT_RULE",
    "M5_SENTINEL_SITES",
    "METER_NAME_TO_ID",
    "array_sha256",
    "build_context_manifest",
    "build_query_artifact",
    "context_indices",
    "context_summary",
    "feature_names",
    "manifest_path",
    "parse_context_tag",
    "protocol_source",
    "query_paths",
    "validate_context_manifest",
]
