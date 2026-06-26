"""BDG2 ingestion skeleton for M5 transfer-evaluation slices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BDG2_REQUIRED_COLUMNS = ("timestamp", "building_id", "meter", "meter_reading")
BDG2_LABEL_JOIN_COLUMNS = ("building_id", "meter", "timestamp")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["weekday"] = df["timestamp"].dt.weekday.astype("int8")
    df["month"] = df["timestamp"].dt.month.astype("int8")
    df["dayofyear"] = (
        df["timestamp"].dt.dayofyear + df["timestamp"].dt.hour / 24
    ).astype("float32")
    return df


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"BDG2 smoke data not found: {path}")
    return pd.read_csv(path)


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in BDG2_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "BDG2 frame is missing required columns: " + ", ".join(missing)
        )


def _merge_labels(
    frame: pd.DataFrame,
    label_path: Path,
    *,
    label_column: str,
) -> pd.DataFrame:
    labels = _read_required_csv(label_path)
    required = [*BDG2_LABEL_JOIN_COLUMNS, label_column]
    missing = [col for col in required if col not in labels.columns]
    if missing:
        raise ValueError(
            "BDG2 labels are missing required columns: " + ", ".join(missing)
        )
    labels = labels[required].copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = frame.merge(labels, on=list(BDG2_LABEL_JOIN_COLUMNS), how="left")
    if merged[label_column].isna().any():
        raise ValueError("BDG2 labels must align to every loaded row")
    if label_column != "anomaly":
        merged = merged.rename(columns={label_column: "anomaly"})
    merged["anomaly"] = merged["anomaly"].astype("int8")
    return merged


def load_bdg2_frame(
    data_path: str | Path,
    *,
    label_path: str | Path | None = None,
    label_column: str = "anomaly",
    site_id: str | int | None = None,
    require_labels: bool = False,
    allow_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a bounded BDG2 tabular frame for M5 smoke tests.

    This is an ingestion skeleton, not a full BDG2 pull. The caller must provide
    a local single-site CSV or fixture. `allow_download=True` is reserved as an
    explicit future gate and currently raises instead of downloading data.
    """
    data_path = Path(data_path)
    if allow_download:
        raise NotImplementedError(
            "Full BDG2 download is deferred; provide a bounded local site CSV."
        )
    frame = _read_required_csv(data_path)
    _validate_required_columns(frame)
    frame = frame.copy()
    frame = _add_time_features(frame)
    if "site_id" not in frame.columns:
        if site_id is None:
            raise ValueError("BDG2 frame must include site_id or receive site_id=")
        frame["site_id"] = site_id

    if label_path is not None:
        frame = _merge_labels(frame, Path(label_path), label_column=label_column)
    elif label_column in frame.columns:
        if label_column != "anomaly":
            frame = frame.rename(columns={label_column: "anomaly"})
        frame["anomaly"] = frame["anomaly"].astype("int8")
    elif require_labels:
        raise ValueError("BDG2 labels were required but no label source was provided")

    keep_cols = [
        "building_id",
        "site_id",
        "timestamp",
        "meter",
        "meter_reading",
        "hour",
        "weekday",
        "month",
        "dayofyear",
    ]
    if "anomaly" in frame.columns:
        keep_cols.insert(3, "anomaly")
    if verbose:
        label_state = "labeled" if "anomaly" in frame.columns else "unlabeled"
        print(f"Loaded BDG2 {label_state} frame {frame.shape}", flush=True)
    return frame[keep_cols]
