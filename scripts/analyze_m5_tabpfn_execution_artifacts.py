"""Read-only comparison of immutable TabPFN sentinel lifecycle artifacts.

This deliberately does not invoke TabPFN.  It makes differences visible without
changing an already completed lifecycle stage, and writes a new versioned summary
rather than replacing an earlier partial comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lead import ROOT


AUDIT = (
    ROOT
    / "data"
    / "processed"
    / "m5_hotwater_label_factorial"
    / "deterministic_execution_audit"
)


@dataclass(frozen=True)
class Stage:
    label: str
    path: Path
    device: str
    metadata: Path | None = None


STAGES = {
    "P1_fit_time_gpu": Stage(
        "P1_fit_time_gpu", AUDIT / "predictions" / "P1_fit_immediate.npz", "cuda"
    ),
    "P2_live_gpu_repeat": Stage(
        "P2_live_gpu_repeat", AUDIT / "predictions" / "P2_live_repeat.npz", "cuda"
    ),
    "P4_same_process_gpu_reload": Stage(
        "P4_same_process_gpu_reload",
        AUDIT / "predictions" / "P4_same_process_gpu_load.npz",
        "cuda",
    ),
    "P5_fresh_process_gpu_reload": Stage(
        "P5_fresh_process_gpu_reload",
        AUDIT / "predictions" / "P5_fresh_process_gpu_load.npz",
        "cuda",
        AUDIT / "stage_p5.json",
    ),
    "P6_fresh_gpu_reload_1": Stage(
        "P6_fresh_gpu_reload_1",
        AUDIT / "fresh_gpu" / "P6_fresh_gpu_reload_1.npz",
        "cuda",
        AUDIT / "fresh_gpu" / "P6_fresh_gpu_reload_1.json",
    ),
}

PAIRS = (
    ("P1_fit_time_gpu", "P2_live_gpu_repeat"),
    ("P4_same_process_gpu_reload", "P5_fresh_process_gpu_reload"),
    ("P1_fit_time_gpu", "P4_same_process_gpu_reload"),
    ("P1_fit_time_gpu", "P5_fresh_process_gpu_reload"),
    ("P2_live_gpu_repeat", "P4_same_process_gpu_reload"),
    ("P2_live_gpu_repeat", "P5_fresh_process_gpu_reload"),
    ("P5_fresh_process_gpu_reload", "P6_fresh_gpu_reload_1"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(stage_name: str) -> dict[str, np.ndarray]:
    with np.load(STAGES[stage_name].path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.abs(left.astype("float64") - right.astype("float64"))
    return {"mae": float(delta.mean()), "max_abs": float(delta.max())}


def top_rows(
    delta: np.ndarray, raw_index: np.ndarray, n: int = 10
) -> list[dict[str, float | int]]:
    per_row = delta.max(axis=1)
    return [
        {
            "raw_index": int(raw_index[index]),
            "max_probability_abs": float(per_row[index]),
        }
        for index in np.argsort(per_row)[-n:][::-1]
    ]


def compare(
    left_name: str, right_name: str, raw_index: np.ndarray
) -> dict[str, object]:
    left, right = values(left_name), values(right_name)
    proba_delta = np.abs(
        left["aggregate_probability"].astype("float64")
        - right["aggregate_probability"].astype("float64")
    )
    per_delta = np.abs(
        left["per_estimator_probability"].astype("float64")
        - right["per_estimator_probability"].astype("float64")
    )
    derived_left = np.log(
        np.clip(left["aggregate_probability"][:, 1], 1e-7, 1 - 1e-7)
        / np.clip(left["aggregate_probability"][:, 0], 1e-7, 1 - 1e-7)
    )
    derived_right = np.log(
        np.clip(right["aggregate_probability"][:, 1], 1e-7, 1 - 1e-7)
        / np.clip(right["aggregate_probability"][:, 0], 1e-7, 1 - 1e-7)
    )
    raw_logits = None
    if "raw_logits" in left and "raw_logits" in right:
        raw_logits = scalar_delta(left["raw_logits"], right["raw_logits"])
    per_row = per_delta.max(axis=(0, 2))
    result = {
        "left": left_name,
        "right": right_name,
        "same_device": STAGES[left_name].device == STAGES[right_name].device,
        "proba": scalar_delta(
            left["aggregate_probability"], right["aggregate_probability"]
        ),
        "proba_spearman": float(
            pd.Series(left["positive_score"]).corr(
                pd.Series(right["positive_score"]), method="spearman"
            )
        ),
        "changed_rows": int(np.count_nonzero(proba_delta.max(axis=1))),
        "derived_aggregate_log_odds": scalar_delta(derived_left, derived_right),
        "direct_predict_logits": "unavailable: legacy P1-P5 artifacts did not retain direct predict_logits",
        "per_estimator_probability": scalar_delta(
            left["per_estimator_probability"], right["per_estimator_probability"]
        ),
        "per_estimator_raw_logits": raw_logits
        if raw_logits is not None
        else "unavailable: both stages must retain raw_logits",
        "top_difference_rows": top_rows(proba_delta, raw_index),
        "top_per_estimator_difference_rows": [
            {
                "raw_index": int(raw_index[index]),
                "max_per_estimator_probability_abs": float(per_row[index]),
            }
            for index in np.argsort(per_row)[-10:][::-1]
        ],
    }
    return result


def stage_contract(stage_name: str) -> dict[str, object]:
    stage = STAGES[stage_name]
    output = {
        "file": str(stage.path.relative_to(AUDIT)),
        "sha256": sha(stage.path),
        "mtime_ns": stage.path.stat().st_mtime_ns,
        "device": stage.device,
    }
    if stage.metadata is not None:
        output["metadata"] = json.loads(stage.metadata.read_text(encoding="utf-8"))
    return output


def main() -> int:
    with np.load(AUDIT / "inputs.npz") as data:
        raw_index = np.asarray(data["query_raw_index"], dtype="int64")
    provenance = json.loads((AUDIT / "provenance.json").read_text(encoding="utf-8"))
    comparisons = [compare(left, right, raw_index) for left, right in PAIRS]
    result = {
        "audit_root": "data/processed/m5_hotwater_label_factorial/deterministic_execution_audit",
        "scope": "P1/P2/P4/P5/P6 only; P7 was not run or included",
        "stage_contracts": {name: stage_contract(name) for name in STAGES},
        "frozen_contract": {
            key: provenance[key]
            for key in (
                "manifest_raw_index_sha256",
                "query_raw_index_sha256",
                "raw_train",
                "scaled_train",
                "raw_query",
                "scaled_query",
                "scaler_sha256",
                "checkpoint",
            )
        },
        "p5_fresh_process_provenance_limit": "P5 was launched with subprocess.run in the original audit, but PID, PPID, command line, and start time were not persisted; that historical detail cannot be reconstructed after exit.",
        "comparisons": comparisons,
        "first_same_gpu_boundary": comparisons[0],
        "interpretation": "P1-to-P2 is the first observable same-GPU difference, before any save or load. P2-to-P4 remains a save-or-reload combined boundary because the revised lifecycle schema did not retain a qualifying post-save live prediction.",
    }
    target = AUDIT / "artifact_comparison_lifecycle_p1_p6.json"
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable comparison summary: {target}"
        )
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
