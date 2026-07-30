"""Build deterministic F4 hotwater positive/negative-support factorial contexts.

Each seed begins with one 20k pooled-reference context.  Its hotwater-positive
and hotwater-negative slots are independently replaced with same-label,
non-hotwater reserve rows, so every factorial cell has identical length, label
counts, slot order, and non-hotwater conditional meter allocation rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lead import (
    ROOT,
    array_sha256,
    build_context_manifest,
    context_indices,
    context_summary,
    load_m3_frame,
    protocol_source,
    validate_context_manifest,
)
from lead.m5_context import atomic_json, query_paths, stable_row_priority


OUT = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"
SEEDS = (42, 123, 999)
CELLS = ((True, True), (True, False), (False, True), (False, False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--context-rows", type=int, default=20_000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument(
        "--query-root",
        type=Path,
        default=ROOT / "data" / "processed" / "m5_context_stories",
    )
    return parser.parse_args()


def cell_id(positive_present: bool, negative_present: bool) -> str:
    return f"hw_pos_{'present' if positive_present else 'excluded'}__hw_neg_{'present' if negative_present else 'excluded'}"


def quota(weights: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder meter allocation, used for replacement reserves."""
    values = np.asarray(weights, dtype="float64")
    if total == 0:
        return np.zeros(len(values), dtype="int64")
    if values.sum() <= 0:
        raise AssertionError(
            "no non-hotwater rows available for replacement allocation"
        )
    desired = values / values.sum() * total
    result = np.floor(desired).astype("int64")
    for index in np.argsort(-(desired - result), kind="stable")[
        : total - int(result.sum())
    ]:
        result[index] += 1
    return result


def reserve_replacements(
    frame: Any,
    *,
    pooled: np.ndarray,
    candidates: np.ndarray,
    label: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Return one deterministic replacement for every hotwater slot of label."""
    pooled_rows = frame.iloc[pooled]
    slots = np.flatnonzero(
        (pooled_rows["meter"].to_numpy() == 3)
        & (pooled_rows["anomaly"].to_numpy() == label)
    )
    non_hotwater = pooled_rows.loc[
        (pooled_rows["meter"] != 3) & (pooled_rows["anomaly"] == label), "meter"
    ]
    meters = np.asarray(
        sorted(int(value) for value in non_hotwater.unique()), dtype="int8"
    )
    allocations = quota(
        np.asarray([(non_hotwater == meter).sum() for meter in meters], dtype="int64"),
        len(slots),
    )
    available = np.zeros(len(frame), dtype=bool)
    available[candidates] = True
    available[pooled] = False
    labels = frame["anomaly"].to_numpy()
    meter_values = frame["meter"].to_numpy()
    replacement_by_meter: dict[int, np.ndarray] = {}
    for meter, count in zip(meters, allocations, strict=True):
        eligible = np.flatnonzero(
            available & (labels == label) & (meter_values == meter)
        ).astype("int64")
        priorities = stable_row_priority(
            eligible, seed=seed + 10_000 * label + int(meter)
        )
        order = np.lexsort((eligible, priorities))
        if len(eligible) < count:
            raise AssertionError(
                f"insufficient label={label}, meter={meter} reserve rows"
            )
        replacement_by_meter[int(meter)] = eligible[order[:count]]
    replacement = np.concatenate([replacement_by_meter[int(meter)] for meter in meters])
    replacement_meter = np.concatenate(
        [
            np.repeat(meter, count)
            for meter, count in zip(meters, allocations, strict=True)
        ]
    )
    replacement_order = np.lexsort(
        (replacement, stable_row_priority(replacement, seed=seed + 100_000 + label))
    )
    replacement = replacement[replacement_order]
    replacement_meter = replacement_meter[replacement_order]
    records = [
        {
            "slot": int(slot),
            "label": int(label),
            "original_raw_index": int(pooled[slot]),
            "replacement_raw_index": int(new),
            "replacement_meter": int(meter),
        }
        for slot, new, meter in zip(slots, replacement, replacement_meter, strict=True)
    ]
    return slots, records


def validate_factorial_manifest(
    frame: Any, manifest: dict[str, Any], pooled: np.ndarray
) -> None:
    validate_context_manifest(frame, manifest)
    indices = np.asarray(manifest["raw_index"], dtype="int64")
    factor = manifest["factorial"]
    pooled_rows = frame.iloc[pooled]
    rows = frame.iloc[indices]
    for label, key in (
        (1, "hotwater_positive_present"),
        (0, "hotwater_negative_present"),
    ):
        relevant = (pooled_rows["meter"].to_numpy() == 3) & (
            pooled_rows["anomaly"].to_numpy() == label
        )
        if factor[key]:
            if not np.array_equal(indices[relevant], pooled[relevant]):
                raise AssertionError("present support changed a pooled-reference slot")
        elif np.any(
            (rows["meter"].to_numpy()[relevant] == 3)
            | (rows["anomaly"].to_numpy()[relevant] != label)
        ):
            raise AssertionError(
                "excluded support slot is not a same-label non-hotwater replacement"
            )
    if manifest["pooled_reference_raw_index_sha256"] != array_sha256(pooled):
        raise AssertionError("pooled-reference digest drifted")


def main() -> int:
    args = parse_args()
    if args.context_rows != 20_000 or args.context_rows % 2:
        raise SystemExit("this preregistered factorial requires --context-rows 20000")
    query_manifest, query_npz = query_paths(args.query_root, "screening")
    if not query_manifest.is_file() or not query_npz.is_file():
        raise FileNotFoundError("existing fixed screening query artifact is required")
    args.out_root.mkdir(parents=True, exist_ok=True)
    print("Loading frozen M3 frame for factorial manifests", flush=True)
    frame = load_m3_frame(verbose=True)
    audit_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, int]] = []
    for seed in args.seeds:
        source = protocol_source(frame, seed=seed, validation_rows=args.validation_rows)
        pooled = context_indices(
            frame,
            context_rows=args.context_rows,
            context_tag="pooled_reference",
            seed=seed,
            candidate_rows=source["candidate_rows"],
        )
        pooled_digest = array_sha256(pooled)
        slot_maps: dict[int, tuple[np.ndarray, list[dict[str, int]]]] = {
            label: reserve_replacements(
                frame,
                pooled=pooled,
                candidates=source["candidate_rows"],
                label=label,
                seed=seed,
            )
            for label in (0, 1)
        }
        cell_indices: dict[str, np.ndarray] = {}
        for positive_present, negative_present in CELLS:
            name = cell_id(positive_present, negative_present)
            indices = pooled.copy()
            for label, present in ((1, positive_present), (0, negative_present)):
                slots, records = slot_maps[label]
                if not present:
                    indices[slots] = np.asarray(
                        [row["replacement_raw_index"] for row in records], dtype="int64"
                    )
                    replacement_rows.extend(
                        {"context_seed": seed, "cell_id": name, **row}
                        for row in records
                    )
            factor = {
                "hotwater_positive_present": positive_present,
                "hotwater_negative_present": negative_present,
                "replacement_policy": "same-label non-hotwater reserve; quota follows pooled non-hotwater conditional meter mix; fixed slots",
            }
            manifest = build_context_manifest(
                frame,
                indices,
                story="hotwater_label_role_factorial",
                context_tag="pooled_reference",
                context_rows=args.context_rows,
                context_seed=seed,
                model_seed=args.model_seed,
                feature_tag="F4",
                split={
                    "fit_rule": source["fit_rule"],
                    "holdout_rule": source["holdout_rule"],
                    "validation_rows_excluded": len(source["validation_rows"]),
                },
                source_artifact={
                    "pooled_reference_raw_index_sha256": pooled_digest,
                    "query_manifest": str(query_manifest.resolve()),
                },
                creation_command=" ".join(sys.argv),
            )
            manifest.update(
                {
                    "factorial_cell_id": name,
                    "factorial": factor,
                    "pooled_reference_raw_index_sha256": pooled_digest,
                    "ordered_row_slots": "raw_index list order is fixed across all four cells",
                    "replacement_map": f"replacements/seed{seed}_{name}.csv",
                }
            )
            validate_factorial_manifest(frame, manifest, pooled)
            path = args.out_root / "manifests" / f"seed{seed}" / f"{name}.json"
            atomic_json(path, manifest)
            cell_indices[name] = indices
            summary = context_summary(frame, indices)
            audit_rows.append(
                {
                    "context_seed": seed,
                    "cell_id": name,
                    "positive_present": positive_present,
                    "negative_present": negative_present,
                    "raw_index_sha256": summary["raw_index_sha256"],
                    "rows": summary["rows"],
                    "positive": summary["label_counts"]["positive"],
                    "negative": summary["label_counts"]["negative"],
                    "meter_label_counts": summary["meter_label_counts"],
                }
            )
        for left in cell_indices:
            for right in cell_indices:
                if left < right:
                    overlap_rows.append(
                        {
                            "context_seed": seed,
                            "left_cell": left,
                            "right_cell": right,
                            "overlap_rows": int(
                                len(
                                    np.intersect1d(
                                        cell_indices[left], cell_indices[right]
                                    )
                                )
                            ),
                            "overlap_fraction": float(
                                len(
                                    np.intersect1d(
                                        cell_indices[left], cell_indices[right]
                                    )
                                )
                                / args.context_rows
                            ),
                        }
                    )
        print(
            f"seed {seed}: pooled digest {pooled_digest[:16]}; wrote four validated cells",
            flush=True,
        )
    import pandas as pd

    replacement_columns = [
        "slot",
        "label",
        "original_raw_index",
        "replacement_raw_index",
        "replacement_meter",
    ]
    replacement_frame = pd.DataFrame(replacement_rows)
    replacement_root = args.out_root / "replacements"
    replacement_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for positive_present, negative_present in CELLS:
            name = cell_id(positive_present, negative_present)
            path = replacement_root / f"seed{seed}_{name}.csv"
            selection = (
                pd.DataFrame(columns=replacement_columns)
                if replacement_frame.empty
                else replacement_frame.loc[
                    (replacement_frame["context_seed"] == seed)
                    & (replacement_frame["cell_id"] == name),
                    replacement_columns,
                ]
            )
            selection.to_csv(path, index=False)
    report_root = args.out_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        report_root / "factorial_composition_audit.csv", index=False
    )
    pd.DataFrame(overlap_rows).to_csv(
        report_root / "factorial_cell_overlap.csv", index=False
    )
    print(
        f"wrote {len(audit_rows)} manifests and composition/overlap audits under {args.out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
