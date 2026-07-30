"""Prepare M5 context manifests, a fixed query artifact, and a CPU audit.

This command is intentionally CPU-only.  It freezes the row-level inputs for
the expensive Story A/E probe; model fitting and holdout scoring happen later.

Example::

    uv run python scripts/prepare_m5_context_artifacts.py
    uv run python scripts/prepare_m5_context_artifacts.py --contexts pooled_reference meter_balanced meter_heavy:hotwater:0.5 meter_excluded:hotwater
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lead import (
    M5_CONTEXT_ROOT,
    ROOT,
    array_sha256,
    build_context_manifest,
    build_query_artifact,
    context_indices,
    context_summary,
    load_m3_frame,
    manifest_path,
    protocol_source,
    validate_context_manifest,
)
from lead.m5_context import atomic_json, query_paths


DEFAULT_CONTEXTS = (
    "pooled_reference",
    "meter_balanced",
    "meter_heavy:hotwater:0.5",
    "meter_excluded:hotwater",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--context-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--query-rows-per-cell", type=int, default=16)
    parser.add_argument("--story", default="A_E_composition")
    parser.add_argument("--out-root", type=Path, default=M5_CONTEXT_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the plan without writing artifacts",
    )
    return parser.parse_args(argv)


def _source_provenance() -> dict[str, Any]:
    return {
        "frame": "M3 load_m3_frame(verbose=False)",
        "raw_index": "frame positional index, preserved from train.csv",
        "label_source": "bad_meter_readings.csv positional contract",
        "split": {"fit": "building_id % 2 == 0", "holdout": "building_id % 2 == 1"},
    }


def _write_query(
    frame: Any, source: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    manifest, indices = build_query_artifact(
        frame,
        holdout_rows=source["holdout_rows"],
        seed=args.query_seed,
        rows_per_cell=args.query_rows_per_cell,
    )
    raw = np.asarray(indices, dtype="int64")
    lookup = frame.iloc[raw]
    manifest_path_out, npz_path = query_paths(args.out_root, manifest["query_set"])
    manifest["source_artifact"] = _source_provenance()
    manifest["creation_command"] = " ".join(sys.argv)
    if not args.dry_run:
        atomic_json(manifest_path_out, manifest)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            raw_index=raw,
            anomaly=lookup["anomaly"].to_numpy(dtype="int8"),
            meter=lookup["meter"].to_numpy(dtype="int8"),
            site_id=lookup["site_id"].to_numpy(dtype="int8"),
            building_id=lookup["building_id"].to_numpy(dtype="int16"),
        )
    print(
        f"query {manifest['query_set']}: {len(raw):,} rows, digest {array_sha256(raw)[:16]}"
    )
    return manifest


def _audit_row(context_tag: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "story": manifest["story"],
        "context_tag": context_tag,
        "context_rows": manifest["context_rows"],
        "context_seed": manifest["context_seed"],
        "raw_index_sha256": manifest["raw_index_sha256"],
        "unique_rows": manifest["unique_rows"],
        "positive": manifest["label_counts"]["positive"],
        "negative": manifest["label_counts"]["negative"],
        "building_count": manifest["building_count"],
        "meter_counts": json.dumps(manifest["meter_counts"], sort_keys=True),
        "meter_label_counts": json.dumps(
            manifest["meter_label_counts"], sort_keys=True
        ),
        "site_counts": json.dumps(manifest["site_counts"], sort_keys=True),
        "holdout_overlap": 0,
        "duplicate_raw_index": int(manifest["unique_rows"] != manifest["context_rows"]),
    }


def write_audit(
    rows: list[dict[str, Any]], *, root: Path, story: str, dry_run: bool
) -> tuple[Path, Path]:
    csv_path = root / "reports" / "context_composition_audit.csv"
    md_path = ROOT / "docs" / "reports" / "m5-context-composition-audit.md"
    if dry_run:
        return csv_path, md_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# M5 context composition audit",
        "",
        "CPU-only audit of the exact context rows used by the Story A/E probe.",
        "The contexts are label-balanced, fit-building-only, and identified by an ordered raw-index digest.",
        "",
        "| context | rows | positive | negative | buildings | raw-index digest |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['context_tag']}` | {row['context_rows']:,} | {row['positive']:,} | "
            f"{row['negative']:,} | {row['building_count']:,} | `{row['raw_index_sha256'][:16]}` |"
        )
    lines.extend(
        [
            "",
            "Gates: `holdout_overlap=0`, `duplicate_raw_index=0`, exact 50/50 label counts, and the full ordered digest is stored in each manifest.",
            "",
            f"CSV artifact: `{csv_path.relative_to(ROOT).as_posix()}`",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.context_rows <= 0 or args.context_rows % 2:
        raise SystemExit("--context-rows must be a positive even integer")
    print("Loading frozen M3 frame for CPU artifact preparation", flush=True)
    frame = load_m3_frame(verbose=True)
    source = protocol_source(
        frame, seed=args.seed, validation_rows=args.validation_rows
    )
    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
    _write_query(frame, source, args)
    audit_rows: list[dict[str, Any]] = []
    for context_tag in args.contexts:
        indices = context_indices(
            frame,
            context_rows=args.context_rows,
            context_tag=context_tag,
            seed=args.seed,
            candidate_rows=source["candidate_rows"],
        )
        split = {
            "fit_rule": source["fit_rule"],
            "holdout_rule": source["holdout_rule"],
            "validation_rows_excluded": len(source["validation_rows"]),
            "candidate_rows": len(source["candidate_rows"]),
        }
        for feature_tag in ("F0", "F4"):
            manifest = build_context_manifest(
                frame,
                indices,
                story=args.story,
                context_tag=context_tag,
                context_rows=args.context_rows,
                context_seed=args.seed,
                model_seed=args.model_seed,
                feature_tag=feature_tag,
                source_artifact=_source_provenance(),
                split=split,
                creation_command=" ".join(sys.argv),
            )
            validate_context_manifest(
                frame, manifest, holdout_rows=source["holdout_rows"]
            )
            if feature_tag == "F0":
                audit_rows.append(_audit_row(context_tag, manifest))
            destination = manifest_path(
                args.out_root,
                story=args.story,
                context_tag=context_tag,
                context_rows=args.context_rows,
                seed=args.seed,
            )
            if not args.dry_run:
                atomic_json(
                    destination.with_name(
                        f"{destination.stem}_{feature_tag.lower()}.json"
                    ),
                    manifest,
                )
                if feature_tag == "F0":
                    # Keep the paper-plan path as the canonical context
                    # manifest; feature-specific aliases make the F0/F4 grid
                    # explicit without changing the context identity.
                    atomic_json(destination, manifest)
        summary = context_summary(frame, indices)
        print(
            f"context {context_tag}: {summary['rows']:,} rows, "
            f"labels {summary['label_counts']}, digest {summary['raw_index_sha256'][:16]}"
        )
    csv_path, md_path = write_audit(
        audit_rows, root=args.out_root, story=args.story, dry_run=args.dry_run
    )
    print(f"audit CSV: {csv_path}")
    print(f"audit report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
