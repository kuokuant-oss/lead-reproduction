"""Prepare the deterministic building-budget ladder and its composition audit.

This command is CPU-only and does not fit a model.  It is the required preflight
for the additive building-count curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lead import PROC, ROOT, load_m3_frame
from m5_building_curve_protocol import (
    PROFILES,
    add_building_audit,
    add_cell_composition,
    add_proportional_row_quotas,
    atomic_write_json,
    build_building_profiles,
    build_nested_building_ladder,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument(
        "--sampling-profile", choices=PROFILES, default="representative"
    )
    parser.add_argument(
        "--building-seed", "--seed", dest="building_seed", type=int, default=42
    )
    parser.add_argument("--row-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--early-stop-every", type=int, default=5)
    parser.add_argument(
        "--row-policy",
        choices=("all_rows", "average_building_cap"),
        default="average_building_cap",
    )
    parser.add_argument("--average-building-rows", type=int, default=500)
    parser.add_argument("--max-context-rows", type=int, default=50_000)
    parser.add_argument(
        "--out-root", type=Path, default=PROC / "m5_building_curve" / "protocol"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate in memory without writing artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_m3_frame(verbose=True)
    candidate = frame.loc[frame["building_id"].mod(2).eq(0)]
    profiles = build_building_profiles(candidate)
    ladder, manifest = build_nested_building_ladder(
        profiles,
        args.budgets,
        seed=args.building_seed,
        sampling_profile=args.sampling_profile,
        early_stop_every=args.early_stop_every,
    )
    manifest["experiment"] = "m5_building_count_curve"
    manifest["row_policy"] = args.row_policy
    manifest["average_rows_per_building_limit"] = (
        int(args.average_building_rows)
        if args.row_policy == "average_building_cap"
        else None
    )
    manifest["max_context_rows"] = (
        int(args.max_context_rows)
        if args.row_policy == "average_building_cap"
        else None
    )
    manifest["building_seed"] = int(args.building_seed)
    manifest["row_seed"] = int(args.row_seed)
    manifest["row_selection_seed"] = int(args.row_seed)
    manifest["role_seed"] = None
    manifest["model_seed"] = int(args.model_seed)
    if args.row_policy == "average_building_cap":
        manifest = add_proportional_row_quotas(candidate, manifest)
    manifest = add_cell_composition(candidate, manifest)
    manifest = add_building_audit(candidate, ladder, manifest)
    manifest["split"] = {
        "candidate": "building_id % 2 == 0",
        "canonical_test": "building_id % 2 == 1",
    }
    manifest["creation_command"] = " ".join(sys.argv)
    manifest["source"] = {
        "loader": "lead.load_m3_frame",
        "repository": str(ROOT),
    }

    print(
        f"profile={args.sampling_profile} candidates={len(profiles):,} "
        f"ladder={len(ladder):,}",
        flush=True,
    )
    for budget in manifest["budgets"]:
        cell = manifest["cells"][str(budget)]
        print(
            f"K={budget:>4}: rows={cell['available_rows']:,}, "
            f"anomalies={cell['available_anomalies']:,}, "
            f"tree fit/ES buildings="
            f"{len(cell['tree_fit_buildings'])}/{len(cell['tree_early_stop_buildings'])}",
            flush=True,
        )

    if args.dry_run:
        print(json.dumps(manifest["cells"], indent=2)[:4_000])
        return 0

    output = (
        args.out_root / args.sampling_profile / f"seed{args.building_seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output / "building_profiles.csv", index=False)
    ladder.to_csv(output / "building_ladder.csv", index=False)
    atomic_write_json(output / "building_ladder.json", manifest)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
