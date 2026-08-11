"""Safe entrypoint for the portable M5 V2 building-seed 47--51 extension."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from lead import PROC, ROOT

try:
    from prepare_m5_building_count_v2_seed47_51 import (
        AUDIT_ROOT,
        CANONICAL_HOLDOUT,
        check_raw_files,
        validate_audit_bundle,
        validate_canonical_holdout,
    )
except ModuleNotFoundError:  # Package import used by unittest discovery.
    from scripts.prepare_m5_building_count_v2_seed47_51 import (
        AUDIT_ROOT,
        CANONICAL_HOLDOUT,
        check_raw_files,
        validate_audit_bundle,
        validate_canonical_holdout,
    )


DEFAULT_OUT_ROOT = PROC / "m5_building_curve" / "v2"
DEFAULT_VALIDATION_OUT_ROOT = (
    PROC / "m5_building_curve" / "NON_SCIENTIFIC_VALIDATION_seed47_51"
)
DEFAULT_REPORT = (
    ROOT / "docs" / "reports" / "m5-building-count-experiment_V2_seed47-51.md"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "validation", "formal"), default="plan"
    )
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--validation-out-root", type=Path, default=DEFAULT_VALIDATION_OUT_ROOT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-context-rows", type=int, default=200)
    parser.add_argument("--validation-holdout-rows", type=int, default=200)
    parser.add_argument("--retry-delay", type=int, default=120)
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--finalize-retries", type=int, default=2)
    parser.add_argument("--push-retries", type=int, default=5)
    parser.add_argument("--git-push-timeout", type=int, default=120)
    parser.add_argument("--gpu-wait-checks", type=int, default=30)
    parser.add_argument("--publish-results", action="store_true")
    return parser.parse_args(argv)


def command_for_args(args: argparse.Namespace) -> list[str]:
    if args.mode == "validation":
        return [
            sys.executable,
            str(ROOT / "scripts" / "run_m5_building_count_v2.py"),
            "--audit-root",
            str(args.audit_root),
            "--out-root",
            str(args.validation_out_root),
            "--mode",
            "validation",
            "--families",
            "tree",
            "tabpfn",
            "--model-seed",
            "42",
            "--validation-context-rows",
            str(args.validation_context_rows),
            "--validation-holdout-rows",
            str(args.validation_holdout_rows),
        ]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_m5_building_count_v2_overnight.py"),
        "--audit-root",
        str(args.audit_root),
        "--out-root",
        str(args.out_root),
        "--report",
        str(args.report),
        "--mode",
        args.mode,
        "--pair-order",
        "budget-major",
        "--model-seed",
        "42",
        "--retry-delay",
        str(args.retry_delay),
        "--unit-retries",
        str(args.unit_retries),
        "--finalize-retries",
        str(args.finalize_retries),
        "--push-retries",
        str(args.push_retries),
        "--git-push-timeout",
        str(args.git_push_timeout),
        "--gpu-wait-checks",
        str(args.gpu_wait_checks),
    ]
    if args.publish_results:
        command.append("--publish-results")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_audit_bundle(args.audit_root)
    check_raw_files()
    validate_canonical_holdout(CANONICAL_HOLDOUT)
    if (
        args.mode == "formal"
        and args.report.resolve()
        == (ROOT / "docs" / "reports" / "m5-building-count-experiment_V2.md").resolve()
    ):
        raise SystemExit("seed47-51 must use its separate tracked progress report")
    command = command_for_args(args)
    print(f"Executing seed47-51 entrypoint: {shlex.join(command)}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    raise SystemExit(main())
