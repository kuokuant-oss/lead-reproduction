"""Validate staged E3 results, then import them into the canonical root.

Validation runs against the staged copy only. Nothing touches the canonical
root until every check has passed, and the import itself is a directory-level
`os.replace` per cell, so a canonical cell is either the old one or the new one
and never a half-written mixture.

The precision gate is recomputed here from the raw per-repeat endpoints rather
than trusted from the runner's own `CELL_COMPLETE.json`. If the runner and this
importer disagree about a half-width, that is a hard failure -- the point of
re-deriving it is that a silent agreement is the only acceptable outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e3_runner import evaluate_gate  # noqa: E402

MANIFEST = "e3_file_manifest.sha256"

# The manifest attests to what the *remote* produced. These are derived locally
# afterwards, are regenerable from the attested files by `m5_e3_summary.py`, and
# so are expected to be absent from it. Nothing else may be.
LOCALLY_DERIVED = frozenset(
    {MANIFEST, "e3_summary.json", "e3_decision.json", "e3_input_manifest.json"}
)
REQUIRED_CELL_FILES = (
    "fit_start.json",
    "fit_complete.json",
    "CELL_COMPLETE.json",
    "gate_log.json",
    "model.tabpfn_fit",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(failures: list[str], ok: bool, message: str) -> None:
    if not ok:
        failures.append(message)


def validate_manifest(staged: Path, failures: list[str]) -> int:
    manifest_path = staged / MANIFEST
    if not manifest_path.exists():
        failures.append(f"missing {MANIFEST}")
        return 0
    entries = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        entries[rel.removeprefix("./")] = digest
    for rel, want in entries.items():
        path = staged / rel
        if not path.exists():
            failures.append(f"manifest lists a missing file: {rel}")
        elif sha256_file(path) != want:
            failures.append(f"digest mismatch after transfer: {rel}")
    on_disk = {
        p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()
    }
    for rel in sorted(on_disk - set(entries) - LOCALLY_DERIVED):
        failures.append(f"file present but absent from the remote manifest: {rel}")
    return len(entries)


def validate_cell(staged: Path, cell: str, proto: dict, failures: list[str]) -> dict:
    root = staged / f"worker_{cell}" / f"cell_{cell}"
    for name in REQUIRED_CELL_FILES:
        check(failures, (root / name).exists(), f"cell {cell}: missing {name}")
    check(
        failures,
        not (root / "INTERRUPTED.json").exists(),
        f"cell {cell}: INTERRUPTED marker present",
    )
    if failures:
        return {}

    fit = read_json(root / "fit_complete.json")
    complete = read_json(root / "CELL_COMPLETE.json")
    state_digest = sha256_file(root / "model.tabpfn_fit")
    check(
        failures,
        state_digest == fit["state_sha256"],
        f"cell {cell}: state digest differs from the recorded fit",
    )

    repeats = [read_json(p) for p in sorted((root / "repeats").glob("*.json"))]
    check(failures, bool(repeats), f"cell {cell}: no repeats recorded")
    check(
        failures,
        len({r["score_sha256"] for r in repeats}) == len(repeats),
        f"cell {cell}: duplicate score digests among repeats",
    )

    # Re-derive the gate instead of trusting the runner's own verdict. The
    # targets come from the protocol, not from the recorded gate, so a runner
    # that gated itself against the wrong number cannot pass validation.
    recorded = complete["gate"]
    bounded = proto["inherited"]["bounded_metric_ci_half_width_target"]
    ref = proto["reference_iqr"]["per_cell"][cell]
    margin_target = (
        ref["reference_iqr"]
        * proto["inherited"]["continuous_margin_ci_half_width_iqr_multiplier"]
    )
    check(
        failures,
        abs(margin_target - ref["margin_half_width_target"]) <= 1e-15,
        f"cell {cell}: the protocol's own margin target is not IQR x multiplier",
    )
    recomputed = evaluate_gate(repeats, bounded, margin_target)
    check(
        failures,
        recomputed["both_pass"] and recorded["both_pass"],
        f"cell {cell}: gate did not pass on both endpoints",
    )
    for key in ("auc_half_width", "margin_half_width", "auc_target", "margin_target"):
        got, was = recomputed[key], recorded[key]
        check(
            failures,
            abs(got - was) <= 1e-12,
            f"cell {cell}: {key} does not reproduce ({was} -> {got})",
        )
    check(
        failures,
        len(repeats) == complete["repeats"] == recorded["n"],
        f"cell {cell}: repeat count disagrees with CELL_COMPLETE",
    )
    check(
        failures,
        complete["status"] == "COMPLETE_GATE_PASSED",
        f"cell {cell}: status is {complete['status']}",
    )

    fresh = sorted((root / "fresh").glob("*.json")) if (root / "fresh").is_dir() else []
    spec = proto["fresh_process_diagnostic"]
    if cell != spec["cell"]:
        check(
            failures,
            not fresh,
            f"cell {cell}: fresh runs exist for a cell the protocol did not designate",
        )
    for path in fresh:
        run = read_json(path)
        check(
            failures,
            run["excluded_from_same_process_statistics"] is True
            and run["scientific_estimate"] is False,
            f"cell {cell}: {path.name} is not flagged as excluded and non-scientific",
        )
        check(
            failures,
            run["state_sha256"] == state_digest,
            f"cell {cell}: {path.name} reloaded a different state",
        )
        check(
            failures,
            run["score_sha256"] not in {r["score_sha256"] for r in repeats},
            f"cell {cell}: {path.name} duplicates a same-process repeat digest",
        )

    return {
        "cell": cell,
        "state_sha256": state_digest,
        "repeats": len(repeats),
        "fresh_runs": len(fresh),
        "gate": recomputed,
        "fit_seconds": fit.get("fit_seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="import after validating")
    args = ap.parse_args()

    canonical_proto = args.canonical / "e3_protocol.json"
    staged_proto = args.staged / "e3_protocol.json"
    failures: list[str] = []

    check(failures, canonical_proto.exists(), "canonical protocol artifact is missing")
    check(failures, staged_proto.exists(), "staged protocol artifact is missing")
    if failures:
        print("\n".join(failures))
        return 1

    proto_digest = sha256_file(canonical_proto)
    check(
        failures,
        sha256_file(staged_proto) == proto_digest,
        "staged results were produced under a different protocol artifact",
    )
    proto = read_json(canonical_proto)["protocol"]

    n_files = validate_manifest(args.staged, failures)

    order = proto["supplementary_decisions"]["1_schedule_seed"]["realised_cell_order"]
    staged_cells = sorted(
        p.name.removeprefix("worker_") for p in args.staged.glob("worker_*")
    )
    check(
        failures,
        staged_cells == sorted(order),
        f"staged cells {staged_cells} do not match the protocol's {sorted(order)}",
    )

    cells = []
    if not failures:
        for cell in order:
            cells.append(validate_cell(args.staged, cell, proto, failures))

    if not failures:
        digests = [c["state_sha256"] for c in cells]
        check(
            failures,
            len(set(digests)) == len(digests),
            "two cells share a fit state digest, so they are not independent fits",
        )

    print(f"files verified      : {n_files}")
    print(f"protocol digest     : {proto_digest}")
    print(f"cells validated     : {[c.get('cell') for c in cells]}")
    for c in cells:
        if c:
            print(
                f"  cell {c['cell']}: repeats={c['repeats']} fresh={c['fresh_runs']} "
                f"gate={c['gate']['both_pass']} state={c['state_sha256'][:8]} "
                f"hw_auc={c['gate']['auc_half_width']:.6f} "
                f"hw_margin={c['gate']['margin_half_width']:.6f}"
            )
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nVALIDATION PASSED")

    if not args.apply:
        print("dry run; pass --apply to import")
        return 0

    args.canonical.mkdir(parents=True, exist_ok=True)
    for cell in order:
        src = args.staged / f"worker_{cell}"
        dst = args.canonical / f"worker_{cell}"
        staging = args.canonical / f".incoming_worker_{cell}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
        if dst.exists():
            shutil.rmtree(dst)
        staging.replace(dst)
        print(f"imported worker_{cell}")
    shutil.copy2(args.staged / MANIFEST, args.canonical / MANIFEST)
    print("IMPORT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
