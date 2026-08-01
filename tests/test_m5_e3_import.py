"""Tests for the E3 result importer and summariser.

The importer is the only thing standing between a remotely computed result and
the canonical tree, so the tests that matter are the ones proving it *rejects*
things. A validator that has only ever been run on good input is untested.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

try:  # pytest is not a declared dependency of this repository
    import pytest
except ModuleNotFoundError:  # minimal shim so the suite still runs standalone

    class _Skip(Exception):
        pass

    class _Pytest:
        @staticmethod
        def skip(reason):
            raise _Skip(reason)

        class fixture:  # noqa: N801 - mimics the pytest decorator
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

    pytest = _Pytest()  # type: ignore[assignment]
    _SKIP: type[Exception] = _Skip
else:
    _SKIP = getattr(pytest, "skip", Exception).Exception  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CANONICAL = (
    Path(r"C:\Users\tonykuo\projects\lead-reproduction")
    / "data"
    / "processed"
    / "m5_e3_variance_pilot"
)


def _require_results() -> Path:
    if not (CANONICAL / "e3_file_manifest.sha256").exists():
        pytest.skip("E3 results not imported in this environment")
    return CANONICAL


def _validate(staged: Path) -> int:
    """Run the importer in dry-run mode and return its exit code."""
    import m5_e3_import

    argv = sys.argv
    sys.argv = [
        "m5_e3_import.py",
        "--staged",
        str(staged),
        "--canonical",
        str(CANONICAL),
    ]
    try:
        return m5_e3_import.main()
    finally:
        sys.argv = argv


def _rewrite_manifest(root: Path) -> None:
    """Regenerate the manifest so digest checks cannot catch a tampered file."""
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "e3_file_manifest.sha256":
            rel = p.relative_to(root).as_posix()
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{rel}")
    (root / "e3_file_manifest.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_canonical_results_validate():
    assert _validate(_require_results()) == 0


def test_transferred_file_corruption_is_rejected(tmp_path):
    staged = tmp_path / "staged"
    shutil.copytree(_require_results(), staged)
    target = next((staged / "worker_11" / "cell_11" / "repeats").glob("*.json"))
    record = json.loads(target.read_text(encoding="utf-8"))
    record["endpoints"]["steam_positive_vs_hotwater_negative_pairwise_auc"] += 0.05
    target.write_text(json.dumps(record), encoding="utf-8")
    assert _validate(staged) == 1


def test_falsified_gate_is_rejected_even_with_a_consistent_manifest(tmp_path):
    """The check that actually matters.

    A tampered record with a stale manifest is caught by digests alone, which
    proves nothing about the gate recomputation. This falsifies the *recorded
    verdict* and regenerates the manifest over it, so only re-deriving the
    statistic from the raw repeats can catch it.
    """
    staged = tmp_path / "staged"
    shutil.copytree(_require_results(), staged)
    complete = staged / "worker_00" / "cell_00" / "CELL_COMPLETE.json"
    record = json.loads(complete.read_text(encoding="utf-8"))
    record["gate"]["auc_half_width"] = 0.001
    complete.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    _rewrite_manifest(staged)
    assert _validate(staged) == 1


def test_reused_fit_state_is_rejected(tmp_path):
    """Two cells sharing a state digest are not independent fits."""
    staged = tmp_path / "staged"
    shutil.copytree(_require_results(), staged)
    shutil.copy2(
        staged / "worker_00" / "cell_00" / "model.tabpfn_fit",
        staged / "worker_01" / "cell_01" / "model.tabpfn_fit",
    )
    _rewrite_manifest(staged)
    assert _validate(staged) == 1


def test_fresh_runs_outside_the_designated_cell_are_rejected(tmp_path):
    staged = tmp_path / "staged"
    shutil.copytree(_require_results(), staged)
    stray = staged / "worker_00" / "cell_00" / "fresh"
    stray.mkdir()
    shutil.copy2(
        next((staged / "worker_11" / "cell_11" / "fresh").glob("*.json")),
        stray / "fresh_00.json",
    )
    _rewrite_manifest(staged)
    assert _validate(staged) == 1


def test_fresh_runs_are_excluded_from_the_repeat_statistics():
    """Fresh runs must never widen or tighten a same-process interval."""
    root = _require_results()
    summary = json.loads((root / "e3_summary.json").read_text(encoding="utf-8"))
    cell = next(c for c in summary["cells"] if c["cell"] == "11")
    fresh = cell["fresh_process_runs"]
    assert fresh["pooled_into_same_process_statistics"] is False
    assert len(fresh["runs"]) == 2

    # every reported statistic is computed over the repeats alone
    for block in ("gating_endpoints", "non_gating_endpoints"):
        for stat in cell[block].values():
            assert stat["n"] == cell["repeats"]

    # and the fresh scores are genuinely different draws, not copies
    digests = {r["score_sha256"] for r in fresh["runs"]}
    assert len(digests) == 2


def test_decision_matches_the_frozen_rule():
    root = _require_results()
    protocol = json.loads((root / "e3_protocol.json").read_text(encoding="utf-8"))[
        "protocol"
    ]
    decision = json.loads((root / "e3_decision.json").read_text(encoding="utf-8"))
    assert decision["verdict"] in protocol["e3_decision_rule"]
    assert decision["criterion"] == protocol["e3_decision_rule"][decision["verdict"]]
    # a passing verdict must authorise nothing downstream
    assert decision["authorises"] == []
    assert decision["cells_incomplete"] == []
    assert decision["cells_failing_at_cap"] == []


def test_summary_agrees_with_the_raw_records():
    root = _require_results()
    summary = json.loads((root / "e3_summary.json").read_text(encoding="utf-8"))
    assert summary["execution"]["distinct_fit_states"] == 4
    assert summary["execution"]["total_fits"] == 4
    assert summary["execution"]["any_cell_escalated"] is False

    for cell in summary["cells"]:
        cdir = root / f"worker_{cell['cell']}" / f"cell_{cell['cell']}"
        repeats = sorted((cdir / "repeats").glob("*.json"))
        assert len(repeats) == cell["repeats"]
        # repeated inference is not bitwise reproducible; if it ever becomes so,
        # the variance pilot's premise has changed and this must be revisited
        assert cell["distinct_repeat_score_digests"] == cell["repeats"]
        assert cell["bitwise_identical_repeats"] is False


def _standalone() -> int:
    """Run without pytest, since it is not a declared dependency."""
    import tempfile
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if fn.__code__.co_argcount:
                    fn(Path(tmp))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - standalone reporter
                if type(exc).__name__ == "_Skip":
                    print(f"SKIP {name}: {exc}")
                    continue
                failures += 1
                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
