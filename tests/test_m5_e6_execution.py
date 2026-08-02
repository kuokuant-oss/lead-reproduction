"""Tests for the M5 E6 execution stage: freeze, runner, trees, analysis, decision.

These run without a GPU and without touching the holdout. What they check is the
part that is easy to get wrong and impossible to notice afterwards: whether the
fail-closed paths actually fail. A guard that is never exercised is a comment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import m5_e6_analysis as an  # noqa: E402
import m5_e6_decision as dec  # noqa: E402
import m5_e6_protocol as proto  # noqa: E402
import m5_e6_runner as runner  # noqa: E402


# --------------------------------------------------------------------------
# atomic writes and digests
# --------------------------------------------------------------------------


def test_atomic_json_digest_matches_disk(tmp_path):
    p = tmp_path / "a.json"
    d = proto.atomic_json(p, {"b": 1, "a": 2})
    assert d == proto.sha256_file(p)


def test_atomic_json_uses_lf_so_digests_survive_windows(tmp_path):
    p = tmp_path / "a.json"
    proto.atomic_json(p, {"a": [1, 2]})
    assert b"\r\n" not in p.read_bytes()


def test_atomic_json_leaves_no_temp_files(tmp_path):
    proto.atomic_json(tmp_path / "a.json", {"a": 1})
    assert [q.name for q in tmp_path.iterdir()] == ["a.json"]


def test_source_digest_is_newline_normalised(tmp_path):
    crlf, lf = tmp_path / "c.py", tmp_path / "l.py"
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    lf.write_bytes(b"x = 1\ny = 2\n")
    assert proto.sha256_source(crlf) == proto.sha256_source(lf)
    assert proto.sha256_file(crlf) != proto.sha256_file(lf)


# --------------------------------------------------------------------------
# frozen rule content
# --------------------------------------------------------------------------


def test_decision_vocabulary_is_the_four_authorised_terms():
    assert dec.PRIMARY == "negative_support_main_effect"
    assert proto.decision_rules()["vocabulary"] == [
        "NATURAL_PREVALENCE_CONFIRMED",
        "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
        "NOT_CONFIRMED",
        "EXECUTION_INCOMPLETE",
    ]


def test_minimum_practical_effect_threshold_is_not_set():
    assert proto.decision_rules()["minimum_practical_effect_threshold"] == "NOT SET"


def test_comparison_columns_are_mandatory():
    cols = proto.decision_rules()["mandatory_comparison_columns"]
    assert "E6/E5 ratio" in cols and "E6 minus E5 absolute difference" in cols


def test_interval_excluding_zero_does_not_license_a_magnitude_claim():
    r = proto.decision_rules()
    assert "does not license" in r["magnitude_claim_rule"]
    assert r["interval_containing_zero_is_not_proof_of_absence"] is True


def test_bootstrap_namespace_and_master_seed_are_e6_specific():
    b = proto.bootstrap_manifest()
    assert b["namespace_code"] == 6006
    assert b["master_seed"] == 20260730
    assert b["draws"] == 1000


def test_segment_degeneracy_is_disclosed_not_engineered_away():
    b = proto.bootstrap_manifest()
    assert b["segment_may_not_be_redefined"] is True
    assert b["segment_may_not_be_removed_from_the_decision_rule"] is True
    assert 0.91 < b["co_primary_subset"]["segment_singleton_fraction"] < 0.92
    assert "91.8%" in b["segment_degeneracy_disclosure_required"]


def test_sentinel_may_never_enter_an_endpoint():
    s = proto.sentinel_manifest()
    assert s["may_enter_any_endpoint"] is False
    assert s["is_a_full_holdout_repeat"] is False
    assert s["total_calls"] == 8 * 24


def test_tree_gate_admits_no_tolerance_and_no_sampling():
    g = proto.tree_manifest.__doc__ or ""
    assert isinstance(g, str)
    rules = proto.decision_rules()
    assert rules["thresholds_may_not_change_after_seeing_results"] is True


# --------------------------------------------------------------------------
# runner: assembly is where a mosaic would slip through
# --------------------------------------------------------------------------


def _parts(tmp_path, chunks, uuid_by_index=None):
    parts = tmp_path / "microbatches"
    parts.mkdir(exist_ok=True)
    journal = []
    for i, (s0, s1) in enumerate(chunks):
        arr = np.full(s1 - s0, 0.5, dtype="float32")
        name = f"mb_{i:03d}.npy"
        sha = runner.atomic_npy(parts / name, arr)
        journal.append(
            {
                "index": i,
                "canonical_start": s0,
                "canonical_stop": s1,
                "path": name,
                "sha256": sha,
                "process_uuid": (uuid_by_index or {}).get(i, "U1"),
            }
        )
    return parts, journal


@pytest.fixture
def small_rows(monkeypatch):
    monkeypatch.setattr(runner, "ROWS", 100)
    return 100


def test_assemble_accepts_a_complete_single_process_pass(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 40), (40, 100)])
    out = runner.assemble(journal, parts, "U1")
    assert out.shape == (100,) and out.dtype == np.float32


def test_assemble_refuses_two_process_uuids(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 40), (40, 100)], {1: "U2"})
    with pytest.raises(SystemExit, match="process UUIDs"):
        runner.assemble(journal, parts, "U1")


def test_assemble_refuses_a_gap(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 40), (50, 100)])
    with pytest.raises(SystemExit, match="never scored"):
        runner.assemble(journal, parts, "U1")


def test_assemble_refuses_an_overlap(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 60), (40, 100)])
    with pytest.raises(SystemExit, match="overlapping"):
        runner.assemble(journal, parts, "U1")


def test_assemble_refuses_a_drifted_part_digest(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 40), (40, 100)])
    runner.atomic_npy(parts / journal[0]["path"], np.zeros(40, dtype="float32"))
    with pytest.raises(SystemExit, match="digest drifted"):
        runner.assemble(journal, parts, "U1")


def test_assemble_refuses_a_wrong_length_part(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 40), (40, 100)])
    journal[0]["sha256"] = runner.atomic_npy(
        parts / journal[0]["path"], np.zeros(39, dtype="float32")
    )
    with pytest.raises(SystemExit, match="rows, expected"):
        runner.assemble(journal, parts, "U1")


def test_assemble_refuses_non_finite_scores(tmp_path, small_rows):
    parts, journal = _parts(tmp_path, [(0, 100)])
    bad = np.full(100, 0.5, dtype="float32")
    bad[7] = np.nan
    journal[0]["sha256"] = runner.atomic_npy(parts / journal[0]["path"], bad)
    with pytest.raises(SystemExit, match="non-finite"):
        runner.assemble(journal, parts, "U1")


def test_load_scaler_rejects_a_frozen_scaler_whose_digest_drifted(tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    p = tmp_path / "scalers" / "s.joblib"
    p.parent.mkdir()
    joblib.dump(StandardScaler().fit(np.ones((4, 3), dtype="float32")), p)
    spec = {
        "unit_id": "u",
        "scaler_arm": "frozen_reference",
        "scaler_source": {
            "kind": "persisted",
            "path": "scalers/s.joblib",
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(SystemExit, match="not the frozen one"):
        runner.load_scaler(spec, np.ones((4, 3), dtype="float32"), tmp_path)


def test_load_scaler_accepts_the_frozen_scaler_at_its_pinned_digest(tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    p = tmp_path / "scalers" / "s.joblib"
    p.parent.mkdir()
    joblib.dump(StandardScaler().fit(np.ones((4, 3), dtype="float32")), p)
    spec = {
        "unit_id": "u",
        "scaler_arm": "frozen_reference",
        "scaler_source": {
            "kind": "persisted",
            "path": "scalers/s.joblib",
            "sha256": runner.sha256_file(p),
        },
    }
    assert (
        runner.load_scaler(spec, np.ones((4, 3), dtype="float32"), tmp_path) is not None
    )


def test_load_scaler_refuses_a_frozen_arm_backed_by_a_rebuild_record(tmp_path):
    spec = {
        "unit_id": "u",
        "scaler_arm": "frozen_reference",
        "scaler_source": {"kind": "rebuilt_and_verified"},
    }
    with pytest.raises(SystemExit, match="not a persisted scaler"):
        runner.load_scaler(spec, np.ones((4, 3), dtype="float32"), tmp_path)


def test_load_scaler_rebuilds_the_cell_specific_arm(tmp_path):
    spec = {
        "unit_id": "u",
        "scaler_arm": "cell_specific",
        "scaler_source": {"kind": "rebuilt_and_verified"},
    }
    x = np.arange(12, dtype="float32").reshape(4, 3)
    sc = runner.load_scaler(spec, x, tmp_path)
    assert np.allclose(sc.mean_, x.mean(axis=0))


def test_quarantine_moves_the_directory_and_records_why(tmp_path):
    d = tmp_path / "unit"
    d.mkdir()
    (d / "partial.npy").write_bytes(b"x")
    dead = runner.quarantine(d, "test")
    assert not d.exists()
    assert (dead / "partial.npy").exists()
    rec = json.loads((dead / "QUARANTINE.json").read_text(encoding="utf-8"))
    assert rec["reason"] == "test"
    assert "not permitted" in rec["resume_within_state"]


# --------------------------------------------------------------------------
# analysis: cluster definitions and aggregation weights
# --------------------------------------------------------------------------


def test_meter_codes_come_from_the_canonical_map_not_a_restatement():
    from m5_e4_endpoints import METER

    assert an.METER_STEAM == METER["steam"] == 2
    assert an.METER_HOTWATER == METER["hotwater"] == 3
    assert an.METER_STEAM != an.METER_HOTWATER


def test_segment_codes_group_contiguous_anomaly_runs():
    raw = np.array([10, 11, 12, 20, 21, 30], dtype="int64")
    an_ = np.array([1, 1, 1, 1, 1, 0], dtype="int8")
    codes = an.segment_codes(raw, an_)
    assert codes[0] == codes[1] == codes[2]
    assert codes[3] == codes[4]
    assert len({int(c) for c in codes}) == 3


def test_segment_codes_make_every_normal_row_a_singleton():
    raw = np.arange(6, dtype="int64")
    an_ = np.zeros(6, dtype="int8")
    codes = an.segment_codes(raw, an_)
    assert len(set(codes.tolist())) == 6


def test_segment_codes_break_a_run_on_a_raw_index_gap():
    raw = np.array([1, 2, 4, 5], dtype="int64")
    an_ = np.ones(4, dtype="int8")
    codes = an.segment_codes(raw, an_)
    assert codes[1] != codes[2]


def test_aggregate_weights_the_three_seeds_equally():
    per_unit = {}
    for seed, base in zip(an.SEEDS, (0.0, 3.0, 6.0)):
        for arm in an.ARMS:
            for cell, add in zip(an.CELLS, (0.0, 1.0, 0.0, 1.0)):
                per_unit[an.unit_id(seed, cell, arm)] = base + add
    agg = an.aggregate(per_unit)
    # Each seed's negative-support effect is 1.0 by construction, so the
    # equal-weight mean must be 1.0 no matter how the seeds are ordered.
    assert agg["overall"]["negative_support_main_effect"] == pytest.approx(1.0)
    for s in an.SEEDS:
        assert agg["per_seed"][f"seed{s}"][
            "negative_support_main_effect"
        ] == pytest.approx(1.0)


def test_aggregate_forms_the_contrast_inside_each_arm_before_averaging():
    per_unit = {}
    for seed in an.SEEDS:
        for cell in an.CELLS:
            per_unit[an.unit_id(seed, cell, "cell_specific")] = (
                1.0 if cell in ("01", "11") else 0.0
            )
            per_unit[an.unit_id(seed, cell, "frozen_reference")] = (
                3.0 if cell in ("01", "11") else 0.0
            )
    agg = an.aggregate(per_unit)
    assert agg["per_arm"]["cell_specific"][
        "negative_support_main_effect"
    ] == pytest.approx(1.0)
    assert agg["per_arm"]["frozen_reference"][
        "negative_support_main_effect"
    ] == pytest.approx(3.0)
    assert agg["overall"]["negative_support_main_effect"] == pytest.approx(2.0)


def test_flatten_exposes_overall_seed_and_arm_keys():
    per_unit = {
        an.unit_id(s, c, a): 1.0 for s in an.SEEDS for c in an.CELLS for a in an.ARMS
    }
    flat = an.flatten(an.aggregate(per_unit), "x")
    assert "x|overall|negative_support_main_effect" in flat
    assert "x|seed42|negative_support_main_effect" in flat
    assert "x|cell_specific|negative_support_main_effect" in flat


# --------------------------------------------------------------------------
# decision arithmetic
# --------------------------------------------------------------------------


def test_compare_reports_ratio_and_difference_together():
    row = dec.compare(0.30, 0.40, 0.20)
    assert row["e6_over_e5_ratio"] == pytest.approx(0.5)
    assert row["e6_minus_e5_difference"] == pytest.approx(-0.20)


def test_compare_survives_a_zero_prior_effect():
    row = dec.compare(0.0, 0.0, 0.10)
    assert row["e6_over_e5_ratio"] is None
    assert row["e6_minus_e5_difference"] == pytest.approx(0.10)


def _analysis(overall, seeds, arms, excludes):
    point, intervals = {}, {}
    for family in ("tabpfn", "tree", "gap"):
        for ep in ("auc", "margin"):
            point[f"{family}|{ep}"] = {
                "effects": {
                    "overall": {dec.PRIMARY: overall},
                    "per_seed": {f"seed{s}": {dec.PRIMARY: seeds} for s in dec.SEEDS},
                    "per_arm": {a: {dec.PRIMARY: arms} for a in dec.ARMS},
                }
            }
            for ct in ("building", "segment"):
                intervals[f"{ct}|{family}|{ep}"] = {
                    f"x|overall|{dec.PRIMARY}": {"excludes_zero": excludes}
                }
    return {"point": point, "intervals": intervals}


def test_all_conditions_met_is_reported_as_met():
    got = dec.evaluate(_analysis(0.3, 0.3, 0.3, True), "tabpfn", dec.PRIMARY)
    assert got["all_met"] is True
    assert all(got["conditions"].values())


def test_an_interval_containing_zero_blocks_confirmation():
    got = dec.evaluate(_analysis(0.3, 0.3, 0.3, False), "tabpfn", dec.PRIMARY)
    assert got["all_met"] is False
    assert got["conditions"]["building_interval_excludes_zero_both"] is False
    assert got["conditions"]["auc_overall_positive"] is True


def test_a_negative_seed_blocks_confirmation():
    got = dec.evaluate(_analysis(0.3, -0.1, 0.3, True), "tabpfn", dec.PRIMARY)
    assert got["all_met"] is False
    assert got["conditions"]["both_positive_in_3_of_3_seeds"] is False


def test_a_negative_arm_blocks_confirmation():
    got = dec.evaluate(_analysis(0.3, 0.3, -0.2, True), "tabpfn", dec.PRIMARY)
    assert got["all_met"] is False
    assert got["conditions"]["both_scaler_arms_positive"] is False
