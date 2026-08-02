"""Tests for the M5 E6 design audit.

The two that carry weight are the equivalence proofs: the scalable estimators
must equal the naive resampling path value for value, on synthetic data and on
the real E5 192-row data. Everything else guards a rule the design depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:  # pytest is not a declared dependency of this repository
    import pytest
except ModuleNotFoundError:  # minimal shim so the suite still runs standalone

    class _Skip(Exception):
        pass

    class _Approx:
        def __init__(self, value, rel=None, abs=None):
            self.value, self.rel, self.abs = value, rel, abs

        def __eq__(self, other):
            tol = (
                self.abs
                if self.abs is not None
                else abs(self.value) * (self.rel if self.rel is not None else 1e-9)
                + 1e-12
            )
            return abs(other - self.value) <= tol

    class _Pytest:
        @staticmethod
        def approx(value, rel=None, abs=None):
            return _Approx(value, rel, abs)

        @staticmethod
        def skip(reason):
            raise _Skip(reason)

    pytest = _Pytest()  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

MAIN = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
DRAFT = MAIN / "data" / "processed" / "m5_e6_protocol_draft"
E5 = MAIN / "data" / "processed" / "m5_e5_independent_replication"

HOLDOUT_ROWS = 10_137_155
HOLDOUT_DIGEST = "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"

from m5_e6_clustered import (  # noqa: E402
    CLUSTER_CODE,
    DRAWS,
    MASTER_SEED,
    NAMESPACE_CODE,
    SortedAUC,
    cluster_multiplicities,
    draw_generator,
    naive_auc,
    naive_margin,
    naive_resampled_rows,
    weighted_margin,
)


def _synthetic(n=400, n_clusters=37, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, n_clusters, size=n)
    positive = rng.random(n) < 0.3
    # deliberately coarse, so ties occur and the tie handling is exercised
    score = np.round(rng.normal(0, 1, n), 2)
    return score, positive, codes, n_clusters


# --------------------------------------------------------------------------
# the equivalence proofs
# --------------------------------------------------------------------------


def test_weighted_auc_equals_naive_resampling_on_synthetic_data():
    score, positive, codes, k = _synthetic()
    fast = SortedAUC(score, positive)
    for draw_id in range(40):
        rng = draw_generator("building", draw_id)
        mult = cluster_multiplicities(rng, codes, k)
        rows = naive_resampled_rows(codes, mult)
        want = naive_auc(score, positive, rows)
        got = fast(mult)
        if np.isnan(want):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-12), f"draw {draw_id}"


def test_weighted_auc_handles_ties_like_roc_auc_score():
    """Every score identical: the answer must be exactly 0.5, not 0 or 1."""
    score = np.zeros(50)
    positive = np.zeros(50, dtype=bool)
    positive[:20] = True
    fast = SortedAUC(score, positive)
    assert fast(np.ones(50)) == pytest.approx(0.5, rel=1e-12)
    assert naive_auc(score, positive, np.arange(50)) == pytest.approx(0.5, rel=1e-12)


def test_weighted_margin_equals_naive_resampling_on_synthetic_data():
    score, positive, codes, k = _synthetic()
    for draw_id in range(40):
        rng = draw_generator("segment", draw_id)
        mult = cluster_multiplicities(rng, codes, k)
        rows = naive_resampled_rows(codes, mult)
        want = naive_margin(score, positive, rows)
        got = weighted_margin(score, positive, mult)
        if np.isnan(want):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-12), f"draw {draw_id}"


def test_estimators_equal_naive_on_the_real_e5_data():
    """The same proof on real scores, not just synthetic ones."""
    qpath = (
        MAIN
        / "data/processed/m5_hotwater_label_factorial/independent_query/queries.npz"
    )
    unit = E5 / "seed42__cell11__cell_specific" / "repeats" / "repeat_000.json"
    if not (qpath.exists() and unit.exists()):
        pytest.skip("E5 results not present in this environment")

    with np.load(qpath, allow_pickle=True) as z:
        meter = np.asarray(z["meter"], dtype="int8")
        anom = np.asarray(z["anomaly"], dtype="int8")
        building = np.asarray(z["building_id"], dtype="int64")
    score = np.asarray(json.loads(unit.read_text(encoding="utf-8"))["score"], "float64")

    keep = ((meter == 2) & (anom == 1)) | ((meter == 3) & (anom == 0))
    s, p, b = score[keep], (meter[keep] == 2), building[keep]
    names, codes = np.unique(b, return_inverse=True)
    k = names.size
    fast = SortedAUC(s, p)
    for draw_id in range(25):
        mult = cluster_multiplicities(draw_generator("building", draw_id), codes, k)
        rows = naive_resampled_rows(codes, mult)
        want_a, want_m = naive_auc(s, p, rows), naive_margin(s, p, rows)
        if not np.isnan(want_a):
            assert fast(mult) == pytest.approx(want_a, rel=1e-12), f"auc draw {draw_id}"
        if not np.isnan(want_m):
            assert weighted_margin(s, p, mult) == pytest.approx(want_m, rel=1e-12)


# --------------------------------------------------------------------------
# addressable RNG
# --------------------------------------------------------------------------


def test_e6_namespace_differs_from_e4_and_e5():
    assert (MASTER_SEED, NAMESPACE_CODE) == (20260730, 6006)
    assert CLUSTER_CODE == {"building": 1, "segment": 2}
    codes = np.arange(30) % 6
    a = cluster_multiplicities(draw_generator("building", 5), codes, 6)
    for other in (4004, 5005):
        b = cluster_multiplicities(draw_generator("building", 5, other), codes, 6)
        assert not np.array_equal(a, b)


def test_draws_are_addressable_and_loop_order_independent():
    codes = np.arange(60) % 11
    fwd = [
        cluster_multiplicities(draw_generator("segment", d), codes, 11)
        for d in range(15)
    ]
    bwd = {
        d: cluster_multiplicities(draw_generator("segment", d), codes, 11)
        for d in reversed(range(15))
    }
    for d in range(15):
        assert np.array_equal(fwd[d], bwd[d])


def test_draw_id_outside_the_frozen_range_is_refused():
    for bad in (-1, DRAWS, DRAWS + 1):
        try:
            draw_generator("building", bad)
        except ValueError:
            continue
        raise AssertionError(f"draw_id {bad} should have been refused")


# --------------------------------------------------------------------------
# row identity and the shard plan
# --------------------------------------------------------------------------


def _row_manifest():
    p = DRAFT / "e6_row_manifest.json"
    if not p.exists():
        pytest.skip("E6 row manifest not built in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_row_manifest_pins_the_holdout_identity():
    m = _row_manifest()["row_set"]
    assert m["rows"] == HOLDOUT_ROWS
    assert m["unique_raw_index"] == HOLDOUT_ROWS
    assert m["sorted_raw_index_sha256"] == HOLDOUT_DIGEST
    assert m["building_id_parity"] == [1]
    assert m["buildings"] == 724
    assert m["sites"] == 16
    assert m["anomaly_rows"] == 637_397
    assert m["disjoint_from_fit_half"] is True


def test_row_manifest_does_not_read_the_score_column():
    m = _row_manifest()
    assert "tabpfn" not in m["columns_read"]
    assert "tabpfn" in m["columns_present_but_not_read"]


def test_required_and_forbidden_wording_is_recorded():
    m = _row_manifest()
    assert "previously characterised holdout rows" in m["required_wording"]
    for banned in ("untouched holdout", "first contact", "previously unseen row set"):
        assert banned in m["forbidden_wording"]


def _shards():
    p = DRAFT / "e6_shard_manifest.json"
    if not p.exists():
        pytest.skip("E6 shard manifest not built in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_shard_union_is_the_holdout_and_shards_are_disjoint():
    s = _shards()
    plan = s["plan"]
    assert sum(x["rows"] for x in plan) == HOLDOUT_ROWS
    covered = []
    for x in sorted(plan, key=lambda y: y["canonical_position_start"]):
        covered.append((x["canonical_position_start"], x["canonical_position_end"]))
    assert covered[0][0] == 0
    assert covered[-1][1] == HOLDOUT_ROWS
    for (a0, a1), (b0, _b1) in zip(covered, covered[1:]):
        assert a1 == b0, "shards must tile the holdout with no gap and no overlap"
        assert a0 < a1


def test_every_shard_is_scored_by_all_24_states():
    for x in _shards()["plan"]:
        assert x["states_to_score"] == 24
        assert x["may_be_split_by_state"] is False


def test_state_based_machine_assignment_is_forbidden():
    s = _shards()
    assert "state-based machine assignment is forbidden" in s["rule"]


# --------------------------------------------------------------------------
# the cost model and the draft
# --------------------------------------------------------------------------


def _cost():
    p = DRAFT / "e6_cost_model.json"
    if not p.exists():
        pytest.skip("E6 cost model not built in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_cost_model_is_measured_not_extrapolated_from_e5():
    c = _cost()
    assert "not extrapolated" in c["basis"]
    for key in ("tabpfn_rows_per_second_steady", "tree_rows_per_second"):
        assert c["measured_inputs"][key]["value"] > 0
        assert c["measured_inputs"][key]["source"]


def test_cost_model_covers_all_three_repeat_policies():
    names = {p["policy"] for p in _cost()["repeat_policies"]}
    assert names == {"R1", "R8", "R1_PLUS_SENTINEL"}
    by = {p["policy"]: p for p in _cost()["repeat_policies"]}
    assert by["R8"]["row_scores"] == 8 * by["R1"]["row_scores"]
    assert by["R1_PLUS_SENTINEL"]["row_scores"] == by["R1"]["row_scores"]
    assert by["R1_PLUS_SENTINEL"]["sentinel_repeats_per_state"] == 8
    assert by["R1_PLUS_SENTINEL"]["sentinel_rows"] < HOLDOUT_ROWS


def test_existing_feature_artifacts_are_recorded_as_not_reusable():
    """They are already scaled, so they cannot carry E6's 24 per-unit scalers."""
    r = _cost()["raw_feature_rebuild"]
    assert r["required"] is True
    assert "already scaled" in r["reason"]
    assert r["built_once_shared_by_all_24_states"] is True


def test_segment_singleton_degeneracy_is_recorded():
    """91.8% singletons is the fact that makes the segment interval need a ruling."""
    cp = _cost()["strata"]["co_primary"]
    assert cp["segment_singleton_clusters"] == 545_430
    assert cp["segment_singleton_fraction"] > 0.9
    assert cp["building_clusters"] == 215


def _draft():
    p = DRAFT / "e6_protocol.DRAFT.json"
    if not p.exists():
        pytest.skip("E6 protocol draft not built in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_draft_is_marked_unlaunchable():
    d = _draft()
    assert d["not_a_launch_artifact"] is True
    assert "NOT FROZEN" in d["status"]
    assert "protocol freeze" in d["prohibitions_this_round"]
    assert "launch" in d["prohibitions_this_round"]


def test_draft_keeps_e4_e5_science_unchanged():
    inh = _draft()["inherited_unchanged"]
    assert inh["context_seeds"] == [42, 123, 999]
    assert inh["persisted_states"] == 24
    assert inh["row_probability_averaging"] == "forbidden"
    assert inh["cross_cell_repeat_pairing"] == "forbidden"
    assert inh["model_seed_factor"] == "not added"


def test_draft_has_no_practical_threshold_yet():
    d = _draft()["decision_rule_candidate"]
    assert d["minimum_practical_effect_threshold"].startswith("NOT SET")
    assert len(_draft()["open_items_requiring_human_ruling"]) >= 4


def test_draft_forbids_the_untouched_wording():
    w = _draft()["wording_constraints"]
    assert "previously characterised holdout rows" in w["required"]
    for banned in ("untouched holdout", "first contact"):
        assert banned in w["forbidden"]
    assert "sole mechanism" in w["mechanism"]


# --------------------------------------------------------------------------
# the microbatch manifest and the corrected census
# --------------------------------------------------------------------------

PROTOCOL = MAIN / "data" / "processed" / "m5_e6_protocol"


def _microbatches():
    p = PROTOCOL / "e6_microbatch_manifest.json"
    if not p.exists():
        pytest.skip("E6 microbatch manifest not built in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_microbatches_tile_the_holdout_exactly_once():
    m = _microbatches()
    mbs = sorted(m["microbatches"], key=lambda x: x["canonical_start"])
    assert mbs[0]["canonical_start"] == 0
    assert mbs[-1]["canonical_stop"] == HOLDOUT_ROWS
    total = 0
    for a, b in zip(mbs, mbs[1:]):
        assert a["canonical_stop"] == b["canonical_start"], "gap or overlap"
        total += a["row_count"]
    total += mbs[-1]["row_count"]
    assert total == HOLDOUT_ROWS
    assert m["census"]["rows_covered"] == HOLDOUT_ROWS


def test_no_microbatch_exceeds_the_maximum():
    m = _microbatches()
    assert all(x["row_count"] <= m["microbatch_max_rows"] for x in m["microbatches"])
    assert all(x["row_count"] > 0 for x in m["microbatches"])


def test_call_census_rejects_the_hand_written_audit_numbers():
    """The audit divided rows by microbatch size. Every shard ends short."""
    c = _microbatches()["census"]
    assert c["microbatches_per_state"] == 516
    assert c["full_holdout_predict_proba_calls_all_states"] == 12_384
    assert c["r8_full_holdout_calls_all_states"] == 99_072
    # the superseded figures must not reappear
    assert c["full_holdout_predict_proba_calls_all_states"] != 12_165
    assert c["r8_full_holdout_calls_all_states"] != 97_317


def test_sentinel_census_is_24_by_8():
    c = _microbatches()["census"]
    assert c["sentinel_predict_proba_calls_per_state"] == 8
    assert c["sentinel_predict_proba_calls_all_states"] == 192
    assert c["r1_plus_sentinel_total_calls"] == 12_384 + 192
    assert c["sentinel_row_scores"] == 352 * 8 * 24


def test_cost_model_call_counts_come_from_the_manifest():
    by = {p["policy"]: p for p in _cost()["repeat_policies"]}
    c = _microbatches()["census"]
    assert (
        by["R1"]["predict_proba_calls_full_holdout"]
        == (c["full_holdout_predict_proba_calls_all_states"])
    )
    assert (
        by["R8"]["predict_proba_calls_full_holdout"]
        == (c["r8_full_holdout_calls_all_states"])
    )
    assert (
        by["R1_PLUS_SENTINEL"]["predict_proba_calls_total"]
        == (c["r1_plus_sentinel_total_calls"])
    )
    assert by["R1"]["microbatches_per_state"] == 516


def test_failure_recomputation_is_a_whole_state_not_a_microbatch():
    """A microbatch checkpoint is progress, not scientific completion."""
    r = _cost()["restart_cost"]
    assert r["unit_of_recomputation"] == "one complete state pass"
    assert r["max_recompute_rows_on_failure"] == HOLDOUT_ROWS
    assert r["microbatch_checkpoints_are_progress_only"] is True
    assert r["max_recompute_rows_on_failure"] != 20_000


def _standalone() -> int:
    import traceback

    passed = failed = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "_Skip":
                print(f"SKIP {name}: {exc}")
                skipped += 1
                continue
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
