"""Tests for the frozen M5 E5 independent replication protocol.

E5's whole claim rests on two things being true: that nothing was fitted, and
that the verdict rules were fixed before any score existed. Most of these tests
attack those two points directly.
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

        class fixture:  # noqa: N801
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

    pytest = _Pytest()  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CANONICAL = (
    Path(r"C:\Users\tonykuo\projects\lead-reproduction")
    / "data"
    / "processed"
    / "m5_e5_independent_replication"
)
FACTORIAL = (
    Path(r"C:\Users\tonykuo\projects\lead-reproduction")
    / "data"
    / "processed"
    / "m5_hotwater_label_factorial"
)

QUERY_NPZ_SHA = "d780f0f8a96c47f49ffe061a72906728f1301056555350cabd979348aa41a2a0"
QUERY_RAW_SHA = "2fc4a638a2a0880f2b4d7feac87875c941d155f5fe5172b75b13d041b654fa16"
E4_ORDER_DIGEST = "63ca76f1167768252b29992fd791c450ba33447f5908b8938f1b67d0ecc732e3"


# --------------------------------------------------------------------------
# the no-fit guard
# --------------------------------------------------------------------------


def test_guard_blocks_every_fit_entry_point():
    """Calling fit after arming must raise, not warn and not no-op."""
    import m5_e5_guard

    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:  # noqa: BLE001 - environment dependent
        pytest.skip(f"tabpfn not importable here: {exc}")

    original = {name: getattr(TabPFNClassifier, name, None) for name in ("fit",)}
    try:
        blocked = m5_e5_guard.arm()
        assert "TabPFNClassifier.fit" in blocked
        m5_e5_guard.assert_armed()
        model = TabPFNClassifier.__new__(TabPFNClassifier)
        try:
            model.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
        except m5_e5_guard.FitAttemptedError as err:
            assert "pure re-scoring" in str(err)
        else:
            raise AssertionError("fit() was allowed after the guard was armed")
    finally:
        for name, fn in original.items():
            if fn is not None:
                setattr(TabPFNClassifier, name, fn)
        m5_e5_guard._ARMED = False


def test_guard_must_be_armed_before_scoring():
    import m5_e5_guard

    was = m5_e5_guard._ARMED
    m5_e5_guard._ARMED = False
    try:
        m5_e5_guard.assert_armed()
    except m5_e5_guard.FitAttemptedError:
        pass
    else:
        raise AssertionError("assert_armed passed while disarmed")
    finally:
        m5_e5_guard._ARMED = was


def test_guard_blocks_the_tree_estimators_too():
    """The fixed tree comparator is reloaded, never refit."""
    import m5_e5_guard
    from sklearn.ensemble import RandomForestClassifier

    original = RandomForestClassifier.fit
    try:
        blocked = m5_e5_guard.arm()
        assert "RandomForestClassifier.fit" in blocked
        try:
            RandomForestClassifier().fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
        except m5_e5_guard.FitAttemptedError:
            pass
        else:
            raise AssertionError("tree fit was allowed")
    finally:
        RandomForestClassifier.fit = original
        m5_e5_guard._ARMED = False


# --------------------------------------------------------------------------
# clustered namespace separation
# --------------------------------------------------------------------------


def test_e5_namespace_gives_different_draws_from_e4():
    """E5 must not reuse E4's draw sequence, and must be reproducible itself."""
    from m5_e4_clustered import cluster_members, draw_generator, resample_rows

    labels = np.array([f"b{i % 7}" for i in range(40)], dtype=object)
    names, members = cluster_members(labels)
    e4 = resample_rows(draw_generator("building", 3, 4004), names, members)
    e5 = resample_rows(draw_generator("building", 3, 5005), names, members)
    again = resample_rows(draw_generator("building", 3, 5005), names, members)
    assert not np.array_equal(e4, e5)
    assert np.array_equal(e5, again)


def test_default_namespace_still_reproduces_e4():
    """Adding the parameter must not have moved E4's stream."""
    from m5_e4_clustered import CLUSTER_CODE, MASTER_SEED, draw_generator

    expected = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([MASTER_SEED, 4004, CLUSTER_CODE["segment"], 314])
        )
    )
    got = draw_generator("segment", 314)
    assert np.array_equal(expected.integers(0, 2**31, 20), got.integers(0, 2**31, 20))


# --------------------------------------------------------------------------
# the 192-row query
# --------------------------------------------------------------------------


def test_query_identity_and_zero_overlap():
    import hashlib

    q = FACTORIAL / "independent_query" / "queries.npz"
    if not q.exists():
        pytest.skip("192-row query not present in this environment")
    assert hashlib.sha256(q.read_bytes()).hexdigest() == QUERY_NPZ_SHA
    with np.load(q, allow_pickle=True) as z:
        raw = np.asarray(z["raw_index"], dtype="int64")
        stratum = np.asarray(z["stratum"])
    assert hashlib.sha256(raw.tobytes()).hexdigest() == QUERY_RAW_SHA
    assert raw.size == 192 and np.unique(raw).size == 192
    counts = {s: int((stratum == s).sum()) for s in set(stratum.tolist())}
    assert counts == {"steam_positive": 64, "hw01_negative": 64, "hw01_positive": 64}

    screening = (
        Path(r"C:\Users\tonykuo\projects\lead-reproduction")
        / "data/processed/m5_context_stories/queries/screening/queries.npz"
    )
    with np.load(screening) as z:
        raw352 = np.asarray(z["raw_index"], dtype="int64")
    assert np.intersect1d(raw, raw352).size == 0


def test_endpoint_selectors_pick_the_intended_strata():
    """The E4 endpoint code selects by meter and anomaly, not by stratum name.

    On this query those must coincide exactly, or the co-primary endpoints would
    quietly mean something different than they did in E4.
    """
    q = FACTORIAL / "independent_query" / "queries.npz"
    if not q.exists():
        pytest.skip("192-row query not present in this environment")
    with np.load(q, allow_pickle=True) as z:
        meter = np.asarray(z["meter"], dtype="int8")
        anom = np.asarray(z["anomaly"], dtype="int8")
        stratum = np.asarray(z["stratum"])
    assert np.array_equal((meter == 2) & (anom == 1), stratum == "steam_positive")
    assert np.array_equal((meter == 3) & (anom == 0), stratum == "hw01_negative")
    assert int((meter == 1).sum()) == 0  # no chilledwater


# --------------------------------------------------------------------------
# the pre-declared decision rules
# --------------------------------------------------------------------------


def _decide(per_seed_auc, per_seed_margin, ci_excl=True):
    """Drive m5_e5_decision.evaluate with a synthetic result set."""
    from m5_e5_decision import evaluate

    eps = [
        "steam_positive_vs_hotwater_negative_pairwise_auc",
        "steam_positive_minus_hotwater_negative_score_margin",
    ]
    data = {}
    for ep, vals in zip(eps, (per_seed_auc, per_seed_margin)):
        arms = {}
        for arm in ("cell_specific", "frozen_reference"):
            mean = float(np.mean(list(vals.values())))
            arms[arm] = {
                "overall": mean,
                "per_seed": {k: float(v) for k, v in vals.items()},
                "seeds_positive": sum(1 for v in vals.values() if v > 0),
                "range": 0.0,
                "sample_sd": 0.0,
                "sign_consistency": "",
                "clustered": {
                    c: {
                        "q025": 0.01 if ci_excl else -0.01,
                        "q975": 0.5,
                        "excludes_zero": ci_excl,
                        "excludes_zero_positive": ci_excl,
                    }
                    for c in ("building", "segment")
                },
            }
        data[ep] = arms
    return evaluate(data, eps)["verdict"]


def test_replicated_requires_everything():
    pos = {"42": 0.4, "123": 0.3, "999": 0.5}
    assert _decide(pos, pos, ci_excl=True) == "REPLICATED"


def test_intervals_touching_zero_downgrade_to_inconclusive():
    pos = {"42": 0.4, "123": 0.3, "999": 0.5}
    assert (
        _decide(pos, pos, ci_excl=False) == "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE"
    )


def test_one_negative_seed_downgrades_but_does_not_fail():
    mixed = {"42": 0.4, "123": -0.1, "999": 0.5}
    assert (
        _decide(mixed, mixed, ci_excl=False)
        == "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE"
    )


def test_two_negative_seeds_is_not_replicated():
    bad = {"42": -0.4, "123": -0.1, "999": 0.5}
    assert _decide(bad, bad, ci_excl=True) == "NOT_REPLICATED"


def test_a_negative_overall_on_either_endpoint_is_not_replicated():
    pos = {"42": 0.4, "123": 0.3, "999": 0.5}
    neg = {"42": -0.4, "123": -0.3, "999": -0.5}
    assert _decide(pos, neg, ci_excl=True) == "NOT_REPLICATED"
    assert _decide(neg, pos, ci_excl=True) == "NOT_REPLICATED"


def test_an_endpoint_may_not_be_dropped_to_reach_a_verdict():
    """Both co-primary endpoints must be present in the rules."""
    from m5_e5_decision import EFFECT

    assert EFFECT == "negative_support_main_effect"


# --------------------------------------------------------------------------
# the frozen protocol artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def protocol() -> dict:
    p = CANONICAL / "e5_protocol.json"
    if not p.exists():
        pytest.skip("E5 protocol not frozen in this environment")
    return json.loads(p.read_text(encoding="utf-8"))["protocol"]


def test_protocol_forbids_fitting(protocol):
    pr = protocol["pure_rescoring"]
    assert pr["refit_prohibited"] is True
    for call in ("model.fit", "tree refit", "context resampling"):
        assert call in pr["prohibited_calls"]
    assert "load_fitted_tabpfn_model" in pr["permitted_pipeline"]
    assert protocol["completion_census"]["fits_performed"] == 0


def test_protocol_freezes_the_decision_vocabulary(protocol):
    r = protocol["decision_rules"]
    assert r["vocabulary"] == [
        "REPLICATED",
        "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE",
        "NOT_REPLICATED",
        "EXECUTION_INCOMPLETE",
    ]
    assert r["thresholds_may_not_change_after_seeing_results"] is True
    assert r["interval_containing_zero_is_not_proof_of_absence"] is True
    assert r["primary_replication_target"].startswith("hotwater-negative")


def test_protocol_inherits_e4_order_and_namespace(protocol):
    assert protocol["execution_order"]["realised_order_digest"] == E4_ORDER_DIGEST
    assert protocol["execution_order"]["reshuffling"] == "forbidden"
    assert len(protocol["execution_order"]["units"]) == 24
    c = protocol["clustered_uncertainty"]
    assert (c["master_seed"], c["namespace_code"]) == (20260730, 5005)
    assert c["cluster_code"] == {"building": 1, "segment": 2}
    assert c["draws"] == 1000


def test_protocol_records_scaler_verification_as_mandatory(protocol):
    s = protocol["scaler_policy"]
    assert s["arm_is_inherited_never_reselected"] is True
    assert "no scoring without it" in s["verification_is_mandatory"]
    assert s["cell_11_arms_still_scored_separately"] is True


def test_state_manifest_covers_24_states_in_e4_order():
    p = CANONICAL / "e5_state_manifest.json"
    if not p.exists():
        pytest.skip("E5 state manifest not frozen in this environment")
    import hashlib

    states = json.loads(p.read_text(encoding="utf-8"))["states"]
    assert len(states) == 24
    assert len({s["unit_id"] for s in states}) == 24
    assert len({s["state_sha256"] for s in states}) == 24
    order = hashlib.sha256(
        "\n".join(s["unit_id"] for s in states).encode("utf-8")
    ).hexdigest()
    assert order == E4_ORDER_DIGEST

    reps = json.loads(
        (CANONICAL / "e5_repeat_manifest.json").read_text(encoding="utf-8")
    )["repeats"]
    assert len(reps) == 192
    assert all(r["expected_score_length"] == 192 for r in reps)
    from collections import Counter

    assert set(Counter(r["unit_id"] for r in reps).values()) == {8}


# --------------------------------------------------------------------------
# the fixed-tree execution override (human ruling of 2026-08-02)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree_override() -> dict:
    p = CANONICAL / "e5_tree_execution_override.json"
    if not p.exists():
        pytest.skip("tree execution override not frozen in this environment")
    return json.loads(p.read_text(encoding="utf-8"))


def test_override_supplements_rather_than_rewrites_the_protocol(tree_override):
    assert tree_override["supplements"] == "e5_protocol.json"
    assert tree_override["protocol_history_rewritten"] is False
    assert tree_override["original_rule"].startswith("all E5 scientific scoring")


def test_override_records_the_evidence_on_both_hosts(tree_override):
    """The ruling must carry what broke the rule, not just the conclusion."""
    g = tree_override["gpu_host_reproduction"]
    laptop = tree_override["laptop_bit_exact_evidence"]
    assert g["bit_exact_rows"] == 0
    assert g["max_abs_difference"] > 0
    assert laptop["max_abs_difference"] == 0.0
    assert tree_override["observed_hard_failure"]["first_unit"]
    assert len(tree_override["observed_hard_failure"]["ruled_out_by_diagnosis"]) >= 4


def test_override_changes_only_where_trees_run(tree_override):
    assert tree_override["human_decision"] == "OPTION_A"
    assert tree_override["tabpfn_execution_host"] == "gpu-host"
    assert tree_override["tree_execution_host"] == "original laptop environment"
    assert tree_override["no_refit"] is True
    assert tree_override["gpu_host_tree_outputs_prohibited"] is True
    for unchanged in (
        "scientific_design_unchanged",
        "decision_rules_unchanged",
        "endpoints_unchanged",
        "states_unchanged",
        "clustered_estimator_unchanged",
        "tabpfn_specific_threshold_not_lowered",
    ):
        assert tree_override[unchanged] is True


def test_override_requires_bit_exact_identity_with_no_sampling(tree_override):
    assert tree_override["comparator_identity_requirement"] == (
        "bit_exact_on_e4_352_query"
    )
    gate = tree_override["comparator_gate"]
    assert gate["units_required"] == 24
    assert gate["max_abs_diff_required"] == 0.0
    assert gate["sampling_not_permitted"] is True
    assert gate["tolerance_not_permitted"] is True
    assert gate["on_failure"].startswith("stop")


def test_override_pins_one_shared_feature_matrix(tree_override):
    """Both hosts must score the same input artifact, not two equal-looking ones."""
    s = tree_override["shared_input_requirement"]
    assert s["shape"] == [192, 137]
    assert len(s["sha256"]) == 64
    assert s["laptop_may_not_build_its_own"] is True


def test_override_is_labelled_a_provenance_limitation(tree_override):
    assert "execution-provenance limitation" in tree_override["reporting_requirement"]
    assert "not be" in tree_override["reporting_requirement"]


def test_input_manifest_carries_the_override(tree_override):
    import hashlib

    p = CANONICAL / "e5_input_manifest.json"
    if not p.exists():
        pytest.skip("E5 input manifest not present")
    inputs = json.loads(p.read_text(encoding="utf-8"))
    actual = hashlib.sha256(
        (CANONICAL / "e5_tree_execution_override.json").read_bytes()
    ).hexdigest()
    assert inputs["tree_execution_override_sha256"] == actual
    assert inputs["tabpfn_execution_host"] == "gpu-host"
    assert inputs["tree_execution_host"] == "original laptop environment"
    assert (
        inputs["base_192_row_feature_sha256"]
        == tree_override["shared_input_requirement"]["sha256"]
    )


def _standalone() -> int:
    import inspect
    import traceback

    proto = None
    p = CANONICAL / "e5_protocol.json"
    if p.exists():
        proto = json.loads(p.read_text(encoding="utf-8"))["protocol"]
    passed = failed = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
            if "tree_override" in params:
                op = CANONICAL / "e5_tree_execution_override.json"
                if not op.exists():
                    print(f"SKIP {name}: override not frozen")
                    skipped += 1
                    continue
                fn(json.loads(op.read_text(encoding="utf-8")))
            elif "protocol" in params:
                if proto is None:
                    print(f"SKIP {name}: protocol not frozen")
                    skipped += 1
                    continue
                fn(proto)
            else:
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
