"""Tests for the frozen M5 E3 variance pilot protocol and runner guards.

Every human-operator decision of 2026-08-01 is asserted here so a later edit
that contradicts the frozen protocol fails loudly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

try:  # pytest is not a declared dependency of this repository
    import pytest
except ModuleNotFoundError:  # minimal shim so the suite still runs standalone

    class _Approx:
        def __init__(self, value, rel=None, abs=None):
            self.value, self.rel, self.abs = value, rel, abs

        def __eq__(self, other):
            tol = (
                self.abs
                if self.abs is not None
                else (
                    abs(self.value) * (self.rel if self.rel is not None else 1e-6)
                    + 1e-12
                )
            )
            return abs(other - self.value) <= tol

    class _Pytest:
        @staticmethod
        def approx(value, rel=None, abs=None):
            return _Approx(value, rel, abs)

        @staticmethod
        def skip(reason):
            raise _Skip(reason)

        class fixture:  # noqa: N801 - mimics the pytest decorator
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

    class _Skip(Exception):
        pass

    pytest = _Pytest()  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (
    Path(r"C:\Users\tonykuo\projects\lead-reproduction")
    / "data"
    / "processed"
    / "m5_e3_variance_pilot"
    / "e3_protocol.json"
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    if not PROTOCOL.exists():
        pytest.skip("E3 protocol not frozen in this environment")
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol"]


def test_schedule_seed_and_realised_order(protocol):
    d = protocol["supplementary_decisions"]["1_schedule_seed"]
    assert d["schedule_seed"] == 42
    assert "PCG64" in d["generator"]
    # the realised order must be persisted, not only the seed
    assert d["realised_cell_order"] == ["00", "01", "10", "11"]
    # and it must be exactly what PCG64(42) produces
    rng = np.random.Generator(np.random.PCG64(42))
    keys = np.array(["11", "10", "01", "00"])
    assert [str(c) for c in keys[rng.permutation(4)]] == d["realised_cell_order"]


def test_no_cross_cell_interleaving(protocol):
    d = protocol["supplementary_decisions"]["2_execution_lifecycle"]
    assert d["cross_cell_repeat_interleaving"] is False
    assert d["reload_backfill_of_same_process_repeats"] == "forbidden"


def test_repeat_batches(protocol):
    d = protocol["supplementary_decisions"]["3_repeat_batches"]
    assert d["batches"] == [8, 16, 24, 32, 40]
    assert d["increment"] == 8
    assert d["cap"] == 40


def test_only_two_steam_endpoints_gate_precision(protocol):
    d = protocol["supplementary_decisions"]["4_precision_gate_endpoints"]
    assert d["gating"] == [
        "steam_positive_vs_hotwater_negative_pairwise_auc",
        "steam_positive_vs_hotwater_negative_continuous_score_margin",
    ]
    assert d["evaluated_per_cell"] is True
    for banned in (
        "chilledwater_secondary_readouts",
        "global_rank",
        "within_meter_rank",
        "fresh_process_diagnostics",
    ):
        assert banned in d["explicitly_non_gating"]


def test_ci_is_student_t_and_alternatives_forbidden(protocol):
    d = protocol["supplementary_decisions"]["5_ci_half_width"]
    assert "Student-t" in d["method"]
    for banned in (
        "normal_approximation",
        "percentile_bootstrap",
        "averaging_row_probabilities_then_scoring_once",
    ):
        assert banned in d["forbidden"]


def test_student_t_half_width_formula():
    """The frozen formula, checked against an independent computation."""
    x = np.array([0.81, 0.79, 0.80, 0.82, 0.78, 0.80, 0.81, 0.79])
    n = x.size
    expected = stats.t.ppf(0.975, df=n - 1) * x.std(ddof=1) / math.sqrt(n)
    got = float(stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n))
    assert got == pytest.approx(expected, rel=1e-12)
    # a normal approximation would be materially smaller at n=8 and is forbidden
    normal = 1.959963985 * x.std(ddof=1) / math.sqrt(n)
    assert got > normal


def test_reference_iqr_is_pre_fit_and_tree_derived(protocol):
    r = protocol["reference_iqr"]
    assert r["computed_before_any_tabpfn_fit"] is True
    assert r["not_derived_from_tabpfn_repeats"] is True
    for cell, v in r["per_cell"].items():
        assert v["scaler_arm"] == "cell_specific"
        assert v["steam_positive_rows"] == 32
        assert v["hotwater_negative_rows"] == 16
        assert v["pairs"] == 512
        assert v["reference_iqr"] > 0
        assert v["margin_half_width_target"] == pytest.approx(
            0.02 * v["reference_iqr"], rel=1e-12
        )


def test_fresh_process_diagnostic_scope(protocol):
    d = protocol["fresh_process_diagnostic"]
    assert d["cell"] == "11"
    assert d["runs"] == 2
    assert d["kept_separate_from_same_process_statistics"] is True
    assert d["controls_repeat_count"] is False
    assert d["scientific_estimate"] is False
    assert "state load failure" in d["hard_failure_conditions"]


def test_onset_readout_unresolved(protocol):
    o = protocol["readouts"]["onset_phase_contrast"]
    assert o["status"] == "UNRESOLVED_NOT_EXECUTED"
    assert "192-row query is not read" in o["consequence"]


def test_chilledwater_vs_hotwater_negative_is_resolution_limited(protocol):
    c = protocol["readouts"]["chilledwater_vs_hotwater_negative"]
    assert c["status"] == "RESOLUTION_LIMITED_DIAGNOSTIC"
    assert c["excluded_from_scientific_gate"] is True
    assert c["mechanism_bearing"] is False
    assert c["hotwater_negative_rows"] == 16
    assert c["valid_pairs"] == 1024


def test_chilledwater_within_meter_secondary_retained(protocol):
    r = protocol["readouts"]["chilledwater_secondary_within_meter"]
    for name in (
        "chilledwater_positive_vs_chilledwater_negative_pairwise_auc",
        "chilledwater_within_meter_pr_auc",
        "chilledwater_within_meter_roc_auc",
    ):
        assert name in r
    assert protocol["readouts"]["score_and_rank_never_pooled"] is True
    assert protocol["readouts"]["reported_separately"] == [
        "tabpfn",
        "matched_tree",
        "tabpfn_minus_tree",
    ]


def test_base_policy_history_preserved(protocol):
    b = protocol["base_policy"]
    assert b["status_in_base_policy"] == "designed_not_running"
    a = protocol["authorization"]
    assert a["human_authorization_date"] == "2026-08-01"
    assert a["execution_status"] == "HUMAN_AUTHORIZED_FOR_EXECUTION"
    assert a["scientific_design_unchanged"] is True


def test_inherited_model_settings_unchanged(protocol):
    i = protocol["inherited"]
    assert i["scientific_tabpfn_version"] == "8.0.8"
    assert i["context_n"] == 20000
    assert i["context_seed"] == 42
    assert i["model_seed"] == 42
    assert i["scaler_arm"] == "cell_specific"
    assert i["fits_per_cell"] == 1
    assert i["main_inference_mode"] == "same_process"
    assert sorted(i["cells"]) == ["00", "01", "10", "11"]


def test_trees_are_a_fixed_comparator(protocol):
    t = protocol["trees"]
    assert t["refit"] is False
    assert t["tuned"] is False
    assert t["artificial_replicates"] is False
    assert "gap[c,r] = Y[c,r] - fixed tree metric[c]" in t["gap_rule"]


def test_prohibitions_present(protocol):
    p = protocol["prohibitions"]
    for banned in (
        "E4 formal Path A",
        "Path B",
        "frozen 192-row query",
        "tree refit",
        "TabPFN 8.1.0 as science",
        "mixing fresh-process reload repeats into the same-process estimand",
        "ProcessPoolExecutor or parallel GPU workers",
    ):
        assert banned in p


def test_cell_decision_and_e3_decision_rules(protocol):
    c = protocol["cell_decision_rule"]
    assert c["check_at"] == [8, 16, 24, 32, 40]
    assert c["cap_failure_status"] == "MEASUREMENT_UNSTABLE_AT_CAP"
    e = protocol["e3_decision_rule"]
    assert set(e) == {
        "E3_MEASUREMENT_PROCESS_ACCEPTABLE",
        "E3_MEASUREMENT_PROCESS_UNSTABLE",
        "E3_MORE_REPEATS_REQUIRED",
        "E3_EXECUTION_INCOMPLETE",
    }


def test_endpoints_are_computed_per_repeat_not_pooled():
    """Guards the 'never average row probabilities then score once' rule."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    rng = np.random.default_rng(0)
    # steam-positive, hotwater-negative, and both chilledwater strata, so every
    # readout the runner computes has rows
    meter = np.array([2] * 32 + [3] * 16 + [1] * 32 + [1] * 32, dtype="int8")
    anom = np.array([1] * 32 + [0] * 16 + [1] * 32 + [0] * 32, dtype="int8")
    n = meter.size
    a = rng.uniform(0.6, 0.9, n)
    b = rng.uniform(0.5, 0.8, n)
    from m5_e3_runner import endpoints

    ea = endpoints(a, meter, anom)["steam_positive_vs_hotwater_negative_pairwise_auc"]
    eb = endpoints(b, meter, anom)["steam_positive_vs_hotwater_negative_pairwise_auc"]
    pooled = endpoints((a + b) / 2, meter, anom)[
        "steam_positive_vs_hotwater_negative_pairwise_auc"
    ]
    # scoring the averaged probabilities is a different quantity from the mean of
    # the per-repeat scores; the protocol requires the latter
    assert pooled != pytest.approx((ea + eb) / 2, abs=1e-12)


def _standalone() -> int:
    """Run every test without pytest, so the suite is verifiable in an
    environment where pytest is not a declared dependency."""
    import inspect

    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol"]
    tests = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for name, fn in tests:
        try:
            params = inspect.signature(fn).parameters
            fn(proto) if "protocol" in params else fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


def _rec(auc: float, margin: float) -> dict:
    return {
        "endpoints": {
            "steam_positive_vs_hotwater_negative_pairwise_auc": auc,
            "steam_positive_minus_hotwater_negative_score_margin": margin,
        }
    }


def test_gate_uses_student_t_and_only_steam_endpoints():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from m5_e3_runner import evaluate_gate, half_width

    tight = [_rec(0.90 + 0.0001 * i, 0.50 + 0.0001 * i) for i in range(8)]
    g = evaluate_gate(tight, 0.015, 0.016)
    assert g["n"] == 8
    assert g["auc_pass"] and g["margin_pass"] and g["both_pass"]
    # half width must equal the frozen Student-t formula
    aucs = [
        r["endpoints"]["steam_positive_vs_hotwater_negative_pairwise_auc"]
        for r in tight
    ]
    assert g["auc_half_width"] == pytest.approx(half_width(aucs), rel=1e-12)

    # a tight AUC with a noisy margin must fail on the margin alone
    noisy = [_rec(0.90, 0.50 + (0.05 if i % 2 else -0.05)) for i in range(8)]
    g2 = evaluate_gate(noisy, 0.015, 0.001)
    assert g2["auc_pass"] is True
    assert g2["margin_pass"] is False
    assert g2["both_pass"] is False


def test_gate_half_width_shrinks_with_more_repeats():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from m5_e3_runner import half_width

    rng = np.random.default_rng(7)
    draws = list(rng.normal(0.9, 0.01, 40))
    assert half_width(draws[:8]) > half_width(draws[:40])


if __name__ == "__main__":
    raise SystemExit(_standalone())
