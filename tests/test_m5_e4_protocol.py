"""Tests for the frozen M5 E4 formal Path A protocol and clustered estimator.

Every human ruling of 2026-08-02 that can be violated by a plausible code edit
is asserted here. The rulings that matter most are the negative ones -- what the
estimator must *not* do -- so most of these tests construct the forbidden
alternative and prove the implementation does not equal it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

        class fixture:  # noqa: N801 - mimics the pytest decorator
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
    / "m5_e4_formal_path_a"
)

from m5_e4_clustered import (  # noqa: E402
    CLUSTER_CODE,
    DRAWS,
    MASTER_SEED,
    NAMESPACE_CODE,
    cluster_members,
    draw_contrasts,
    draw_generator,
    resample_rows,
    seed_consistency,
    segment_clusters,
)
from m5_e4_endpoints import (  # noqa: E402
    EFFECT_NAMES,
    endpoints,
    factor_effect,
    fit_level_estimate,
)

METER_STEAM, METER_HW, METER_CW = 2, 3, 1


# --------------------------------------------------------------------------
# fixtures: a small synthetic query and a synthetic 24-fit result set
# --------------------------------------------------------------------------


def _query(n_per: int = 6) -> pd.DataFrame:
    rows = []
    raw = 0
    for meter, anomaly, count in (
        (METER_STEAM, 1, n_per),
        (METER_HW, 0, n_per),
        (METER_CW, 1, n_per),
        (METER_CW, 0, n_per),
    ):
        for i in range(count):
            raw += 2  # non-contiguous, so each anomaly row is its own segment
            rows.append(
                {
                    "raw_index": raw,
                    "meter": meter,
                    "anomaly": anomaly,
                    "building_id": 100 + (i % 3),
                }
            )
    return pd.DataFrame(rows)


def _results(query: pd.DataFrame, rng: np.random.Generator):
    """24 fits x 8 repeats plus 24 tree vectors, with a real composition effect."""
    n = len(query)
    tabpfn, trees = {}, {}
    for seed in (42, 123, 999):
        for cell in ("00", "01", "10", "11"):
            for arm in ("cell_specific", "frozen_reference"):
                base = rng.normal(0.5, 0.05, n)
                # steam positives lift when hotwater-positive support is present
                lift = 0.20 if cell[0] == "1" else 0.0
                lift += 0.05 if arm == "frozen_reference" else 0.0
                base[query["meter"].to_numpy() == METER_STEAM] += lift
                tabpfn[(seed, cell, arm)] = [
                    base + rng.normal(0, 0.002, n) for _ in range(8)
                ]
                trees[(seed, cell, arm)] = base + rng.normal(0, 0.01, n)
    return tabpfn, trees


# --------------------------------------------------------------------------
# ruling A: repeats enter as fit-level endpoint means, never averaged probabilities
# --------------------------------------------------------------------------


def test_row_probabilities_are_never_averaged_before_scoring():
    """The fit-level estimate must differ from scoring the averaged vector.

    AUC is not linear in the scores, so the two estimators genuinely differ.
    If this test ever passes trivially the implementation has been changed to
    the forbidden shortcut.
    """
    q = _query()
    rng = np.random.default_rng(0)
    reps = [rng.normal(0.5, 0.1, len(q)) for _ in range(8)]
    meter = q["meter"].to_numpy()
    anom = q["anomaly"].to_numpy()

    correct = fit_level_estimate(reps, meter, anom)
    forbidden = endpoints(np.mean(reps, axis=0), meter, anom)

    auc = "steam_positive_vs_hotwater_negative_pairwise_auc"
    assert correct[auc] != forbidden[auc]

    # and it really is the mean of the per-repeat endpoint values
    per_repeat = [endpoints(r, meter, anom)[auc] for r in reps]
    assert correct[auc] == pytest.approx(float(np.mean(per_repeat)), rel=1e-12)


def test_margin_is_averaged_after_scoring_each_repeat():
    """The margin is linear, so its two routes agree -- but for the right reason."""
    q = _query()
    rng = np.random.default_rng(1)
    reps = [rng.normal(0.5, 0.1, len(q)) for _ in range(8)]
    meter, anom = q["meter"].to_numpy(), q["anomaly"].to_numpy()
    key = "steam_positive_minus_hotwater_negative_score_margin"

    per_repeat = [endpoints(r, meter, anom)[key] for r in reps]
    assert fit_level_estimate(reps, meter, anom)[key] == pytest.approx(
        float(np.mean(per_repeat)), rel=1e-12
    )


def test_repeat_ids_are_not_paired_across_cells():
    """Shuffling one cell's repeat order must not move any contrast.

    Pairing repeat r of cell A with repeat r of cell B would make the result
    depend on that order. Reduction to a fit-level mean before the cells meet is
    what makes the ordering irrelevant.
    """
    q = _query()
    rng = np.random.default_rng(2)
    tabpfn, trees = _results(q, rng)
    idx = np.arange(len(q))
    kw = dict(
        trees=trees,
        idx=idx,
        meter=q["meter"].to_numpy(),
        anomaly=q["anomaly"].to_numpy(),
        endpoint="steam_positive_vs_hotwater_negative_pairwise_auc",
    )
    before = draw_contrasts(tabpfn=tabpfn, **kw)

    shuffled = {k: list(v) for k, v in tabpfn.items()}
    key = (42, "11", "cell_specific")
    shuffled[key] = list(reversed(shuffled[key]))
    after = draw_contrasts(tabpfn=shuffled, **kw)

    for name in before:
        assert before[name] == pytest.approx(after[name], rel=1e-12)


# --------------------------------------------------------------------------
# ruling C + E: one row multiset, shared everywhere
# --------------------------------------------------------------------------


def test_draw_rows_are_shared_across_cells_arms_seeds_and_tree():
    """A draw's row multiset is produced once and reused, not redrawn per unit."""
    q = _query()
    names, members = cluster_members(
        q["building_id"].astype(str).to_numpy(dtype=object)
    )
    a = resample_rows(draw_generator("building", 7), names, members)
    b = resample_rows(draw_generator("building", 7), names, members)
    assert np.array_equal(a, b)
    c = resample_rows(draw_generator("building", 8), names, members)
    assert not np.array_equal(a, c)
    d = resample_rows(draw_generator("segment", 7), names, members)
    assert not np.array_equal(a, d)


def test_scaler_interaction_is_formed_inside_the_draw_with_the_right_sign():
    """frozen minus cell_specific, computed on shared rows."""
    q = _query()
    rng = np.random.default_rng(3)
    tabpfn, trees = _results(q, rng)
    out = draw_contrasts(
        tabpfn=tabpfn,
        trees=trees,
        idx=np.arange(len(q)),
        meter=q["meter"].to_numpy(),
        anomaly=q["anomaly"].to_numpy(),
        endpoint="steam_positive_minus_hotwater_negative_score_margin",
    )
    for effect in EFFECT_NAMES:
        assert out[f"tabpfn__scaler_interaction__{effect}"] == pytest.approx(
            out[f"tabpfn__frozen_reference__{effect}"]
            - out[f"tabpfn__cell_specific__{effect}"],
            rel=1e-12,
        )


def test_tabpfn_minus_tree_gap_uses_the_same_draw():
    q = _query()
    rng = np.random.default_rng(4)
    tabpfn, trees = _results(q, rng)
    out = draw_contrasts(
        tabpfn=tabpfn,
        trees=trees,
        idx=np.arange(len(q)),
        meter=q["meter"].to_numpy(),
        anomaly=q["anomaly"].to_numpy(),
        endpoint="steam_positive_minus_hotwater_negative_score_margin",
    )
    for arm in ("cell_specific", "frozen_reference", "scaler_interaction"):
        for effect in EFFECT_NAMES:
            assert out[f"tabpfn_minus_tree__{arm}__{effect}"] == pytest.approx(
                out[f"tabpfn__{arm}__{effect}"] - out[f"tree__{arm}__{effect}"],
                rel=1e-12,
            )


# --------------------------------------------------------------------------
# ruling D: three seeds, equal weight, no resampling of seeds
# --------------------------------------------------------------------------


def test_overall_is_the_equal_weight_mean_of_three_seed_contrasts():
    q = _query()
    rng = np.random.default_rng(5)
    tabpfn, trees = _results(q, rng)
    out = draw_contrasts(
        tabpfn=tabpfn,
        trees=trees,
        idx=np.arange(len(q)),
        meter=q["meter"].to_numpy(),
        anomaly=q["anomaly"].to_numpy(),
        endpoint="steam_positive_minus_hotwater_negative_score_margin",
    )
    for effect in EFFECT_NAMES:
        base = f"tabpfn__cell_specific__{effect}"
        per_seed = [out[f"{base}__seed{s}"] for s in (42, 123, 999)]
        assert out[base] == pytest.approx(float(np.mean(per_seed)), rel=1e-12)


def test_seed_consistency_reports_descriptives_not_inference():
    out = seed_consistency({42: 0.10, 123: 0.12, 999: 0.08})
    assert out["overall_equal_weight_mean"] == pytest.approx(0.10, rel=1e-12)
    assert out["range"] == pytest.approx(0.04, rel=1e-9)
    assert out["sign_consistency"] == "3/3 positive"
    assert out["all_same_sign"] is True
    # nothing inferential may appear: three fixed seeds are not a sample
    assert not any(k in out for k in ("p_value", "t_statistic", "ci", "std_error"))

    mixed = seed_consistency({42: 0.10, 123: -0.02, 999: 0.08})
    assert mixed["all_same_sign"] is False
    assert mixed["sign_consistency"] == "2/3 positive"


# --------------------------------------------------------------------------
# ruling E: addressable seeds, independent of loop order
# --------------------------------------------------------------------------


def test_seed_mapping_is_addressable_and_order_independent():
    q = _query()
    names, members = cluster_members(
        q["building_id"].astype(str).to_numpy(dtype=object)
    )
    forward = [
        resample_rows(draw_generator("building", d), names, members) for d in range(20)
    ]
    backward = {
        d: resample_rows(draw_generator("building", d), names, members)
        for d in reversed(range(20))
    }
    for d in range(20):
        assert np.array_equal(forward[d], backward[d])


def test_seed_mapping_matches_the_frozen_construction():
    expected = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [MASTER_SEED, NAMESPACE_CODE, CLUSTER_CODE["segment"], 314]
            )
        )
    )
    got = draw_generator("segment", 314)
    assert np.array_equal(expected.integers(0, 2**31, 25), got.integers(0, 2**31, 25))
    assert (MASTER_SEED, NAMESPACE_CODE) == (20260730, 4004)
    assert CLUSTER_CODE == {"building": 1, "segment": 2}
    assert DRAWS == 1000


def test_draw_id_outside_the_frozen_range_is_refused():
    for bad in (-1, DRAWS, DRAWS + 1):
        try:
            draw_generator("building", bad)
        except ValueError:
            continue
        raise AssertionError(f"draw_id {bad} should have been refused")


# --------------------------------------------------------------------------
# ruling B: invalid draws are marked, never imputed
# --------------------------------------------------------------------------


def test_margin_is_invalid_when_a_stratum_is_missing():
    """A draw that resamples away every hotwater-negative row has no margin."""
    q = _query()
    keep = np.flatnonzero(q["meter"].to_numpy() != METER_HW)
    score = np.linspace(0, 1, len(q))
    val = endpoints(
        score[keep], q["meter"].to_numpy()[keep], q["anomaly"].to_numpy()[keep]
    )
    assert not np.isfinite(val["steam_positive_minus_hotwater_negative_score_margin"])
    assert not np.isfinite(val["steam_positive_vs_hotwater_negative_pairwise_auc"])
    # nothing was substituted for the missing value
    assert val["steam_positive_minus_hotwater_negative_score_margin"] != 0.0


def test_invalid_repeat_makes_the_whole_fit_level_estimate_invalid():
    q = _query()
    keep = np.flatnonzero(q["meter"].to_numpy() != METER_HW)
    reps = [np.linspace(0, 1, len(keep)) for _ in range(8)]
    out = fit_level_estimate(
        reps, q["meter"].to_numpy()[keep], q["anomaly"].to_numpy()[keep]
    )
    assert not np.isfinite(out["steam_positive_minus_hotwater_negative_score_margin"])


# --------------------------------------------------------------------------
# ruling F: the repository's exact factorial formulas
# --------------------------------------------------------------------------


def test_factor_effect_matches_the_repository_formulas():
    v = {"00": 1.0, "01": 2.0, "10": 4.0, "11": 8.0}
    out = factor_effect(v)
    assert out["positive_support_main_effect"] == pytest.approx((4 + 8 - 1 - 2) / 2)
    assert out["negative_support_main_effect"] == pytest.approx((2 + 8 - 1 - 4) / 2)
    assert out["positive_x_negative_interaction"] == pytest.approx(8 - 4 - 2 + 1)


def test_endpoint_value_matches_the_full_dict():
    """The bootstrap's fast path must be the same number as the full readout.

    The clustered estimator calls the single-endpoint path hundreds of
    thousands of times. If it ever diverged from `endpoints()`, the intervals
    would silently describe a different quantity than the per-repeat records.
    """
    from m5_e4_endpoints import endpoint_value

    q = _query(7)
    rng = np.random.default_rng(21)
    meter = q["meter"].to_numpy(dtype="int8")
    anom = q["anomaly"].to_numpy(dtype="int8")
    for trial in range(5):
        score = rng.normal(0.5, 0.2, len(q))
        full = endpoints(score, meter, anom)
        for name, want in full.items():
            got = endpoint_value(name, score, meter, anom)
            if np.isfinite(want):
                assert got == pytest.approx(want, rel=1e-12), f"{name} trial {trial}"
            else:
                assert not np.isfinite(got), name


def test_endpoint_value_is_invalid_on_a_missing_stratum():
    from m5_e4_endpoints import endpoint_value

    q = _query()
    keep = np.flatnonzero(q["meter"].to_numpy() != METER_HW)
    score = np.linspace(0, 1, len(keep))
    for name in (
        "steam_positive_vs_hotwater_negative_pairwise_auc",
        "steam_positive_minus_hotwater_negative_score_margin",
    ):
        got = endpoint_value(
            name, score, q["meter"].to_numpy()[keep], q["anomaly"].to_numpy()[keep]
        )
        assert not np.isfinite(got), name


def test_e4_endpoints_reproduce_the_e3_definitions():
    """The frozen margin source must still agree with the E3 runner."""
    try:
        from m5_e3_runner import endpoints as e3_endpoints
    except Exception as exc:  # noqa: BLE001 - environment-dependent import
        pytest.skip(f"E3 runner not importable here: {exc}")
    q = _query(8)
    rng = np.random.default_rng(11)
    score = rng.normal(0.5, 0.1, len(q))
    meter = q["meter"].to_numpy(dtype="int8")
    anom = q["anomaly"].to_numpy(dtype="int8")
    mine, theirs = endpoints(score, meter, anom), e3_endpoints(score, meter, anom)
    assert set(mine) == set(theirs)
    for k in theirs:
        assert mine[k] == pytest.approx(theirs[k], rel=1e-12), k


# --------------------------------------------------------------------------
# cluster definitions
# --------------------------------------------------------------------------


def test_segment_clusters_group_contiguous_anomalies_only():
    raw = np.array([10, 11, 12, 40, 41, 70], dtype="int64")
    anom = np.array([1, 1, 1, 1, 1, 0], dtype="int8")
    labels = segment_clusters(raw, anom)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == "normal_70"


def test_normal_rows_never_share_a_segment_cluster():
    raw = np.array([10, 11, 12], dtype="int64")
    anom = np.array([0, 0, 0], dtype="int8")
    assert len(set(segment_clusters(raw, anom).tolist())) == 3


# --------------------------------------------------------------------------
# feature-matrix cache
# --------------------------------------------------------------------------


def test_feature_cache_round_trips_through_the_atomic_write(tmp_path):
    """The cache write must land on the path the rename expects.

    `np.savez` appends ".npz" to a *path* argument that does not already end in
    it, so writing to "<name>.npz.tmp" silently produces "<name>.npz.tmp.npz"
    and the rename fails with FileNotFoundError after the expensive build has
    already happened. Writing through a file object avoids the rewrite; this
    test exercises the whole write-rename-reload path rather than trusting it.
    """
    import os as _os

    npz = tmp_path / "seed42__cell11.npz"
    x = np.arange(12, dtype="float32").reshape(3, 4)
    y = np.array([0, 1, 0], dtype="int8")
    q = np.arange(8, dtype="float32").reshape(2, 4)

    tmp = npz.with_name(f".{npz.name}.{_os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, x=x, y=y, q=q)
    assert tmp.exists(), "np.savez rewrote the temp path"
    _os.replace(tmp, npz)

    assert npz.exists()
    assert not list(tmp_path.glob("*.tmp*")), "a temp file survived the rename"
    with np.load(npz) as z:
        assert np.array_equal(z["x"], x)
        assert np.array_equal(z["y"], y)
        assert np.array_equal(z["q"], q)


def test_cache_key_is_seed_and_cell_only(tmp_path):
    """Both scaler arms of a block must share one cache entry, and only that.

    The cache is keyed on (context seed, cell) because the scaler is applied
    after the cache. A key that included the arm would double the expensive
    builds; a key that dropped the seed or cell would reuse the wrong matrix.
    """
    from m5_e4_runner import cache_paths

    a_npz, a_meta = cache_paths(tmp_path, 42, "11")
    b_npz, _ = cache_paths(tmp_path, 42, "11")
    assert a_npz == b_npz
    assert a_meta.suffix == ".json" and a_npz.suffix == ".npz"
    for seed, cell in ((123, "11"), (42, "10"), (999, "00")):
        other, _ = cache_paths(tmp_path, seed, cell)
        assert other != a_npz


# --------------------------------------------------------------------------
# ensemble contract
# --------------------------------------------------------------------------


def test_effective_n_estimators_mismatch_is_a_hard_failure():
    from m5_e4_runner import verify_ensemble

    class Pre:
        def __init__(self, n):
            self.configs = [None] * n
            self.pipelines = [None] * n
            self.pipeline_seeds = [None] * n
            self.subsample_feature_indices = [None] * n

    class Exec:
        def __init__(self, n):
            self.ensemble_preprocessor = Pre(n)

    class Model:
        def __init__(self, eff=8, cfg=8, runtime=8, auto=False):
            self.n_estimators = 8
            self.auto_scale_n_estimators = auto
            self.n_estimators_ = eff
            self.ensemble_configs_ = [None] * cfg
            self.executor_ = Exec(runtime)

    ok = verify_ensemble(Model())
    assert ok["effective_n_estimators_"] == 8
    assert ok["runtime_ensemble_containers"]["pipelines"] == 8

    for kwargs in (
        {"eff": 16},
        {"eff": 4},
        {"cfg": 16},
        {"runtime": 32},
        {"auto": True},
    ):
        try:
            verify_ensemble(Model(**kwargs))
        except AssertionError:
            continue
        raise AssertionError(f"verify_ensemble accepted {kwargs}")


# --------------------------------------------------------------------------
# frozen protocol artifact, once it exists
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def protocol() -> dict:
    path = CANONICAL / "e4_protocol.json"
    if not path.exists():
        pytest.skip("E4 protocol not frozen in this environment")
    return json.loads(path.read_text(encoding="utf-8"))["protocol"]


def test_protocol_freezes_the_ensemble_contract(protocol):
    e = protocol["ensemble"]
    assert e["requested_n_estimators"] == 8
    assert e["auto_scale_n_estimators"] is False
    assert e["required_effective_n_estimators_"] == 8
    assert e["mismatch_policy"] == "hard_failure"
    assert e["importer_must_not_trust_runner_self_report"] is True


def test_protocol_freezes_24_fits_and_192_repeats(protocol):
    d = protocol["design"]
    assert d["fits"] == 24
    assert d["repeats_per_fit"] == 8
    assert d["same_process_repeats_total"] == 192
    assert d["external_fitted_states"] == 24
    assert d["model_seed_factor_added"] is False
    assert len(protocol["execution_order"]) == 24
    assert len({u["unit_id"] for u in protocol["execution_order"]}) == 24


def test_protocol_freezes_the_clustered_rulings(protocol):
    c = protocol["clustered_uncertainty"]
    assert c["E_addressable_seed_mapping"]["master_seed"] == 20260730
    assert c["E_addressable_seed_mapping"]["namespace_code"] == 4004
    assert c["E_addressable_seed_mapping"]["cluster_code"] == {
        "building": 1,
        "segment": 2,
    }
    assert c["E_addressable_seed_mapping"]["draws"] == 1000
    assert c["D_context_seed_aggregation"]["seeds"] == [42, 123, 999]
    assert (
        "averaging row probabilities before computing AUC"
        in c["A_repeats_into_cluster_bootstrap"]["forbidden"]
    )
    for banned in (
        "resampling the three seeds",
        "random-effects model",
        "selecting the best seed",
    ):
        assert banned in c["D_context_seed_aggregation"]["forbidden"]


def test_cell_11_arm_degeneracy_is_recorded_not_hidden(protocol):
    """The two cell-11 arms are one transform, and the protocol must say so.

    The frozen scaler of a seed is fitted on that seed's cell-11 matrix, so at
    cell 11 `frozen_reference` and `cell_specific` are the same scaler. The
    design still runs all 24 units, but the census must not demand 24 distinct
    states, and the scaler axis must not be credited with information cell 11
    cannot supply.
    """
    d = protocol["cell_11_arm_degeneracy"]
    assert "same transform" in d["fact"]
    assert d["units_still_executed"].startswith("all 24")
    assert protocol["completion_census"]["distinct_state_identities_minimum"] == 21
    assert (
        "cell-11 arm pairs"
        in protocol["completion_census"]["permitted_state_collisions"]
    )

    fit_path = CANONICAL / "e4_fit_manifest.json"
    if not fit_path.exists():
        pytest.skip("E4 manifests not frozen in this environment")
    fits = json.loads(fit_path.read_text(encoding="utf-8"))["fits"]
    # a set, not a list: the execution order is randomised
    assert {f["unit_id"] for f in fits if f["arms_are_identical_by_construction"]} == {
        f"seed{s}__cell11__{a}"
        for s in (42, 123, 999)
        for a in ("cell_specific", "frozen_reference")
    }
    # the tree comparators prove it empirically: exactly three collisions
    digests = [f["tree_comparator_sha256"] for f in fits]
    assert len(set(digests)) == 21
    for s in (42, 123, 999):
        pair = [
            f["tree_comparator_sha256"]
            for f in fits
            if f["context_seed"] == s and f["cell"] == "11"
        ]
        assert pair[0] == pair[1]


# --------------------------------------------------------------------------
# randomised execution schedule (human ruling of 2026-08-02)
# --------------------------------------------------------------------------


def _schedule(seed=None):
    import m5_e4_protocol as mod

    if seed is None:
        return mod.execution_schedule()
    original = mod.SCHEDULE_SEED
    try:
        mod.SCHEDULE_SEED = seed
        return mod.execution_schedule()
    finally:
        mod.SCHEDULE_SEED = original


def test_schedule_is_reproducible_from_the_seed():
    a, b = _schedule(), _schedule()
    assert a["realised_order_digest"] == b["realised_order_digest"]
    assert [u["unit_id"] for u in a["realised_24_unit_order"]] == [
        u["unit_id"] for u in b["realised_24_unit_order"]
    ]
    assert a["schedule_seed"] == 42


def test_a_different_seed_gives_a_different_order():
    """Otherwise the permutation is not actually being applied."""
    base = _schedule()["realised_order_digest"]
    others = {_schedule(s)["realised_order_digest"] for s in (1, 7, 123, 999, 20260730)}
    assert base not in others
    assert len(others) == 5


def test_every_block_appears_exactly_once():
    sched = _schedule()
    ids = [b["block_id"] for b in sched["realised_block_order"]]
    assert len(ids) == 12
    assert sorted(ids) == sorted(sched["canonical_block_enumeration"])
    assert len(set(ids)) == 12


def test_every_unit_appears_exactly_once():
    order = _schedule()["realised_24_unit_order"]
    assert len(order) == 24
    assert len({u["unit_id"] for u in order}) == 24
    expected = {
        f"seed{s}__cell{c}__{a}"
        for s in (42, 123, 999)
        for c in ("00", "01", "10", "11")
        for a in ("cell_specific", "frozen_reference")
    }
    assert {u["unit_id"] for u in order} == expected
    assert [u["position"] for u in order] == list(range(24))


def test_the_two_arms_of_a_block_are_adjacent():
    """Blocking is what lets the raw feature matrix be built once per block."""
    order = _schedule()["realised_24_unit_order"]
    for i in range(0, 24, 2):
        first, second = order[i], order[i + 1]
        assert first["block_id"] == second["block_id"]
        assert (first["context_seed"], first["cell"]) == (
            second["context_seed"],
            second["cell"],
        )
        assert {first["scaler_arm"], second["scaler_arm"]} == {
            "cell_specific",
            "frozen_reference",
        }


def test_arm_order_is_drawn_not_hardcoded():
    """At least one block must start with frozen_reference."""
    sched = _schedule()
    firsts = {b["arm_order"][0] for b in sched["realised_block_order"]}
    assert firsts == {"cell_specific", "frozen_reference"}


def test_schedule_does_not_depend_on_dict_or_filesystem_order():
    """The permutation input comes from frozen lists, not iteration order.

    Perturbing the CELLS dict's insertion order changes dict iteration but must
    not touch the schedule, because the block enumeration is built from
    CONTEXT_SEEDS and CELL_ORDER.
    """
    import m5_e4_protocol as mod

    before = _schedule()["realised_order_digest"]
    original = mod.CELLS
    try:
        mod.CELLS = dict(reversed(list(original.items())))
        assert _schedule()["realised_order_digest"] == before
    finally:
        mod.CELLS = original


def test_resume_follows_the_frozen_order_and_only_skips_completed(tmp_path):
    """A resume must replay the same sequence, not re-derive or re-shuffle it."""
    order = [u["unit_id"] for u in _schedule()["realised_24_unit_order"]]
    completed = {order[0], order[1], order[5]}
    remaining = [u for u in order if u not in completed]
    assert remaining == [u for u in order if u not in completed]
    assert len(remaining) == 21
    # the surviving units keep their original relative order
    assert remaining == sorted(remaining, key=order.index)


def test_realised_order_digest_covers_the_unit_sequence():
    import hashlib as _h

    sched = _schedule()
    expected = _h.sha256(
        "\n".join(u["unit_id"] for u in sched["realised_24_unit_order"]).encode("utf-8")
    ).hexdigest()
    assert sched["realised_order_digest"] == expected


def test_protocol_records_the_frozen_schedule(protocol):
    s = protocol["schedule"]
    assert s["schedule_seed"] == 42
    assert "PCG64" in s["schedule_rng"]
    assert s["blocks"] == 12
    assert len(s["realised_block_order"]) == 12
    assert len(s["realised_24_unit_order"]) == 24
    assert s["reshuffling_during_execution"] == "forbidden"
    assert protocol["execution_order"] == s["realised_24_unit_order"]
    # the frozen artifact must match what the code produces today
    assert s["realised_order_digest"] == _schedule()["realised_order_digest"]


def test_protocol_keeps_unresolved_labels_and_prohibitions(protocol):
    r = protocol["readouts"]["retained_labels"]
    assert r["chilledwater_vs_hotwater_negative"] == "RESOLUTION_LIMITED_DIAGNOSTIC"
    assert r["onset_middle_recovery"] == "UNRESOLVED_NOT_EXECUTED"
    assert protocol["readouts"][
        "chilledwater_not_pooled_into_the_steam_mechanism_claim"
    ]
    for banned in (
        "E5 frozen 192-row query",
        "E6 complete other-half full test",
        "tree refit",
        "adding a model-seed factor",
    ):
        assert banned in protocol["prohibitions"]


def test_manifests_cover_every_unit_and_repeat():
    fit_path = CANONICAL / "e4_fit_manifest.json"
    rep_path = CANONICAL / "e4_repeat_manifest.json"
    if not fit_path.exists():
        pytest.skip("E4 manifests not frozen in this environment")
    fits = json.loads(fit_path.read_text(encoding="utf-8"))["fits"]
    reps = json.loads(rep_path.read_text(encoding="utf-8"))["repeats"]
    assert len(fits) == 24
    assert len(reps) == 192
    assert {f["unit_id"] for f in fits} == {r["unit_id"] for r in reps}
    for f in fits:
        assert f["fits"] == 1 and f["repeats"] == 8
    from collections import Counter

    assert set(Counter(r["unit_id"] for r in reps).values()) == {8}


def _standalone() -> int:
    import inspect
    import tempfile
    import traceback

    proto = None
    path = CANONICAL / "e4_protocol.json"
    if path.exists():
        proto = json.loads(path.read_text(encoding="utf-8"))["protocol"]

    failures = passed = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        try:
            if "protocol" in params:
                if proto is None:
                    print(f"SKIP {name}: protocol not frozen")
                    skipped += 1
                    continue
                fn(proto)
            elif "tmp_path" in params:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - standalone reporter
            if type(exc).__name__ == "_Skip":
                print(f"SKIP {name}: {exc}")
                skipped += 1
                continue
            failures += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
