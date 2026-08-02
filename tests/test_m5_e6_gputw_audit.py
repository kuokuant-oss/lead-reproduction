"""GPUtw 稽核的測試:每一條安全性質都要有一個會失敗的路徑被實際觸發。

這些測試不需要 GPU,也不接觸現行 E6 的任何檔案。它們檢查的是「如果有人
把 holdout 餵進 benchmark、把兩個 worker 寫成同一個 process、或在沒有實測
throughput 的情況下報出成本數字,程式會不會擋下來」。沒被觸發過的守衛只是
註解。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import m5_e6_gputw_allocation as alloc  # noqa: E402
import m5_e6_gputw_cost as cost  # noqa: E402
import m5_e6_gputw_devices as dev  # noqa: E402
import m5_e6_gputw_guard as guard  # noqa: E402
import m5_e6_gputw_two_worker as two  # noqa: E402


# ---------------------------------------------------------------------------
# non-holdout 保證
# ---------------------------------------------------------------------------


def _holdout(n=guard.HOLDOUT_ROWS):
    return np.arange(1, 2 * n, 2, dtype="int64")  # 奇數


def test_disjoint_proof_accepts_even_only_probe(monkeypatch):
    monkeypatch.setattr(guard, "HOLDOUT_ROWS", 1000)
    monkeypatch.setattr(guard, "PROBE_ROWS", 100)
    holdout = np.arange(1, 2001, 2, dtype="int64")
    monkeypatch.setattr(
        guard,
        "HOLDOUT_SORTED_DIGEST",
        __import__("hashlib").sha256(np.sort(holdout).tobytes()).hexdigest(),
    )
    probe = np.arange(0, 200, 2, dtype="int64")
    proof = guard.disjoint_proof(probe, holdout)
    assert proof["intersection_size"] == 0


def test_disjoint_proof_rejects_any_holdout_row(monkeypatch):
    monkeypatch.setattr(guard, "HOLDOUT_ROWS", 1000)
    monkeypatch.setattr(guard, "PROBE_ROWS", 100)
    holdout = np.arange(1, 2001, 2, dtype="int64")
    monkeypatch.setattr(
        guard,
        "HOLDOUT_SORTED_DIGEST",
        __import__("hashlib").sha256(np.sort(holdout).tobytes()).hexdigest(),
    )
    probe = np.arange(0, 200, 2, dtype="int64")
    probe[5] = 7  # 一列 holdout 就足以否決
    with pytest.raises(SystemExit, match="重疊"):
        guard.disjoint_proof(probe, holdout)


def test_disjoint_proof_rejects_a_wrong_holdout_set(monkeypatch):
    monkeypatch.setattr(guard, "HOLDOUT_ROWS", 1000)
    monkeypatch.setattr(guard, "PROBE_ROWS", 100)
    with pytest.raises(SystemExit, match="holdout digest 不符"):
        guard.disjoint_proof(
            np.arange(0, 200, 2, dtype="int64"),
            np.arange(3, 2003, 2, dtype="int64"),
        )


def test_holdout_digest_constant_matches_the_frozen_protocol():
    assert guard.HOLDOUT_SORTED_DIGEST.startswith("f0867d3e")
    assert guard.PROBE_ARTIFACT_DIGEST.startswith("afe80b11")


# ---------------------------------------------------------------------------
# 設備清查的事實紀律
# ---------------------------------------------------------------------------


def test_dgx_unified_memory_is_not_called_vram():
    c = dev.CANDIDATES["dgx_spark_gb10"]
    assert "unified" in c["memory_model"]
    assert "VRAM" in c["MEMORY_MODEL_WARNING"]
    assert "不是" in c["MEMORY_MODEL_WARNING"]


def test_dgx_is_recorded_as_aarch64():
    assert dev.CANDIDATES["dgx_spark_gb10"]["cpu_architecture"] == "aarch64"
    assert dev.CANDIDATES["rtx_pro_6000_ws"]["cpu_architecture"] == "x86_64"


def test_torch_arm64_is_flagged_as_blocking():
    t = dev.COMPATIBILITY["dgx_spark_gb10"]["torch_2_12_1_aarch64_cuda13"]
    assert t["status"] == "BLOCKING_RISK"
    assert dev.COMPATIBILITY["dgx_spark_gb10"]["overall"] == "ARM64_TOOLCHAIN_RISK"


def test_tabpfn_is_pure_python_so_not_the_arm_blocker():
    t = dev.COMPATIBILITY["dgx_spark_gb10"]["tabpfn_8_0_8_package"]
    assert t["status"] == "COMPATIBLE"
    assert "py3-none-any" in t["evidence"]


def test_pricing_is_unverified_and_carries_no_guessed_number():
    p = dev.PRICING
    assert p["status"] == "PRICE_UNVERIFIED"
    assert p["dgx_spark_hourly_ntd"] is None
    assert p["rtx_pro_6000_hourly_ntd"] is None
    assert p["minimum_billing_unit"] is None


def test_capacity_is_explicitly_separated_from_benefit():
    note = dev.CONCURRENCY["capacity_is_not_the_question"]
    assert "容量" in note and "拒絕" in note


def test_prior_expectation_is_not_presented_as_a_verdict():
    assert "不是實測結果" in dev.CONCURRENCY["prior_expectation_not_a_result"]


def test_inventory_declares_zero_holdout_and_zero_fits():
    inv = dev.inventory()
    assert inv["holdout_rows_scored"] == 0
    assert inv["fits_performed"] == 0
    assert inv["benchmarks_executed"] == 0
    assert inv["current_e6_untouched"] is True


# ---------------------------------------------------------------------------
# seed block 分配
# ---------------------------------------------------------------------------


def _states():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/reports/m5-e6-artifacts/e6_state_manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["states"]


def test_each_seed_block_has_eight_states():
    blocks = alloc.seed_blocks(_states())
    assert set(blocks) == {42, 123, 999}
    for b in blocks.values():
        assert b["states"] == 8
        assert len(b["cells"]) == 4
        assert len(b["arms"]) == 2


def test_execution_order_is_interleaved_not_blocked():
    blocks = alloc.seed_blocks(_states())
    # 若順序是分塊的,每個 seed 的 position 會是連續的 8 個。
    for b in blocks.values():
        p = b["positions"]
        assert p != list(range(p[0], p[0] + 8)), "順序若連續,交錯的結論就不成立"


def test_a_block_is_unstarted_only_if_its_first_position_is_ahead():
    blocks = alloc.seed_blocks(_states())
    avail = alloc.availability(blocks, completed=0, running=None)
    # position 0 屬於 seed42,所以 completed=0(正在跑 position 0)時 seed42 已開始
    assert avail[42]["fully_unstarted"] is False
    assert avail[123]["fully_unstarted"] is True
    assert avail[999]["fully_unstarted"] is True


def test_blocks_close_as_the_run_advances():
    blocks = alloc.seed_blocks(_states())
    late = alloc.availability(blocks, completed=10, running=None)
    assert late[123]["fully_unstarted"] is False
    assert late[999]["fully_unstarted"] is False


def test_queue_rule_is_deterministic_and_result_blind():
    blocks = alloc.seed_blocks(_states())
    q1 = alloc.queue_rule(blocks[999])
    q2 = alloc.queue_rule(blocks[999])
    assert q1["worker_slots"] == q2["worker_slots"]
    assert q1["deterministic"] is True
    assert q1["depends_on_results"] is False


def test_queue_rule_never_gives_one_arm_entirely_to_one_worker():
    blocks = alloc.seed_blocks(_states())
    for seed in (42, 123, 999):
        q = alloc.queue_rule(blocks[seed])
        assert q["both_workers_see_both_arms"], (
            f"seed{seed} 把某個 arm 綁死在單一 worker"
        )


def test_queue_rule_splits_eight_states_four_and_four():
    blocks = alloc.seed_blocks(_states())
    q = alloc.queue_rule(blocks[999])
    assert q["states_per_worker"] == {0: 4, 1: 4}


def test_draft_override_is_not_launchable():
    blocks = alloc.seed_blocks(_states())
    d = alloc.draft_override(
        "rtx_pro_6000_ws", 999, blocks[999], alloc.queue_rule(blocks[999]), "0" * 64
    )
    assert d["launchable"] is False
    assert d["human_authorisation_required"] is True
    assert d["is_frozen"] is False
    assert d["gputw_tree_outputs"] == "forbidden"
    assert d["no_cross_host_microbatch_splice"] is True
    assert d["one_state_one_host_one_worker_one_process"] is True


def test_draft_override_admits_the_confounding_it_creates():
    blocks = alloc.seed_blocks(_states())
    d = alloc.draft_override(
        "rtx_pro_6000_ws", 999, blocks[999], alloc.queue_rule(blocks[999]), "0" * 64
    )
    lim = d["execution_provenance_limitation"]
    assert "並未消除" in lim
    assert "不得宣稱" in lim


# ---------------------------------------------------------------------------
# 兩 worker 判定
# ---------------------------------------------------------------------------


def test_verdict_thresholds():
    assert two.verdict(1.80, True, True) == "TWO_WORKERS_BENEFICIAL"
    assert two.verdict(1.40, True, True) == "TWO_WORKERS_MARGINAL"
    assert two.verdict(1.05, True, True) == "TWO_WORKERS_HARMFUL"


def test_unstable_rounds_cannot_be_beneficial():
    assert two.verdict(1.90, False, True) == "TWO_WORKERS_MARGINAL"


def test_swap_or_dirty_run_is_harmful_regardless_of_speed():
    assert two.verdict(1.95, True, False) == "TWO_WORKERS_HARMFUL"


def test_rounds_swap_the_two_states_between_slots():
    a, b = two.ROUNDS["A"], two.ROUNDS["B"]
    assert a[0] == b[1] and a[1] == b[0]


def test_three_rounds_are_defined_and_use_both_arms():
    assert set(two.ROUNDS) == {"A", "B", "C"}
    units = {u for pair in two.ROUNDS.values() for u in pair}
    assert any("frozen_reference" in u for u in units)
    assert any("cell_specific" in u for u in units)


# ---------------------------------------------------------------------------
# 成本模型
# ---------------------------------------------------------------------------


def test_cost_is_null_without_a_verified_price():
    d = cost.device_model("x", 1500.0, 2400.0, None)
    assert d["eight_state_block_cost_ntd_single_worker"] is None
    assert d["eight_state_block_cost_ntd_two_worker"] is None


def test_hours_are_null_without_measured_throughput():
    d = cost.device_model("x", None, None, 100.0)
    assert d["single_worker_state_hours"] is None
    assert d["eight_state_block_hours_single_worker"] is None


def test_cost_is_derived_from_measured_throughput_not_a_constant():
    fast = cost.device_model("fast", 2840.0, None, None)
    slow = cost.device_model("slow", 1420.0, None, None)
    assert fast["single_worker_state_hours"] == pytest.approx(
        slow["single_worker_state_hours"] / 2
    )


def test_two_worker_aggregate_time_uses_aggregate_throughput():
    d = cost.device_model("x", 1420.0, 2840.0, None)
    # 8 個 state,合計吞吐 2 倍 => 牆鐘時間是單 worker 的一半
    assert d["eight_state_block_hours_two_worker"] == pytest.approx(
        d["eight_state_block_hours_single_worker"] / 2
    )


def test_breakeven_says_never_when_the_device_is_slower():
    d = cost.device_model("slow", 700.0, None, None)
    assert isinstance(d["breakeven_state_count"], str)
    assert "never" in d["breakeven_state_count"]


def test_breakeven_is_positive_when_the_device_is_faster():
    d = cost.device_model("fast", 2840.0, None, None)
    assert isinstance(d["breakeven_state_count"], float)
    assert d["breakeven_state_count"] > 0


def test_baseline_marginal_cost_is_zero_because_gpu_host_is_owned():
    # 這是成本比較的關鍵不對稱,必須寫死在模型裡而不是留給讀者推。
    assert cost.BASELINE_STATE_HOURS == pytest.approx(1.983, abs=0.01)


def test_full_feature_matrix_is_not_transferred_this_round():
    assert cost.FEATURE_MATRIX_STRATEGY["rebuild_on_device"]["recommended"] is True
    assert "不得實際傳輸" in cost.FEATURE_MATRIX_STRATEGY["transfer"]["note"]


# ---------------------------------------------------------------------------
# 不得碰現行 E6
# ---------------------------------------------------------------------------


def test_no_script_writes_into_the_current_e6_output_roots():
    scripts = (Path(__file__).resolve().parents[1] / "scripts").glob("m5_e6_gputw_*.py")
    forbidden = ("m5-e6-run", "m5_e6_protocol", "lead-reproduction-e6-run")
    for s in scripts:
        text = s.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{s.name} 提到現行 E6 的輸出路徑 {token}"


def test_credentials_are_not_present_in_any_audit_artifact():
    root = Path(__file__).resolve().parents[1]
    for s in (root / "scripts").glob("m5_e6_gputw_*.py"):
        text = s.read_text(encoding="utf-8").lower()
        for token in (
            "api_key",
            "api-token",
            "bearer ",
            "ssh-rsa",
            "begin private key",
        ):
            assert token not in text, f"{s.name} 可能含 credential"
