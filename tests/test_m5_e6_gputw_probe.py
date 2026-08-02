"""GPUtw RTX PRO 6000 probe 的 PHASE A 測試。

檢查的是「守衛會不會真的擋下來」,不是「守衛存在」。每一條禁止事項都有一個
會失敗的路徑被實際觸發 —— 沒被觸發過的守衛只是註解。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m5_e6_gputw_collect as collect  # noqa: E402
import m5_e6_gputw_dual_worker as dual  # noqa: E402
import m5_e6_gputw_prepare_bundle as prep  # noqa: E402
import m5_e6_gputw_single_worker as single  # noqa: E402

ARTIFACTS = Path(r"C:\Users\tonykuo\projects\lead-reproduction") / (
    "data/processed/m5_e6_gputw_probe"
)


def _artifact(name: str) -> dict:
    p = ARTIFACTS / name
    if not p.exists():
        pytest.skip(f"{name} 尚未產生")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# bundle 內容:holdout 絕不能在裡面
# ---------------------------------------------------------------------------


def test_probe_digest_constants_are_full_length_not_abbreviated():
    for d in (
        prep.PROBE_NPZ_SHA256,
        prep.PROBE_RAW_INDEX_SHA256,
        prep.HOLDOUT_SORTED_SHA256,
        prep.CHECKPOINT_SHA256,
    ):
        assert len(d) == 64, "必須是完整 SHA-256,不是報告裡的縮寫"
        assert all(c in "0123456789abcdef" for c in d)


def test_bundle_manifest_declares_no_holdout():
    m = _artifact("probe_manifest.json")
    assert m["holdout_rows_in_bundle"] == 0
    assert m["holdout_raw_index_in_bundle"] is False
    assert m["score_columns_in_bundle"] is False
    assert m["full_feature_matrix_in_bundle"] is False


def test_probe_is_even_building_only_with_zero_holdout_intersection():
    m = _artifact("probe_manifest.json")["probe"]
    assert m["all_even_buildings"] is True
    assert m["holdout_intersection"] == 0
    assert m["rows"] == prep.PROBE_ROWS
    assert m["dtype"] == "float32"


def test_rebuilt_probe_row_set_matches_the_existing_audit_artifact():
    """列集合必須與既有 artifact 完全相同 —— 這是內容正確性的權威檢查。"""
    m = _artifact("probe_manifest.json")["probe"]
    assert m["raw_index_sha256"] == prep.PROBE_RAW_INDEX_SHA256
    assert m["raw_index_matches_existing_artifact"] is True
    assert m["read_from_running_gpu_host"] is False


def test_npz_file_digest_difference_is_recorded_with_its_cause():
    """.npz 是 zip,entry 記錄寫入平台,那個 byte 進 digest 卻不影響陣列。

    既有 artifact 在 Linux 寫、本機重建在 Windows 寫,所以檔案 digest 必然
    不同而內容相同。這個差異必須被記錄並說明原因,不能默默忽略,也不能被
    誤讀成輸入錯誤。
    """
    m = _artifact("probe_manifest.json")["probe"]
    assert m["npz_file_digest_differs_from_existing_artifact"] is True
    assert "create_system" in m["npz_difference_cause"]
    assert m["content_authority"] == "x_sha256 + raw_index_sha256"
    assert len(m["x_sha256"]) == 64


def test_bundle_excludes_the_large_and_dangerous_artifacts():
    b = _artifact("bundle_manifest.json")
    joined = " ".join(b["excluded_by_design"])
    for token in (
        "full feature matrix",
        "full-holdout raw_index",
        "full-holdout score",
        "credentials",
    ):
        assert token in joined
    names = " ".join(b["expected_extracted_tree"])
    assert "holdout" not in names.lower()
    assert "e6_holdout_raw_f4_137" not in names


def test_bundle_contains_no_ssh_key_or_credential():
    b = _artifact("bundle_manifest.json")
    for name in b["expected_extracted_tree"]:
        low = name.lower()
        for bad in ("id_rsa", "id_ed25519", ".pem", ".key", "credential", "token"):
            assert bad not in low, f"bundle 含疑似 credential 檔:{name}"


def test_bundle_carries_the_three_representative_states_only():
    b = _artifact("bundle_manifest.json")
    assert sorted(b["states"]) == sorted(prep.TARGET_UNITS)
    assert len(b["states"]) == 3


# ---------------------------------------------------------------------------
# 環境契約
# ---------------------------------------------------------------------------


def test_environment_contract_pins_exact_versions():
    c = prep.ENVIRONMENT_CONTRACT
    assert c["python"] == "3.12.13"
    assert c["tabpfn"] == "8.0.8"
    assert c["torch"] == "2.12.1"
    assert c["cuda_runtime"] == "13.0"
    assert c["architecture"] == "x86_64"
    assert c["gpu_model_required"] == "RTX PRO 6000"


def test_environment_contract_forbids_the_things_that_would_change_the_path():
    f = " ".join(prep.ENVIRONMENT_CONTRACT["forbidden"])
    for token in (
        "latest",
        "nightly",
        "CPU fallback",
        "torch.compile",
        "mixed precision",
        "TF32",
        "multi-GPU",
        "MPS",
        "MIG",
    ):
        assert token in f


def test_contract_is_not_read_from_the_running_gpu_host():
    assert "未從執行中的 gpu-host 讀取" in prep.ENVIRONMENT_CONTRACT["source"]


# ---------------------------------------------------------------------------
# microbatch 與 worker 規則
# ---------------------------------------------------------------------------


def test_microbatch_is_frozen_at_twenty_thousand():
    assert prep.MICROBATCH == 20_000
    assert single.MICROBATCH == 20_000
    assert prep.MICROBATCHES_PER_STATE == 10
    plan = _artifact("benchmark_plan.json")
    assert plan["microbatch_rows"] == 20_000
    assert plan["microbatch_is_frozen"] is True


def test_plan_forbids_a_third_worker_and_full_holdout():
    plan = _artifact("benchmark_plan.json")
    assert plan["max_workers"] == 2
    assert plan["third_worker"] == "forbidden"
    assert plan["full_holdout_scoring"] == "forbidden"
    assert plan["fits"] == "forbidden"


def test_worker_id_outside_zero_and_one_is_rejected():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "m5_e6_gputw_single_worker.py"),
            "--bundle-root",
            ".",
            "--out",
            ".",
            "--worker-id",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "禁止第三 worker" in (r.stdout + r.stderr)


def test_dual_worker_rounds_swap_slots_and_cover_both_arms():
    plan = _artifact("benchmark_plan.json")["dual_worker_rounds"]
    assert plan["A"][0] == plan["B"][1] and plan["A"][1] == plan["B"][0]
    units = {u for pair in plan.values() for u in pair}
    assert any("frozen_reference" in u for u in units)
    assert any("cell_specific" in u for u in units)


def test_dual_worker_thresholds_are_fixed():
    assert dual.BENEFICIAL == 1.60
    assert dual.MARGINAL == 1.20
    assert dual.STALL_LIMIT_SECONDS == 600


# ---------------------------------------------------------------------------
# collector:必須能拒絕不完整或偽造的輸出
# ---------------------------------------------------------------------------


def _single_record(unit="u", rate=1000.0, batches=10):
    pb = [
        {
            "index": i,
            "rows": 20_000,
            "seconds": 20_000 / rate,
            "rows_per_second": rate,
            "digest": f"{i:064d}",
        }
        for i in range(batches)
    ]
    return {
        "unit_id": unit,
        "per_batch": pb,
        "sustained_rows_per_second": rate,
        "projected_state_hours": collect.FULL_HOLDOUT_ROWS / rate / 3600,
        "scores_retained": 0,
        "fits_performed": 0,
        "aggregate_rows_per_second": rate,
    }


def test_collector_accepts_a_self_consistent_record():
    assert collect.rederive_single(_single_record()) == []


def test_collector_rejects_a_short_run():
    bad = collect.rederive_single(_single_record(batches=7))
    assert any("microbatch" in b for b in bad)


def test_collector_rejects_a_forged_throughput_summary():
    rec = _single_record()
    rec["sustained_rows_per_second"] *= 2  # 只改彙總,不改 per_batch
    bad = collect.rederive_single(rec)
    assert any("sustained_rows_per_second" in b for b in bad)


def test_collector_rejects_an_inconsistent_projection():
    rec = _single_record()
    rec["projected_state_hours"] /= 2
    bad = collect.rederive_single(rec)
    assert any("projected_state_hours" in b for b in bad)


def test_collector_rejects_retained_scores_or_any_fit():
    for field in ("scores_retained", "fits_performed"):
        rec = _single_record()
        rec[field] = 1
        assert any(field in b for b in collect.rederive_single(rec))


def test_collector_rejects_duplicated_microbatch_digests():
    rec = _single_record()
    for b in rec["per_batch"]:
        b["digest"] = "0" * 64  # 全部相同 => 疑似複製輸出
    assert any("digest" in b for b in collect.rederive_single(rec))


def _dual(u0="a", u1="b", same_uuid=False, same_pid=False):
    single_res = {"results": [_single_record("a", 1000.0), _single_record("b", 1000.0)]}
    seq = 2 / (1 / 1000.0 + 1 / 1000.0)
    total = 1600.0
    d = {
        "rounds": {
            "A": {
                "worker0": {
                    "unit_id": u0,
                    "pid": 1,
                    "process_uuid": "x",
                    "aggregate_rows_per_second": 800.0,
                },
                "worker1": {
                    "unit_id": u1,
                    "pid": 1 if same_pid else 2,
                    "process_uuid": "x" if same_uuid else "y",
                    "aggregate_rows_per_second": 800.0,
                },
                "two_worker_aggregate_rows_per_second": total,
                "sequential_equivalent_rows_per_second": seq,
                "aggregate_speedup": total / seq,
            }
        },
        "holdout_rows_scored": 0,
        "scores_retained": 0,
        "fits_performed": 0,
        "third_worker_started": False,
    }
    return d, single_res


def test_collector_accepts_a_consistent_dual_round():
    d, s = _dual()
    assert collect.rederive_dual(d, s) == []


def test_collector_rejects_two_workers_that_are_one_process():
    d, s = _dual(same_uuid=True)
    assert any("同一個 process" in b for b in collect.rederive_dual(d, s))


def test_collector_rejects_two_workers_sharing_a_pid():
    d, s = _dual(same_pid=True)
    assert any("PID" in b for b in collect.rederive_dual(d, s))


def test_collector_rejects_one_state_split_across_two_workers():
    d, s = _dual(u0="a", u1="a")
    assert any("拆給兩個 worker" in b for b in collect.rederive_dual(d, s))


def test_collector_rejects_a_forged_speedup():
    d, s = _dual()
    d["rounds"]["A"]["aggregate_speedup"] = 3.0
    assert any("speedup" in b for b in collect.rederive_dual(d, s))


def test_collector_rejects_a_declared_third_worker():
    d, s = _dual()
    d["third_worker_started"] = True
    assert any("第三 worker" in b for b in collect.rederive_dual(d, s))


# ---------------------------------------------------------------------------
# 不得碰正式 E6,不得洩漏 credential
# ---------------------------------------------------------------------------


def test_no_probe_script_touches_the_current_e6_outputs():
    forbidden = (
        "m5-e6-run",
        "lead-reproduction-e6-run",
        "tmux e6",
        "e6_holdout_raw_f4_137",
        "m5_e6_features",
    )
    for s in (ROOT / "scripts").glob("m5_e6_gputw_*"):
        text = s.read_text(encoding="utf-8")
        for token in forbidden:
            if token == "m5_e6_features" and s.name == "m5_e6_gputw_prepare_bundle.py":
                continue  # 只讀既有 352 列 sentinel,不讀 holdout matrix
            assert token not in text, f"{s.name} 提到現行 E6 的 {token}"


def _code_lines(path: Path) -> str:
    """去掉註解後的內容。註解裡談到某個危險設定,不等於使用了它。"""
    return "\n".join(
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_launcher_reads_credentials_only_from_environment():
    path = ROOT / "scripts" / "m5_e6_gputw_launch.sh"
    text = path.read_text(encoding="utf-8")
    code = _code_lines(path)
    for var in ("GPUTW_HOST", "GPUTW_USER", "GPUTW_PORT", "GPUTW_SSH_KEY"):
        assert f'"${{{var}' in text or f"${var}" in text
    assert "StrictHostKeyChecking=no" not in code, "實際設定裡不得停用 host key 檢查"
    assert "StrictHostKeyChecking=yes" in code
    assert "BatchMode=yes" in code
    assert "IdentitiesOnly=yes" in code
    assert "ServerAliveInterval" in code


def test_launcher_enforces_key_permissions_and_gpu_model():
    text = (ROOT / "scripts" / "m5_e6_gputw_launch.sh").read_text(encoding="utf-8")
    assert "0600" in text
    assert "RTX PRO 6000" in text


def test_no_credential_material_in_any_tracked_probe_file():
    # token 用組合方式寫,否則這個檔案會掃到自己的字面值而誤報。
    markers = [
        "BEGIN " + "OPENSSH PRIVATE KEY",
        "BEGIN " + "RSA PRIVATE KEY",
        "ssh-" + "rsa AAAA",
        "ssh-" + "ed25519 AAAA",
    ]
    for p in (ROOT / "scripts").glob("m5_e6_gputw_*"):
        text = p.read_text(encoding="utf-8")
        for token in markers:
            assert token not in text, f"{p.name} 含 key material"


def test_gitignore_covers_the_runtime_credential_directory():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".gputw-probe-runtime" in gi


# ---------------------------------------------------------------------------
# no-fit guard 真的會擋
# ---------------------------------------------------------------------------


def test_no_fit_guard_raises_when_a_fit_is_attempted():
    sys.path.insert(0, str(ROOT / "scripts"))
    from m5_e5_guard import FitAttemptedError, arm, assert_armed

    blocked = arm()
    assert_armed()
    assert len(blocked) > 0
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    import tabpfn

    with pytest.raises(FitAttemptedError):
        tabpfn.TabPFNClassifier().fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# 遠端腳本 dry run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "m5_e6_gputw_launch.sh",
        "m5_e6_gputw_remote_setup.sh",
        "m5_e6_gputw_status.sh",
        "m5_e6_gputw_abort.sh",
    ],
)
def test_shell_scripts_parse(name):
    r = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_launcher_fails_closed_without_connection_variables():
    env = {"PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "m5_e6_gputw_launch.sh")],
        capture_output=True,
        env=env,
    )
    assert r.returncode != 0
    combined = (r.stdout or b"") + (r.stderr or b"")
    assert b"GPUTW_HOST" in combined


def test_abort_script_excludes_its_own_pid():
    text = (ROOT / "scripts" / "m5_e6_gputw_abort.sh").read_text(encoding="utf-8")
    assert "SELF=$$" in text
    assert '[ "$pid" = "$SELF" ] && continue' in text
