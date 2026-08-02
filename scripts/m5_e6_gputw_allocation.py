"""完整 seed block 分工的可行性,以及一份不可啟動的 override 草案。

只依 state identity、seed、cell、arm、凍結執行順序與完成計數決定,
絕不查看任何 scientific score value。完成計數由呼叫端從 monitor heartbeat
傳入,這支腳本不去讀遠端輸出目錄,也不需要。

本模組最重要的產出不是分配建議,而是一個時間事實:E4 的隨機化執行順序把
三個 seed 交錯排列,所以「完整未開始的 seed block」會隨著現行 run 前進而
一個個消失。窗口寬度是可以精確算出來的,而它比直覺短很多。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

STATE_HOURS = 1.99  # 現行 gpu-host 實測,1,414-1,420 rows/s
FULL_HOLDOUT_ROWS = 10_137_155
SEEDS = (42, 123, 999)


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def seed_blocks(states: list[dict]) -> dict[int, dict]:
    """每個 seed 的 8 個 state 在凍結執行順序中的位置。"""
    out = {}
    for seed in SEEDS:
        members = [s for s in states if s["context_seed"] == seed]
        positions = sorted(s["position"] for s in members)
        out[seed] = {
            "states": len(members),
            "positions": positions,
            "first_position": positions[0],
            "last_position": positions[-1],
            "cells": sorted({s["cell"] for s in members}),
            "arms": sorted({s["scaler_arm"] for s in members}),
            "unit_ids": [
                s["unit_id"] for s in sorted(members, key=lambda m: m["position"])
            ],
        }
        if len(members) != 8:
            raise SystemExit(f"seed{seed} 有 {len(members)} 個 state,預期 8")
    return out


def availability(blocks: dict[int, dict], completed: int, running: int | None) -> dict:
    """哪些 seed block 仍完整未開始,以及各自還剩多久。

    現行 run 依凍結順序執行 position 0,1,2,...。`completed` 是已完成的
    state 數,因此正在執行的是 position == completed。一個 seed block 只要
    first_position > completed 就仍完整未開始。
    """
    current = completed if running is None else running
    out = {}
    for seed, b in blocks.items():
        started = b["first_position"] <= current
        # 現行 run 還要多久才會碰到這個 block 的第一個 state
        hours_left = max(0.0, (b["first_position"] - current) * STATE_HOURS)
        out[seed] = {
            **b,
            "fully_unstarted": not started,
            "current_position": current,
            "hours_until_current_run_reaches_this_block": hours_left,
            "window_closes_at_position": b["first_position"],
        }
    return out


def queue_rule(block: dict) -> dict:
    """固定、可重現、不看結果的 state queue。

    以 cell 為配對單位交替,而不是逐一輪流。逐一輪流在這裡是錯的:E4 的凍結
    順序把同一個 cell 的兩個 scaler arm 排成相鄰,所以「第 i 個給 worker
    i mod 2」會把 cell_specific 整個給 worker 0、frozen_reference 整個給
    worker 1 —— 正是規則禁止的「某一 arm 全部固定給同一個 worker」。這個錯誤
    是被測試抓出來的,不是事後想到的。

    改成以 cell 為單位交替後,每個 worker 拿到兩個 cell 的完整 arm 對,因此
    兩個 worker 都同時見到兩個 arm。兩個 worker 在同一台機器同一張 GPU 上,
    worker 只是排程 slot,不是不同的執行環境,所以這個分派本身不引入任何
    machine confounding;它要避免的只是 arm 與 slot 的系統性綁定。
    """
    units = block["unit_ids"]
    by_cell: dict[str, list[str]] = {}
    for u in units:  # 已依 position 排序,故各 cell 內的 arm 次序也是固定的
        by_cell.setdefault(u.split("__")[1], []).append(u)
    slots = {0: [], 1: []}
    for j, cell in enumerate(by_cell):  # dict 保序,來源是 position 序
        slots[j % 2].extend(by_cell[cell])
    arms = {w: sorted({u.rsplit("__", 1)[1] for u in us}) for w, us in slots.items()}
    cells = {w: sorted({u.split("__")[1] for u in us}) for w, us in slots.items()}
    balanced_arms = all(len(a) == 2 for a in arms.values())
    return {
        "rule": "以 cell 為配對單位交替:依 position 序取出各 cell,第 j 個 cell "
        "的兩個 scaler arm 一起指派給 worker (j mod 2)",
        "why_not_round_robin": (
            "E4 的凍結順序把同一 cell 的兩個 arm 排成相鄰,逐一輪流會把某一個 "
            "scaler arm 整個綁在單一 worker,違反分派規則。"
        ),
        "deterministic": True,
        "depends_on_results": False,
        "worker_slots": slots,
        "arms_per_worker": arms,
        "cells_per_worker": cells,
        "both_workers_see_both_arms": balanced_arms,
        "states_per_worker": {w: len(us) for w, us in slots.items()},
        "one_state_at_a_time_per_worker": True,
        "next_state_only_after_previous_completes": True,
        "process_pool_executor_for_models": "forbidden",
        "outer_queue": "explicit worker slot, 每個 state 記錄 worker ID、"
        "GPU UUID 與 process UUID",
    }


def draft_override(
    candidate: str, block_seed: int, block: dict, queue: dict, protocol_sha: str
) -> dict:
    return {
        "schema": "m5_e6_gputw_execution_override_DRAFT_v1",
        "generated": time.time(),
        "launchable": False,
        "human_authorisation_required": True,
        "is_frozen": False,
        "original_protocol_sha256": protocol_sha,
        "original_protocol_commit": "3a800a0",
        "original_execution_host": "gpu-host, single GPU worker, single tmux session",
        "gputw_host_candidate": candidate,
        "transferred_seed_block": block_seed,
        "transferred_unit_ids": block["unit_ids"],
        "states_already_started_stay_on_original_host": True,
        "no_state_is_split": True,
        "no_cross_host_microbatch_splice": True,
        "one_state_one_host_one_worker_one_process": True,
        "endpoints_unchanged": True,
        "decision_rules_unchanged": True,
        "tree_outputs_unchanged": True,
        "gputw_tree_outputs": "forbidden",
        "gputw_full_holdout_scoring_this_round": "forbidden",
        "queue_rule": queue,
        "execution_provenance_limitation": (
            "以 seed block 分 host,會讓 execution host 與 context seed 綁定。"
            "這不是原始的單 host protocol。該 seed 內的四個 factorial cell、"
            "兩個 scaler arm、negative-support contrast、interaction 與 scaler "
            "比較確實都留在同一個執行環境,所以 seed 內的對比不被 machine 切開;"
            "但 seed 之間的比較就同時混入了機器差異。這只削減了 machine "
            "confounding,並未消除 —— 不得宣稱已完全消除 machine confounding。"
        ),
        "requires_human_execution_override_before_use": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-manifest", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument(
        "--completed",
        type=int,
        required=True,
        help="現行 run 已完成的 state 數,取自 monitor heartbeat,不讀 score",
    )
    ap.add_argument("--candidate", default="UNDECIDED")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    states = json.loads(args.state_manifest.read_text(encoding="utf-8"))["states"]
    protocol_sha = json.loads(args.input_manifest.read_text(encoding="utf-8"))[
        "protocol_sha256"
    ]
    blocks = seed_blocks(states)
    avail = availability(blocks, args.completed, None)

    unstarted = [s for s, b in avail.items() if b["fully_unstarted"]]
    print(
        f"現行 run 已完成 {args.completed} 個 state,正在執行 position {args.completed}"
    )
    for seed, b in sorted(avail.items(), key=lambda kv: kv[1]["first_position"]):
        mark = "完整未開始" if b["fully_unstarted"] else "已開始"
        print(
            f"  seed{seed}: positions {b['positions']}  first={b['first_position']:>2}  "
            f"{mark}  現行 run 還有 {b['hours_until_current_run_reaches_this_block']:.1f} h "
            f"才會碰到"
        )

    recommendation = {}
    if not unstarted:
        recommendation = {
            "action": "STOP",
            "reason": "沒有任何完整未開始的 seed block。依規定不得提出任意 state 切割。",
        }
    else:
        # 優先給窗口最寬的 block:現行 run 最晚才會碰到的那個。
        best = max(unstarted, key=lambda s: avail[s]["first_position"])
        recommendation = {
            "action": "TRANSFER_ONE_COMPLETE_SEED_BLOCK",
            "seed": best,
            "unit_ids": avail[best]["unit_ids"],
            "reason": (
                f"seed{best} 是仍完整未開始、且現行 run 最晚才會碰到的 block,"
                f"窗口最寬({avail[best]['hours_until_current_run_reaches_this_block']:.1f} 小時)。"
            ),
            "second_block_if_two_are_wanted": sorted(
                (s for s in unstarted if s != best),
                key=lambda s: -avail[s]["first_position"],
            ),
        }

    queue = (
        queue_rule(blocks[recommendation["seed"]]) if "seed" in recommendation else {}
    )
    payload = {
        "schema": "m5_e6_gputw_state_allocation_v1",
        "generated": time.time(),
        "reads_scientific_scores": False,
        "allocation_inputs": [
            "state identity",
            "seed",
            "cell",
            "scaler arm",
            "frozen execution order",
            "completed count from heartbeat",
        ],
        "state_hours_measured": STATE_HOURS,
        "completed_states": args.completed,
        "seed_blocks": avail,
        "fully_unstarted_seed_blocks": sorted(unstarted),
        "execution_order_is_interleaved": True,
        "interleaving_consequence": (
            "E4 的隨機化順序把三個 seed 交錯排列,所以 seed block 不是"
            "「前 8 個、中 8 個、後 8 個」。現行 run 每完成一個 state 就可能"
            "吃掉某個 block 的完整性,窗口是持續收斂的,不是靜態的。"
        ),
        "hard_constraints": {
            "one_state_one_host": True,
            "one_state_one_worker": True,
            "one_state_one_process": True,
            "no_state_splitting": True,
            "no_cross_host_microbatch_splice": True,
            "started_states_never_transfer": True,
            "completed_states_never_rerun_and_cherry_picked": True,
            "allocation_never_inspects_scores": True,
        },
        "recommendation": recommendation,
        "queue_rule": queue,
    }
    digest = atomic_json(args.out / "state_allocation.json", payload)

    if "seed" in recommendation:
        draft = draft_override(
            args.candidate,
            recommendation["seed"],
            blocks[recommendation["seed"]],
            queue,
            protocol_sha,
        )
        d2 = atomic_json(args.out / "e6_gputw_execution_override.DRAFT.json", draft)
        print(f"\ndraft override sha256 = {d2}  (launchable=False)")
    print(f"state_allocation.json sha256 = {digest}")
    print(
        f"建議:{recommendation['action']}"
        + (f" seed{recommendation['seed']}" if "seed" in recommendation else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
