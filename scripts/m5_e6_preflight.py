"""M5 E6 remote preflight: everything checkable before the first holdout row.

This runs on the execution host and refuses to report readiness on anything it
did not verify itself. It re-digests the frozen artifacts, the 24 persisted
states, the context caches and the feature matrix; it arms the no-fit guard; and
it reloads one state on the GPU to prove the ensemble contract and the scaler
path hold in this environment rather than in the one that froze the protocol.

It scores nothing. The sentinel and the holdout both wait for the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROWS = 10_137_155
FEATURES = 137
STATES = 24


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--context-cache", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print("--- frozen artifacts ---")
    inputs = read_json(args.protocol_root / "e6_input_manifest.json")
    for name, key in (
        ("e6_protocol.json", "protocol_sha256"),
        ("e6_row_manifest.json", "row_manifest_sha256"),
        ("e6_shard_manifest.json", "shard_manifest_sha256"),
        ("e6_microbatch_manifest.json", "microbatch_manifest_sha256"),
        ("e6_state_manifest.json", "state_manifest_sha256"),
        ("e6_sentinel_manifest.json", "sentinel_manifest_sha256"),
        ("e6_tree_manifest.json", "tree_manifest_sha256"),
        ("e6_bootstrap_manifest.json", "bootstrap_manifest_sha256"),
        ("e6_decision_rules.json", "decision_rules_sha256"),
        ("e6_cost_model.json", "cost_model_sha256"),
    ):
        got = sha256_file(args.protocol_root / name)
        check(f"artifact digest {name}", got == inputs[key], got[:16])

    proto = read_json(args.protocol_root / "e6_protocol.json")["protocol"]
    check(
        "protocol is authorised for execution",
        proto["authorization"]["execution_status"] == "HUMAN_AUTHORIZED_FOR_EXECUTION",
    )
    check(
        "repeat policy is R1_PLUS_SENTINEL",
        proto["repeat_policy"]["name"] == "R1_PLUS_SENTINEL",
    )
    check(
        "one full-holdout pass per state",
        proto["repeat_policy"]["full_holdout_passes_per_state"] == 1,
    )

    print("\n--- source digests ---")
    repo = Path(__file__).resolve().parents[1]
    drift = []
    for rel, want in proto["source_digests"].items():
        got = sha256_source(repo / rel)
        if got != want:
            drift.append(rel)
    check("source files match the frozen digests", not drift, drift or "all match")

    print("\n--- microbatch plan ---")
    mb = read_json(args.protocol_root / "e6_microbatch_manifest.json")
    entries = mb["microbatches"]
    covered = sum(e["row_count"] for e in entries)
    bounds_ok = all(
        e["canonical_stop"] - e["canonical_start"] == e["row_count"] for e in entries
    )
    contiguous = all(
        entries[i]["canonical_stop"] == entries[i + 1]["canonical_start"]
        for i in range(len(entries) - 1)
    )
    check("every row covered exactly once", covered == ROWS, f"{covered:,}")
    check("microbatch bounds are self-consistent", bounds_ok)
    check("microbatches partition the holdout contiguously", contiguous)
    check(
        "census matches the manifest length",
        mb["census"]["microbatches_per_state"] == len(entries),
        len(entries),
    )

    print("\n--- feature matrix ---")
    feat = read_json(args.feature_root / "e6_feature_manifest.json")
    matrix = args.feature_root / feat["path"]
    got = sha256_file(matrix)
    check("feature matrix digest matches its manifest", got == feat["sha256"], got[:16])
    check(
        "feature matrix digest matches the frozen protocol",
        got == proto["feature_artifact"]["sha256"],
    )
    x = np.load(matrix, mmap_mode="r")
    check("feature matrix shape", x.shape == (ROWS, FEATURES), str(x.shape))
    check("feature matrix dtype is float32", x.dtype == np.float32, str(x.dtype))
    check("feature matrix is unscaled", feat["scaled"] is False)
    raw_index = np.load(args.feature_root / feat["raw_index_path"])
    check(
        "canonical raw_index digest",
        hashlib.sha256(raw_index.tobytes()).hexdigest() == feat["raw_index_sha256"],
    )
    check(
        "sorted raw_index reproduces the frozen holdout",
        hashlib.sha256(np.sort(raw_index).tobytes()).hexdigest()
        == proto["holdout"]["sorted_raw_index_sha256"],
    )
    sent = args.feature_root / feat["sentinel_query_path"]
    check(
        "sentinel query digest",
        sha256_file(sent) == feat["sentinel_query_sha256"],
    )

    print("\n--- states and contexts ---")
    states = read_json(args.protocol_root / "e6_state_manifest.json")["states"]
    check("state manifest holds 24 units", len(states) == STATES, len(states))
    order = hashlib.sha256(
        "\n".join(s["unit_id"] for s in states).encode("utf-8")
    ).hexdigest()
    check(
        "execution order reproduces E4's frozen order",
        order == proto["execution_order"]["realised_order_digest"],
        order[:16],
    )
    bad_states, bad_ctx = [], []
    for s in states:
        if sha256_file(args.repo_root / s["state_path"]) != s["state_sha256"]:
            bad_states.append(s["unit_id"])
        c = args.context_cache / f"seed{s['context_seed']}__cell{s['cell']}.npz"
        meta = read_json(c.with_suffix(".json"))
        if sha256_file(c) != meta["npz_sha256"]:
            bad_ctx.append(c.name)
        elif meta["context_manifest_sha256"] != s["context_manifest_sha256"]:
            bad_ctx.append(f"{c.name} (wrong context manifest)")
    check("all 24 state digests match", not bad_states, bad_states or "24/24")
    check("all context caches match", not bad_ctx, bad_ctx or "ok")

    print("\n--- environment ---")
    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()
    check("no-fit guard armed", len(blocked) > 0, f"{len(blocked)} entry points")

    import torch
    from tabpfn import __version__ as tabpfn_version

    check(
        "TabPFN version", tabpfn_version == proto["tabpfn"]["version"], tabpfn_version
    )
    check("CUDA available (CPU fallback prohibited)", torch.cuda.is_available())

    free_gb = shutil.disk_usage(args.out).free / 1e9
    need_gb = STATES * ROWS * 4 / 1e9 * 1.5
    check(
        "disk space for outputs",
        free_gb > need_gb,
        f"{free_gb:.1f} GB free, need ~{need_gb:.1f} GB",
    )

    print("\n--- one live state ---")
    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler
    from m5_e6_runner import load_scaler

    probe = states[0]
    with np.load(
        args.context_cache / f"seed{probe['context_seed']}__cell{probe['cell']}.npz"
    ) as z:
        raw_ctx = np.asarray(z["x"])
    scaler = load_scaler(probe, raw_ctx, args.repo_root)
    sc = verify_scaler(scaler, raw_ctx, args.repo_root / probe["state_path"])
    check(f"scaler reproduces E4's context ({probe['unit_id']})", sc["exact"], sc)
    t0 = time.perf_counter()
    model = load_fitted_tabpfn_model(
        args.repo_root / probe["state_path"], device="cuda:0"
    )
    ens = verify_ensemble(model)
    check(
        "ensemble contract is exactly 8",
        ens["effective_n_estimators_"] == 8,
        f"reload {time.perf_counter() - t0:.1f}s",
    )
    del model
    torch.cuda.empty_cache()

    passed = sum(1 for c in checks if c["passed"])
    payload = {
        "schema": "m5_e6_preflight_v1",
        "generated": time.time(),
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tabpfn": tabpfn_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "holdout_rows_scored": 0,
        "sentinel_rows_scored": 0,
    }
    digest = atomic_json(args.out / "e6_preflight.json", payload)
    print(f"\npreflight {passed}/{len(checks)} passed   sha256={digest}")
    if passed != len(checks):
        print("PREFLIGHT FAILED -- not launching")
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
