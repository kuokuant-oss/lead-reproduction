"""RTX PRO 6000 遠端 preflight。任何一項失敗就停止,不進入 benchmark。

在遠端執行,只讀 bundle 內的檔案。bundle 裡沒有 holdout 的任何一列,所以
這支腳本即使被誤用也拿不到 holdout。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_ROWS = 200_000
FEATURES = 137
REQUIRED_N_ESTIMATORS = 8
REQUIRED_GPU_SUBSTRING = "RTX PRO 6000"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def nvidia_smi(query: str) -> str:
    try:
        return (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            .stdout.strip()
            .splitlines()[0]
            .strip()
        )
    except (OSError, IndexError, subprocess.SubprocessError):
        return ""


def collect_environment() -> dict:
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch

    import tabpfn

    props = torch.cuda.get_device_properties(0)
    du = shutil.disk_usage(".")
    mem_total = mem_swap = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) / 1e6
            elif line.startswith("SwapTotal:"):
                mem_swap = int(line.split()[1]) / 1e6
    except OSError:
        pass
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    env = {
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": nvidia_smi("gpu_uuid"),
        "driver": nvidia_smi("driver_version"),
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": props.total_memory / 1e9,
        "gpu_count": torch.cuda.device_count(),
        "cpu_model": cpu_model,
        "cpu_cores": os.cpu_count(),
        "ram_gb": mem_total,
        "swap_gb": mem_swap,
        "disk_free_gb": du.free / 1e9,
        "python": platform.python_version(),
        "tabpfn": tabpfn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    env["environment_digest"] = hashlib.sha256(
        json.dumps(
            {k: env[k] for k in sorted(env) if k not in ("disk_free_gb",)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    from m5_e5_guard import arm, assert_armed

    blocked = arm()
    assert_armed()
    check("no-fit guard armed", len(blocked) > 0, f"{len(blocked)} entry points")

    import torch

    check("CUDA 可見", torch.cuda.is_available())
    if not torch.cuda.is_available():
        atomic_json(
            args.out / "preflight.json", {"checks": checks, "all_passed": False}
        )
        raise SystemExit("CUDA 不可用;禁止 CPU fallback")

    env = collect_environment()
    contract = read_json(args.bundle_root / "environment_contract.json")

    check(
        "架構為 x86_64",
        env["architecture"] == contract["architecture"],
        env["architecture"],
    )
    check(
        f"GPU 為 {REQUIRED_GPU_SUBSTRING}",
        REQUIRED_GPU_SUBSTRING in env["gpu_name"],
        env["gpu_name"],
    )
    check("單一 GPU(不使用 multi-GPU)", env["gpu_count"] == 1, env["gpu_count"])
    for key in ("python", "tabpfn", "torch", "numpy", "pandas", "sklearn", "joblib"):
        want = contract.get(key if key != "sklearn" else "scikit-learn")
        if want is None:
            continue
        got = env[key]
        ok = got == want or (key == "torch" and got.startswith(want))
        check(f"{key} == {want}", ok, got)
    check(
        "CUDA runtime",
        env["cuda_runtime"] == contract["cuda_runtime"],
        env["cuda_runtime"],
    )

    probe_m = read_json(args.bundle_root / "probe_manifest.json")
    probe_npz = args.bundle_root / probe_m["probe"]["path"]
    got = sha256_file(probe_npz)
    check("probe digest 與 manifest 相符", got == probe_m["probe"]["sha256"], got[:16])
    with np.load(probe_npz) as _z:
        _x = np.asarray(_z["x"])
        _raw = np.asarray(_z["raw_index"])
    x_sha = hashlib.sha256(np.ascontiguousarray(_x).tobytes()).hexdigest()
    raw_sha = hashlib.sha256(np.ascontiguousarray(_raw).tobytes()).hexdigest()
    # 內容比對用 array-level digest。.npz 是 zip,其 entry 記錄寫入平台,
    # 那個 byte 進入檔案 digest 卻不影響陣列 —— 用檔案 digest 跨平台比對
    # 會把一份正確的輸入誤判成錯的。
    check("probe x 內容 digest 相符", x_sha == probe_m["probe"]["x_sha256"], x_sha[:16])
    check(
        "probe raw_index 與既有 audit artifact 相符",
        raw_sha == probe_m["probe"]["existing_artifact_raw_index_sha256"],
        raw_sha[:16],
    )
    del _x, _raw
    check("bundle 不含 holdout 列", probe_m["holdout_rows_in_bundle"] == 0)
    check(
        "bundle 不含 holdout raw_index", probe_m["holdout_raw_index_in_bundle"] is False
    )
    check("bundle 不含 score 欄位", probe_m["score_columns_in_bundle"] is False)
    check(
        "bundle 不含 full feature matrix",
        probe_m["full_feature_matrix_in_bundle"] is False,
    )

    with np.load(probe_npz) as z:
        x = np.asarray(z["x"])
    check("probe 形狀", x.shape == (PROBE_ROWS, FEATURES), str(x.shape))
    check("probe dtype 為 float32", x.dtype == np.float32, str(x.dtype))

    sm = read_json(args.bundle_root / "state_manifest.json")
    check("checkpoint digest 已記錄", bool(sm["checkpoint_sha256"]))

    from tabpfn.model_loading import load_fitted_tabpfn_model

    from m5_e5_runner import verify_ensemble, verify_scaler

    import joblib

    reload_ok = 0
    for spec in sm["states"]:
        sp = args.bundle_root / spec["state_file"]
        if sha256_file(sp) != spec["state_sha256"]:
            check(f"{spec['unit_id']} state digest", False)
            continue
        with np.load(args.bundle_root / spec["context_file"]) as z:
            raw_ctx = np.asarray(z["x"])
        if raw_ctx.dtype != np.float32:
            check(f"{spec['unit_id']} context float32", False, str(raw_ctx.dtype))
            continue
        if spec["scaler"]["kind"] == "persisted":
            scaler = joblib.load(args.bundle_root / spec["scaler"]["file"])
        else:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(raw_ctx)
        sc = verify_scaler(scaler, raw_ctx, sp)
        model = load_fitted_tabpfn_model(sp, device="cuda:0")
        ens = verify_ensemble(model)
        ok = (
            sc["exact"]
            and ens["effective_n_estimators_"] == REQUIRED_N_ESTIMATORS
            and ens["len_ensemble_configs_"] == REQUIRED_N_ESTIMATORS
            and all(
                v == REQUIRED_N_ESTIMATORS
                for v in ens["runtime_ensemble_containers"].values()
            )
            and ens["auto_scale_n_estimators"] is False
        )
        check(f"{spec['unit_id']} reload + scaler exact + ensemble=8", ok, ens)
        reload_ok += int(ok)
        del model
        torch.cuda.empty_cache()
    check("三個 state 全部 reload 成功", reload_ok == 3, f"{reload_ok}/3")

    out_files = list(args.out.glob("*.json")) if args.out.exists() else []
    check(
        "output root 為空",
        len([p for p in out_files if p.name != "preflight.json"]) == 0,
    )
    running = subprocess.run(
        [
            "bash",
            "-lc",
            "ps -eo args --no-headers | grep -c '[m]5_e6_gputw_single_worker' || true",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    check("worker count 為 0", running in ("0", ""), running)

    passed = sum(1 for c in checks if c["passed"])
    payload = {
        "schema": "m5_e6_gputw_preflight_v1",
        "generated": time.time(),
        "environment": env,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "holdout_rows_scored": 0,
        "fits_performed": 0,
    }
    digest = atomic_json(args.out / "remote_environment.json", {"environment": env})
    atomic_json(args.out / "preflight.json", payload)
    print(
        f"\npreflight {passed}/{len(checks)} 通過   environment digest={env['environment_digest'][:16]}"
    )
    print(f"remote_environment.json sha256 = {digest}")
    if passed != len(checks):
        print("PREFLIGHT FAILED — 不進入 benchmark")
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
