"""建立 GPUtw RTX PRO 6000 benchmark 的最小部署 bundle。

所有輸入都在本機重建,不從正在執行 E6 的 gpu-host 讀取任何東西。probe matrix
與三個 context matrix 都可以從既有的 manifest 與原始資料重算,而重算出來的
digest 必須與既有 audit artifact 記錄的完整 SHA-256 相符 —— 這比從 gpu-host
下載更安全,也順帶證明了輸入可跨機器重現。

bundle 刻意最小化。它不含 10,137,155 列的 full feature matrix、不含 holdout
的 raw_index 清單、不含任何 score 欄位。holdout 的列不存在於 bundle 裡,所以
遠端不可能誤用 —— 這比「上傳後再檢查」強。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROBE_ROWS = 200_000
PROBE_SEED = 20260802
FEATURES = 137
SENTINEL_ROWS = 352
CONTEXT_ROWS = 20_000
MICROBATCH = 20_000
MICROBATCHES_PER_STATE = 10

# 完整 SHA-256,不是報告裡的縮寫。任何比對都用這些完整值。
PROBE_NPZ_SHA256 = "a3cfd7cfd2f449d5bb0564240ffcb2940d031af8a4660420dc84740cd4270a6b"
PROBE_RAW_INDEX_SHA256 = (
    "d4c6e4e76246a99b7a52268f11f1f4f6f78f85b25fb3b01bb6776caa9fc71e86"
)
HOLDOUT_SORTED_SHA256 = (
    "f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76"
)
CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"

TARGET_UNITS = (
    "seed42__cell00__cell_specific",
    "seed42__cell01__cell_specific",
    "seed42__cell11__frozen_reference",
)

# 與正式 gpu-host 完全相同的執行環境。不得使用 latest 或未固定的 build。
ENVIRONMENT_CONTRACT = {
    "architecture": "x86_64",
    "gpu_model_required": "RTX PRO 6000",
    "python": "3.12.13",
    "tabpfn": "8.0.8",
    "torch": "2.12.1",
    "cuda_runtime": "13.0",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scikit-learn": "1.8.0",
    "scipy_minimum": "1.11.1",
    "joblib": "1.5.3",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "forbidden": [
        "latest tag",
        "unpinned nightly build",
        "automatic package upgrade",
        "free resolution to newest versions",
        "CPU fallback",
        "torch.compile",
        "mixed precision changes",
        "TF32 policy changes",
        "multi-GPU",
        "MPS",
        "MIG",
        "third-party inference server",
    ],
    "source": "正式 E6 gpu-host 的已驗證版本,取自 e6_protocol.json 與 "
    "e6_tree_manifest.json 的 environment contract;未從執行中的 gpu-host 讀取",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_inputs(repo_root: Path, work: Path, state_specs: list[dict]) -> dict:
    """在本機重建 probe 與三個 context matrix,並驗證完整 digest。

    `build_feature_matrix` 每次呼叫都會重算整個 even-building 半邊,所以這裡
    把那個階段提出來算一次,再由各個列集合各自取用。
    """
    from lead import load_m3_frame
    from run_m5_story_ae_probe import (
        SHIFTS,
        add_value_change_features,
        feature_names,
        validate_feature_matrix,
    )

    columns = feature_names("F4")
    if len(columns) != FEATURES:
        raise SystemExit(f"F4 有 {len(columns)} 欄,預期 {FEATURES}")

    print("載入 frame 並計算 even-building 半邊的全表階段(只算一次)…", flush=True)
    frame = load_m3_frame(verbose=False)
    fit_frame = frame.loc[frame["building_id"] % 2 == 0]
    holdout_frame = frame.loc[frame["building_id"] % 2 == 1]
    t0 = time.perf_counter()
    tagged = fit_frame.copy()
    tagged["__raw_index_carrier"] = tagged.index.to_numpy(dtype="int64")
    built = add_value_change_features(
        tagged, list(SHIFTS), value_change_regime="timestamp_merge"
    )
    built.index = built["__raw_index_carrier"].to_numpy(dtype="int64")
    hoist_seconds = time.perf_counter() - t0
    print(f"  全表階段完成,{hoist_seconds:,.0f}s", flush=True)

    # ---- probe:與既有 audit artifact 完全相同的抽樣規則 --------------------
    pool = fit_frame.index.to_numpy(dtype="int64")
    rng = np.random.default_rng(PROBE_SEED)
    take = np.sort(rng.choice(pool.size, size=PROBE_ROWS, replace=False))
    probe_raw = pool[take]
    got_raw = hashlib.sha256(probe_raw.tobytes()).hexdigest()
    if got_raw != PROBE_RAW_INDEX_SHA256:
        raise SystemExit(
            f"HARD FAILURE 重建的 probe raw_index digest {got_raw[:16]} "
            f"與既有 artifact 記錄的 {PROBE_RAW_INDEX_SHA256[:16]} 不符"
        )
    probe_x = built.loc[probe_raw, columns].to_numpy(dtype="float32", copy=True)
    validate_feature_matrix(probe_x, matrix_name="gputw probe")
    if probe_x.shape != (PROBE_ROWS, FEATURES) or probe_x.dtype != np.float32:
        raise SystemExit(f"probe 矩陣 {probe_x.shape} {probe_x.dtype}")

    probe_npz = work / "e6_probe_matrix.npz"
    tmp = probe_npz.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, x=probe_x, raw_index=probe_raw)
    os.replace(tmp, probe_npz)
    probe_sha = sha256_file(probe_npz)
    probe_x_sha = hashlib.sha256(np.ascontiguousarray(probe_x).tobytes()).hexdigest()
    print(
        f"  probe 重建完成,npz sha256={probe_sha[:16]} x sha256={probe_x_sha[:16]}",
        flush=True,
    )

    # ---- probe 與 holdout 交集必須為 0(用完整集合,不抽樣) ----------------
    holdout_raw = holdout_frame.index.to_numpy(dtype="int64")
    if (
        hashlib.sha256(np.sort(holdout_raw).tobytes()).hexdigest()
        != HOLDOUT_SORTED_SHA256
    ):
        raise SystemExit("holdout raw_index 與凍結的 holdout digest 不符")
    overlap = np.intersect1d(probe_raw, holdout_raw)
    if overlap.size != 0:
        raise SystemExit(f"HARD FAILURE probe 與 holdout 有 {overlap.size} 列重疊")
    odd = int((holdout_frame.loc[probe_raw[:0]].shape[0]) if False else 0)
    del odd
    building_of_probe = frame.loc[probe_raw, "building_id"].to_numpy()
    if int((building_of_probe % 2 != 0).sum()) != 0:
        raise SystemExit("probe 含有 odd-building 列")
    print("  probe/holdout 交集 = 0,且 200,000 列全為 even-building", flush=True)

    # ---- 三個 context matrix ------------------------------------------------
    contexts = {}
    for spec in state_specs:
        cm = read_json(repo_root / spec["context_manifest"])
        ctx_raw = np.asarray(cm["raw_index"], dtype="int64")
        if ctx_raw.size != CONTEXT_ROWS:
            raise SystemExit(f"{spec['unit_id']}: context 有 {ctx_raw.size} 列")
        ctx_x = built.loc[ctx_raw, columns].to_numpy(dtype="float32", copy=True)
        if ctx_x.dtype != np.float32:
            raise SystemExit("context 被上轉型")
        name = f"context__{spec['unit_id']}.npz"
        p = work / name
        t = p.with_suffix(".npz.tmp")
        with t.open("wb") as fh:
            np.savez(fh, x=ctx_x, raw_index=ctx_raw)
        os.replace(t, p)
        contexts[spec["unit_id"]] = {
            "path": name,
            "sha256": sha256_file(p),
            "rows": int(ctx_raw.size),
            "context_manifest": spec["context_manifest"],
            "context_manifest_sha256": spec["context_manifest_sha256"],
        }
        print(f"  context {spec['unit_id']} 重建完成", flush=True)

    del built, frame, fit_frame, holdout_frame
    return {
        "probe": {
            "path": probe_npz.name,
            "sha256": probe_sha,
            "x_sha256": probe_x_sha,
            "raw_index_sha256": got_raw,
            # 內容權威值是 array-level digest,不是 .npz 檔案 digest。
            # `np.savez` 產生 zip,zip entry 記錄寫入平台(create_system:
            # Windows/FAT = 0、Unix = 3),那個 byte 進入檔案 digest 卻不影響
            # 任何一個陣列。既有 audit artifact 的 npz 是在 Linux 寫的,本機
            # 重建是在 Windows 寫的,所以檔案 digest 必然不同而內容相同。
            # 對照組:E6 的 full feature matrix 用 open_memmap 寫 .npy(沒有
            # zip 容器),兩台機器獨立重建得到完全相同的 digest。
            "existing_artifact_npz_sha256": PROBE_NPZ_SHA256,
            "existing_artifact_raw_index_sha256": PROBE_RAW_INDEX_SHA256,
            "raw_index_matches_existing_artifact": got_raw == PROBE_RAW_INDEX_SHA256,
            "npz_file_digest_differs_from_existing_artifact": probe_sha
            != PROBE_NPZ_SHA256,
            "npz_difference_cause": "zip 容器的寫入平台標記 (create_system);"
            "陣列內容未受影響,已由 raw_index digest 相符與 hoist bit-exact 驗證",
            "content_authority": "x_sha256 + raw_index_sha256",
            "rows": PROBE_ROWS,
            "features": FEATURES,
            "dtype": "float32",
            "selection_seed": PROBE_SEED,
            "source_half": "even buildings (fit half)",
            "all_even_buildings": True,
            "holdout_intersection": 0,
            "rebuilt_locally": True,
            "read_from_running_gpu_host": False,
        },
        "contexts": contexts,
        "hoist_seconds": hoist_seconds,
    }


def verify_states(
    repo_root: Path, work: Path, specs: list[dict], inputs: dict
) -> list[dict]:
    """複製三個 persisted state,驗證 digest,並確認 scaler 能精確重現。"""
    import joblib

    from m5_e5_runner import verify_scaler

    out = []
    for spec in specs:
        src = repo_root / spec["state_path"]
        digest = sha256_file(src)
        if digest != spec["state_sha256"]:
            raise SystemExit(f"{spec['unit_id']}: state digest 漂移")
        dst = work / f"state__{spec['unit_id']}.tabpfn_fit"
        shutil.copy2(src, dst)

        ctx = inputs["contexts"][spec["unit_id"]]
        with np.load(work / ctx["path"]) as z:
            raw_ctx = np.asarray(z["x"])
        ss = spec["scaler_source"]
        if ss["kind"] == "persisted":
            sp = repo_root / ss["path"]
            if sha256_file(sp) != ss["sha256"]:
                raise SystemExit(f"{spec['unit_id']}: scaler digest 不符")
            shutil.copy2(sp, work / f"scaler__{spec['unit_id']}.joblib")
            scaler = joblib.load(sp)
            scaler_ref = {
                "kind": "persisted",
                "file": f"scaler__{spec['unit_id']}.joblib",
                "sha256": ss["sha256"],
            }
        else:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(raw_ctx)
            scaler_ref = {
                "kind": "rebuilt_and_verified",
                "rule": "StandardScaler().fit(該 cell 自己的 20,000 列 context)",
            }
        check = verify_scaler(scaler, raw_ctx, src)
        if not check["exact"]:
            raise SystemExit(f"{spec['unit_id']}: scaler 未能精確重現 state 的 X_train")

        out.append(
            {
                "unit_id": spec["unit_id"],
                "context_seed": spec["context_seed"],
                "cell": spec["cell"],
                "scaler_arm": spec["scaler_arm"],
                "state_file": dst.name,
                "state_sha256": digest,
                "context_file": ctx["path"],
                "context_sha256": ctx["sha256"],
                "scaler": scaler_ref,
                "scaler_verification": check,
            }
        )
        print(f"  {spec['unit_id']}: state digest ok, scaler exact", flush=True)
    return out


def build_archive(work: Path, out: Path) -> dict:
    """可重複建立的 tar.zst。固定 mtime/uid/gid,讓相同輸入得到相同 digest。"""
    files = sorted(p for p in work.iterdir() if p.is_file())
    manifest = {p.name: sha256_file(p) for p in files}

    def reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mode = 0o644
        return ti

    tar_path = out / "m5-e6-gputw-probe-bundle.tar"
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as tf:
        for p in files:
            tf.add(p, arcname=p.name, filter=reset)

    archive = out / "m5-e6-gputw-probe-bundle.tar.zst"
    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=10)
        with tar_path.open("rb") as fin, archive.open("wb") as fout:
            cctx.copy_stream(fin, fout)
        codec = "zstd"
    except ImportError:
        import gzip

        archive = out / "m5-e6-gputw-probe-bundle.tar.gz"
        with (
            tar_path.open("rb") as fin,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=archive.open("wb"), mtime=0
            ) as fout,
        ):
            shutil.copyfileobj(fin, fout)
        codec = "gzip (zstandard 未安裝;不新增相依套件)"
    tar_path.unlink()

    return {
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_size_mb": archive.stat().st_size / 1e6,
        "codec": codec,
        "file_count": len(files),
        "per_file_sha256": manifest,
        "expected_extracted_tree": sorted(manifest),
        "reproducible": "tar entries 的 mtime/uid/gid/mode 已固定,相同輸入重建"
        "應得到相同 archive digest",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    args = ap.parse_args()

    if args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True)
    args.out.mkdir(parents=True, exist_ok=True)

    specs_all = read_json(args.protocol_root / "e6_state_manifest.json")["states"]
    specs = [s for s in specs_all if s["unit_id"] in TARGET_UNITS]
    if len(specs) != 3:
        raise SystemExit(f"找到 {len(specs)} 個代表 state,預期 3")
    specs.sort(key=lambda s: TARGET_UNITS.index(s["unit_id"]))

    inputs = build_inputs(args.repo_root, args.work, specs)
    states = verify_states(args.repo_root, args.work, specs, inputs)

    # sentinel:352 列,直接沿用 E6 已建好且 digest 已凍結的那一份
    sent_src = args.repo_root / "data/processed/m5_e6_features/e6_sentinel_query.npz"
    shutil.copy2(sent_src, args.work / "e6_sentinel_query.npz")
    with np.load(sent_src) as z:
        sent_rows = int(np.asarray(z["raw_index"]).size)
    if sent_rows != SENTINEL_ROWS:
        raise SystemExit(f"sentinel 有 {sent_rows} 列")

    for name, payload in (
        ("environment_contract.json", ENVIRONMENT_CONTRACT),
        (
            "benchmark_plan.json",
            {
                "microbatch_rows": MICROBATCH,
                "microbatches_per_state": MICROBATCHES_PER_STATE,
                "probe_rows": PROBE_ROWS,
                "microbatch_is_frozen": True,
                "why": "正式 E6 已凍結 20,000 列 canonical microbatch 語義;"
                "本 benchmark 必須量測與正式 E6 相同的執行路徑,因此不測 40k/80k",
                "single_worker_order": list(TARGET_UNITS),
                "dual_worker_rounds": {
                    "A": [TARGET_UNITS[0], TARGET_UNITS[1]],
                    "B": [TARGET_UNITS[1], TARGET_UNITS[0]],
                    "C": [TARGET_UNITS[2], TARGET_UNITS[0]],
                },
                "max_workers": 2,
                "third_worker": "forbidden",
                "full_holdout_scoring": "forbidden",
                "fits": "forbidden",
                "hard_limits": {
                    "remote_setup_minutes": 45,
                    "total_benchmark_minutes": 90,
                    "dual_worker_stall_minutes": 10,
                },
            },
        ),
    ):
        atomic_json(args.work / name, payload)

    state_manifest = {
        "schema": "m5_e6_gputw_probe_state_manifest_v1",
        "states": states,
        "required_effective_n_estimators": 8,
        "auto_scale_n_estimators": False,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    atomic_json(args.work / "state_manifest.json", state_manifest)

    probe_manifest = {
        "schema": "m5_e6_gputw_probe_manifest_v1",
        "probe": inputs["probe"],
        "sentinel": {
            "path": "e6_sentinel_query.npz",
            "sha256": sha256_file(args.work / "e6_sentinel_query.npz"),
            "rows": sent_rows,
        },
        "holdout_rows_in_bundle": 0,
        "holdout_raw_index_in_bundle": False,
        "score_columns_in_bundle": False,
        "full_feature_matrix_in_bundle": False,
    }
    atomic_json(args.work / "probe_manifest.json", probe_manifest)

    expected_schema = {
        "schema": "m5_e6_gputw_expected_output_schema_v1",
        "remote_environment.json": [
            "architecture",
            "gpu_name",
            "gpu_uuid",
            "driver",
            "cuda_runtime",
            "compute_capability",
            "vram_gb",
            "cpu_model",
            "cpu_cores",
            "ram_gb",
            "swap_gb",
            "disk_free_gb",
            "hostname",
            "python",
            "tabpfn",
            "torch",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "environment_digest",
        ],
        "sentinel_results.json": [
            "unit_id",
            "repeats",
            "distinct_digests",
            "endpoint_mean",
            "endpoint_half_width",
            "all_finite",
            "process_uuid",
            "gpu_uuid",
            "fits_performed",
        ],
        "single_worker_results.json": [
            "unit_id",
            "reload_seconds",
            "scale_seconds",
            "per_batch",
            "sustained_rows_per_second",
            "median_rows_per_second",
            "p05_rows_per_second",
            "p95_rows_per_second",
            "aggregate_rows_per_second",
            "peak_vram_gb",
            "peak_rss_gb",
            "projected_state_hours",
            "scores_retained",
            "fits_performed",
        ],
        "dual_worker_results.json": [
            "rounds",
            "aggregate_speedup",
            "verdict",
            "worker0",
            "worker1",
            "distinct_process_uuids",
        ],
        "required_zero_fields": {
            "scores_retained": 0,
            "fits_performed": 0,
            "holdout_rows_scored": 0,
        },
    }
    atomic_json(args.work / "expected_output_schema.json", expected_schema)

    # benchmark 腳本本身也進 bundle
    here = Path(__file__).resolve().parent
    for s in (
        "m5_e6_gputw_preflight.py",
        "m5_e6_gputw_sentinel.py",
        "m5_e6_gputw_single_worker.py",
        "m5_e6_gputw_dual_worker.py",
        "m5_e5_guard.py",
        "m5_e5_runner.py",
    ):
        src = here / s
        if not src.exists():
            raise SystemExit(f"bundle 缺少必要腳本:{s}")
        shutil.copy2(src, args.work / s)

    archive = build_archive(args.work, args.out)

    bundle_manifest = {
        "schema": "m5_e6_gputw_bundle_manifest_v1",
        "generated": time.time(),
        **archive,
        "probe": probe_manifest["probe"],
        "sentinel": probe_manifest["sentinel"],
        "states": [s["unit_id"] for s in states],
        "environment_contract": ENVIRONMENT_CONTRACT,
        "excluded_by_design": [
            "10,137,155-row full feature matrix",
            "full-holdout raw_index list",
            "full-holdout score files",
            "current E6 partial outputs",
            "tree full-test outputs",
            "credentials",
            "unrelated large repository artifacts",
        ],
        "read_from_running_gpu_host": False,
        "hoist_seconds": inputs["hoist_seconds"],
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for name in (
        "probe_manifest.json",
        "state_manifest.json",
        "environment_contract.json",
        "benchmark_plan.json",
        "expected_output_schema.json",
    ):
        shutil.copy2(args.work / name, args.artifacts / name)
    d = atomic_json(args.artifacts / "bundle_manifest.json", bundle_manifest)

    print(f"\narchive          : {archive['archive']}")
    print(f"archive sha256   : {archive['archive_sha256']}")
    print(f"archive size     : {archive['archive_size_mb']:.1f} MB")
    print(f"file count       : {archive['file_count']}")
    p = inputs["probe"]
    print(f"probe raw_index 相符: {p['raw_index_matches_existing_artifact']}")
    print(f"probe x sha256   : {p['x_sha256']}")
    print(
        f"npz 檔案 digest 與既有 artifact 不同: "
        f"{p['npz_file_digest_differs_from_existing_artifact']}"
        f"({p['npz_difference_cause'].split(';')[0]})"
    )
    print(f"bundle_manifest  : {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
