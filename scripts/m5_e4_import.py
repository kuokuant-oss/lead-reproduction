"""Validate the E4 results, then import them into the canonical root.

Nothing the runner wrote about its own results is trusted. Every endpoint is
recomputed here from the raw per-repeat score vectors, and the effective
ensemble size is re-read out of each persisted state rather than taken from
`FIT_COMPLETE.json`. A runner that recorded the wrong number therefore fails
validation even when its files are internally consistent and their digests
match.

Validation runs entirely against the staged copy. The canonical root is not
touched until every check has passed, and the import is one directory-level
`os.replace` per unit, so a canonical unit is either the old one or the new one
and never a mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e4_endpoints import endpoints  # noqa: E402

MANIFEST = "e4_file_manifest.sha256"
REQUIRED_EFFECTIVE_N_ESTIMATORS = 8
REPEATS = 8

# Written locally after the results arrive, so they cannot be in the remote
# manifest. Nothing else may be missing from it.
LOCALLY_DERIVED = frozenset(
    {
        MANIFEST,
        "e4_protocol.json",
        "e4_fit_manifest.json",
        "e4_repeat_manifest.json",
        "e4_input_manifest.json",
        "e4_summary.json",
        "e4_factorial.json",
        "e4_clustered.json",
        "e4_decision.json",
    }
)
REQUIRED_UNIT_FILES = (
    "fit_start.json",
    "FIT_COMPLETE.json",
    "model.tabpfn_fit",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(failures: list[str], ok: bool, message: str) -> None:
    if not ok:
        failures.append(message)


def state_ensemble(state_path: Path) -> dict:
    """Re-read the realised ensemble size out of the persisted state.

    `save_fit_state` stores the constructor arguments and every attribute ending
    in `_`, so both the requested and the effective value are inside the archive
    and can be read without a GPU and without trusting the runner.
    """
    import joblib

    with zipfile.ZipFile(state_path) as z:
        init = json.loads(z.read("init_params.json"))
        attrs = joblib.load(io.BytesIO(z.read("fitted_attrs.joblib")))
    configs = attrs.get("ensemble_configs_")
    return {
        "requested_n_estimators": init.get("n_estimators"),
        "auto_scale_n_estimators": init.get("auto_scale_n_estimators"),
        "effective_n_estimators_": attrs.get("n_estimators_"),
        "len_ensemble_configs_": len(configs) if configs is not None else None,
        "random_state": init.get("random_state"),
        "fit_mode": init.get("fit_mode"),
        "n_features_in_": attrs.get("n_features_in_"),
        "n_train_samples_": attrs.get("n_train_samples_"),
    }


def validate_manifest(staged: Path, failures: list[str]) -> dict[str, str]:
    path = staged / MANIFEST
    if not path.exists():
        failures.append(f"missing {MANIFEST}")
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        entries[rel.removeprefix("./")] = digest
    for rel, want in entries.items():
        target = staged / rel
        if not target.exists():
            failures.append(f"manifest lists a missing file: {rel}")
        elif sha256_file(target) != want:
            failures.append(f"digest mismatch after transfer: {rel}")
    on_disk = {
        p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()
    }
    for rel in sorted(on_disk - set(entries) - LOCALLY_DERIVED):
        failures.append(f"file present but absent from the remote manifest: {rel}")
    return entries


def validate_unit(staged: Path, spec: dict, proto: dict, failures: list[str]) -> dict:
    uid = spec["unit_id"]
    root = staged / uid
    for name in REQUIRED_UNIT_FILES:
        check(failures, (root / name).exists(), f"{uid}: missing {name}")
    check(
        failures,
        not (root / "INTERRUPTED_INCOMPLETE.json").exists(),
        f"{uid}: INTERRUPTED_INCOMPLETE marker present",
    )
    if any(uid in f for f in failures):
        return {}

    complete = read_json(root / "FIT_COMPLETE.json")
    start = read_json(root / "fit_start.json")
    state_path = root / "model.tabpfn_fit"
    state_sha = sha256_file(state_path)

    check(
        failures,
        state_sha == complete["state_sha256"],
        f"{uid}: state digest differs from the recorded fit",
    )
    check(failures, complete["fits"] == 1, f"{uid}: not exactly one fit")
    check(
        failures,
        complete["status"] == "COMPLETE",
        f"{uid}: status is {complete.get('status')}",
    )
    for field, want in (
        ("context_seed", spec["context_seed"]),
        ("cell", spec["cell"]),
        ("scaler_arm", spec["scaler_arm"]),
    ):
        check(
            failures,
            complete[field] == want,
            f"{uid}: {field} is {complete[field]}, manifest says {want}",
        )
    check(
        failures,
        start["manifest_sha256"] == spec["context_manifest_sha256"],
        f"{uid}: context manifest digest does not match the frozen manifest",
    )
    check(
        failures,
        start["query_sha256"] == proto["query"]["raw_index_sha256"],
        f"{uid}: query digest does not match the frozen query",
    )
    check(
        failures,
        start["process_uuid"] == complete["process_uuid"],
        f"{uid}: fit_start and FIT_COMPLETE come from different processes",
    )

    # The ensemble contract, re-read from the state rather than believed.
    observed = state_ensemble(state_path)
    contract = proto["ensemble"]
    check(
        failures,
        observed["effective_n_estimators_"] == REQUIRED_EFFECTIVE_N_ESTIMATORS,
        f"{uid}: effective n_estimators_ is {observed['effective_n_estimators_']}",
    )
    check(
        failures,
        observed["len_ensemble_configs_"] == REQUIRED_EFFECTIVE_N_ESTIMATORS,
        f"{uid}: ensemble_configs_ has {observed['len_ensemble_configs_']} members",
    )
    check(
        failures,
        observed["auto_scale_n_estimators"] is contract["auto_scale_n_estimators"],
        f"{uid}: auto_scale_n_estimators is {observed['auto_scale_n_estimators']}",
    )
    check(
        failures,
        observed["requested_n_estimators"] == contract["requested_n_estimators"],
        f"{uid}: requested n_estimators is {observed['requested_n_estimators']}",
    )
    check(
        failures,
        observed["random_state"] == proto["inherited"]["model_seed"],
        f"{uid}: model seed is {observed['random_state']}",
    )
    recorded = complete.get("ensemble", {})
    check(
        failures,
        recorded.get("effective_n_estimators_") == observed["effective_n_estimators_"],
        f"{uid}: the runner's recorded effective value disagrees with the state",
    )

    # Endpoints recomputed from the raw score vectors, not read back.
    paths = sorted((root / "repeats").glob("repeat_*.json"))
    check(
        failures, len(paths) == REPEATS, f"{uid}: {len(paths)} repeats, want {REPEATS}"
    )
    scores, digests, drift = [], set(), []
    for p in paths:
        rec = read_json(p)
        score = np.asarray(rec["score"], dtype="float64")
        if hashlib.sha256(score.tobytes()).hexdigest() != rec["score_sha256"]:
            drift.append(f"{p.name}:digest")
            continue
        if rec["state_sha256"] != state_sha:
            drift.append(f"{p.name}:state")
        if rec["process_uuid"] != complete["process_uuid"]:
            drift.append(f"{p.name}:process")
        scores.append(score)
        digests.add(rec["score_sha256"])
        recomputed = endpoints(score, MET, ANOM)
        for key, was in rec["endpoints"].items():
            got = recomputed[key]
            if not (
                np.isclose(got, was, rtol=0, atol=1e-12)
                or (not np.isfinite(got) and not np.isfinite(was))
            ):
                drift.append(f"{p.name}:{key}({was}->{got})")
    check(failures, not drift, f"{uid}: repeat records do not reproduce: {drift[:3]}")

    return {
        "unit_id": uid,
        "context_seed": spec["context_seed"],
        "cell": spec["cell"],
        "scaler_arm": spec["scaler_arm"],
        "state_sha256": state_sha,
        "process_uuid": complete["process_uuid"],
        "repeats": len(scores),
        "distinct_score_digests": len(digests),
        "fit_seconds": complete.get("fit_seconds"),
        "peak_gpu_bytes": complete.get("peak_gpu_bytes"),
        "ensemble": observed,
        "scores": scores,
    }


MET: np.ndarray = np.empty(0, dtype="int8")
ANOM: np.ndarray = np.empty(0, dtype="int8")


def load_query(
    proto: dict, repo_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpath = repo_root / proto["query"]["manifest"]
    npz = qpath.with_name("queries.npz")
    if sha256_file(npz) != proto["query"]["npz_sha256"]:
        raise SystemExit("the query npz does not match the frozen digest")
    with np.load(npz) as z:
        return (
            np.asarray(z["raw_index"], dtype="int64"),
            np.asarray(z["meter"], dtype="int8"),
            np.asarray(z["anomaly"], dtype="int8"),
        )


def main() -> int:
    global MET, ANOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []
    proto_path = args.canonical / "e4_protocol.json"
    if not proto_path.exists():
        print("canonical protocol artifact is missing")
        return 1
    proto = read_json(proto_path)["protocol"]
    specs = read_json(args.canonical / "e4_fit_manifest.json")["fits"]
    _, MET, ANOM = load_query(proto, args.repo_root)

    entries = validate_manifest(args.staged, failures)

    order = [u["unit_id"] for u in proto["schedule"]["realised_24_unit_order"]]
    staged_units = sorted(p.name for p in args.staged.glob("seed*") if p.is_dir())
    check(
        failures,
        staged_units == sorted(order),
        f"staged units do not match the frozen 24: {set(order) ^ set(staged_units)}",
    )

    units = []
    if not failures:
        units = [validate_unit(args.staged, s, proto, failures) for s in specs]
        units = [u for u in units if u]

    if not failures:
        check(failures, len(units) == 24, f"{len(units)} units validated, want 24")
        check(
            failures,
            sum(u["repeats"] for u in units) == 192,
            f"{sum(u['repeats'] for u in units)} repeats, want 192",
        )
        check(
            failures,
            len({u["process_uuid"] for u in units}) == 24,
            "two units share a process UUID",
        )
        # Cell-11 arm pairs may share a state; nothing else may.
        by_state: dict[str, list[str]] = {}
        for u in units:
            by_state.setdefault(u["state_sha256"], []).append(u["unit_id"])
        for digest, ids in by_state.items():
            if len(ids) == 1:
                continue
            seeds = {i.split("__")[0] for i in ids}
            cells = {i.split("__")[1] for i in ids}
            check(
                failures,
                len(ids) == 2 and cells == {"cell11"} and len(seeds) == 1,
                f"unexpected state collision {digest[:8]}: {ids}",
            )
        check(
            failures,
            len(by_state) >= 21,
            f"{len(by_state)} distinct states, want at least 21",
        )

    print(f"files verified : {len(entries)}")
    print(f"units validated: {len(units)}")
    for u in units:
        print(
            f"  {u['unit_id']:<40} repeats={u['repeats']} "
            f"distinct={u['distinct_score_digests']} "
            f"n_est={u['ensemble']['effective_n_estimators_']} "
            f"state={u['state_sha256'][:8]}"
        )
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nVALIDATION PASSED")

    if not args.apply:
        print("dry run; pass --apply to import")
        return 0

    args.canonical.mkdir(parents=True, exist_ok=True)
    for uid in order:
        src = args.staged / uid
        dst = args.canonical / uid
        staging = args.canonical / f".incoming_{uid}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
        if dst.exists():
            shutil.rmtree(dst)
        staging.replace(dst)
        print(f"imported {uid}")

    # Everything else the manifest attests to, which is the run's own logs.
    # Importing only the unit directories left the canonical root unable to
    # satisfy its own manifest: the logs were referenced but absent, so
    # self-validation failed on files that were never a science defect.
    for rel in sorted(entries):
        if rel.split("/", 1)[0] in set(order):
            continue
        src = args.staged / rel
        dst = args.canonical / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".incoming_{dst.name}")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        print(f"imported {rel}")

    shutil.copy2(args.staged / MANIFEST, args.canonical / MANIFEST)
    print("IMPORT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
