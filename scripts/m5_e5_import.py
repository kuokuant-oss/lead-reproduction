"""Validate the E5 results, then import them into the canonical root.

Nothing the runner wrote about its own results is trusted. Every endpoint is
recomputed here from the raw per-repeat score vectors, and the effective
ensemble size is re-read out of each persisted state rather than taken from
`UNIT_COMPLETE.json`. A runner that recorded the wrong number therefore fails
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

MANIFEST = "e5_file_manifest.sha256"
REQUIRED_EFFECTIVE_N_ESTIMATORS = 8
REPEATS = 8
QUERY_ROWS = 192

# Written locally after the results arrive, so they cannot be in the remote
# manifest. Nothing else may be missing from it.
LOCALLY_DERIVED = frozenset(
    {
        MANIFEST,
        "e5_query_audit.json",
        "e5_tree_execution_override.json",
        "e5_protocol.json",
        "e5_state_manifest.json",
        "e5_repeat_manifest.json",
        "e5_input_manifest.json",
        "e5_summary.json",
        "e5_factorial.json",
        "e5_clustered.json",
        "e5_decision.json",
    }
)
# E5 produces no state: it reloads E4's. A unit directory therefore holds only
# its reload record, its completion marker and its repeats.
REQUIRED_UNIT_FILES = (
    "reload_start.json",
    "UNIT_COMPLETE.json",
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


def rederive_tree(spec: dict, repo_root: Path, feature_npz: Path) -> np.ndarray:
    """The tree score, recomputed from the persisted ensemble and shared input.

    This is the anchor the tree half rests on. It needs no frame load: the
    192-row feature matrix is the artifact both hosts agreed on, and the
    ensemble and its scaler are persisted alongside E4.
    """
    import joblib

    from run_m5_story_ae_probe import load_tree_runner

    d = (
        repo_root
        / "data/processed/m5_hotwater_label_factorial/recovery/states/trees"
        / f"seed{spec['context_seed']}"
        / spec["cell_dir"]
        / spec["scaler_arm"]
    )
    saved = joblib.load(d / "tree_ensemble.joblib")
    scaler = joblib.load(d / "scaler.joblib")
    with np.load(feature_npz) as z:
        raw = np.asarray(z["q"])
    runner = load_tree_runner()
    x = scaler.transform(raw).astype("float32")
    stacked = [
        runner.predict_probability(n, saved["models"][n], x)
        for n in saved["model_order"]
    ]
    return np.asarray(np.mean(stacked, axis=0).astype("float32"), dtype="float64")


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
        # trees/ was scored on the laptop under the override, so it cannot be in
        # the remote manifest. It is not exempt from checking -- it carries its
        # own manifest with per-file digests and a 24/24 bit-exact gate, and
        # validate_trees below refuses it if any of that is missing.
        if rel.startswith("trees/"):
            continue
        failures.append(f"file present but absent from the remote manifest: {rel}")
    return entries


REPO_ROOT: Path = Path(".")
FEATURE_NPZ: Path = Path(".")
SPEC_BY_UID: dict = {}


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

    complete = read_json(root / "UNIT_COMPLETE.json")
    start = read_json(root / "reload_start.json")
    # The state lives in the E4 result root, named by the frozen state manifest.
    state_path = REPO_ROOT / spec["state_path"]
    if not state_path.exists():
        failures.append(f"{uid}: the E4 state is missing at {spec['state_path']}")
        return {}
    state_sha = sha256_file(state_path)
    check(
        failures,
        state_sha == spec["state_sha256"],
        f"{uid}: the E4 state digest does not match the frozen manifest",
    )

    check(
        failures,
        state_sha == complete["state_sha256"],
        f"{uid}: state digest differs from the recorded fit",
    )
    check(
        failures,
        complete["fits_performed"] == 0,
        f"{uid}: fits_performed is {complete.get('fits_performed')}, must be 0",
    )
    check(failures, complete["reloads"] == 1, f"{uid}: not exactly one reload")
    check(
        failures,
        complete["scaler_verification"].get("exact") is True,
        f"{uid}: the scaler was not verified exactly against the E4 state",
    )
    check(
        failures,
        complete["score_vector_length"] == 192,
        f"{uid}: score vector length is {complete.get('score_vector_length')}",
    )
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
        start["state_sha256"] == spec["state_sha256"],
        f"{uid}: reload_start state digest does not match the manifest",
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
        observed["random_state"] == proto["inherited_from_e4"]["model_seed"],
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
        if score.size != QUERY_ROWS:
            drift.append(f"{p.name}:length={score.size}")
            continue
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


QUERY_INDEX: np.ndarray = np.empty(0, dtype="int64")
MET: np.ndarray = np.empty(0, dtype="int8")
ANOM: np.ndarray = np.empty(0, dtype="int8")


def load_query(
    proto: dict, repo_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = repo_root / proto["query"]["path"] / "queries.npz"
    if sha256_file(npz) != proto["query"]["queries_npz_sha256"]:
        raise SystemExit("the query npz does not match the frozen digest")
    with np.load(npz, allow_pickle=True) as z:
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
    ap.add_argument(
        "--feature-npz",
        type=Path,
        required=True,
        help="the shared 192x137 artifact both hosts scored",
    )
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []
    proto_path = args.canonical / "e5_protocol.json"
    if not proto_path.exists():
        print("canonical protocol artifact is missing")
        return 1
    proto = read_json(proto_path)["protocol"]
    specs = read_json(args.canonical / "e5_state_manifest.json")["states"]
    QI, MET, ANOM = load_query(proto, args.repo_root)
    globals()["REPO_ROOT"] = args.repo_root
    globals()["FEATURE_NPZ"] = args.feature_npz
    globals()["SPEC_BY_UID"] = {sp["unit_id"]: sp for sp in specs}
    globals()["QUERY_INDEX"] = QI

    entries = validate_manifest(args.staged, failures)

    order = [u["unit_id"] for u in proto["execution_order"]["units"]]
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

    # The tree half was scored on the laptop under the override, so it is not in
    # the remote manifest. It gets its own checks rather than a free pass.
    tdir = args.staged / "trees"
    tman = tdir / "e5_tree_manifest.json"
    if not tman.exists():
        failures.append("missing trees/e5_tree_manifest.json")
    else:
        tm = read_json(tman)
        ovr_path = args.canonical / "e5_tree_execution_override.json"
        ovr = read_json(ovr_path) if ovr_path.exists() else {}
        check(failures, tm["units"] == 24, f"tree manifest has {tm['units']} units")
        check(
            failures,
            tm["gate_units_bit_exact"] == 24,
            f"{tm.get('gate_units_bit_exact')}/24 trees passed the bit-exact gate",
        )
        check(failures, tm["refit"] is False, "a tree refit was recorded")
        check(
            failures,
            tm["artificial_replicates"] is False,
            "the tree manifest declares artificial replicates",
        )
        check(
            failures,
            tm["execution_host"] == "original laptop environment",
            f"tree execution host is {tm.get('execution_host')}",
        )
        check(
            failures,
            tm["base_192_row_feature_sha256"]
            == ovr.get("shared_input_requirement", {}).get("sha256"),
            "the tree half used a different 192-row feature artifact",
        )
        check(
            failures,
            sha256_file(ovr_path) == tm["override_sha256"]
            if ovr_path.exists()
            else False,
            "the tree manifest cites a different override artifact",
        )
        for rec in tm["records"]:
            uid = rec["unit_id"]
            npz = tdir / f"{uid}.npz"
            check(failures, npz.exists(), f"{uid}: missing tree score vector")
            if not npz.exists():
                continue
            check(
                failures,
                sha256_file(npz) == rec["npz_sha256"],
                f"{uid}: tree npz digest drifted",
            )
            # Recompute rather than believe. A tampered vector plus a
            # regenerated tree manifest is internally consistent, so the only
            # unforgeable check is re-deriving the score from the persisted
            # ensemble and the shared feature artifact.
            want = rederive_tree(SPEC_BY_UID[uid], REPO_ROOT, FEATURE_NPZ)
            with np.load(npz) as z:
                got = np.asarray(z["score"], dtype="float64")
            check(
                failures,
                got.shape == want.shape and np.array_equal(got, want),
                f"{uid}: the tree score does not reproduce from the persisted "
                "ensemble and the shared feature artifact",
            )
            check(
                failures,
                rec["replicates"] == 1,
                f"{uid}: tree has {rec['replicates']} replicates, must be 1",
            )
            check(
                failures,
                rec["max_abs_diff"] == 0.0 and rec["exact_rows"] == 352,
                f"{uid}: tree is not bit-exact against E4's comparator",
            )
            with np.load(npz) as z:
                sc = np.asarray(z["score"], dtype="float64")
                ri = np.asarray(z["raw_index"], dtype="int64")
            check(
                failures,
                sc.size == QUERY_ROWS and np.all(np.isfinite(sc)),
                f"{uid}: tree score is not 192 finite values",
            )
            check(
                failures,
                np.array_equal(ri, QUERY_INDEX),
                f"{uid}: tree rows are not the frozen query rows in order",
            )
        check(
            failures,
            {r["unit_id"] for r in tm["records"]} == set(order),
            "the tree half does not cover the same 24 units",
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

    # The tree half, which is validated above but is not in the remote manifest
    # because the laptop produced it. Omitting it would leave the canonical root
    # unable to satisfy its own tree manifest.
    tsrc = args.staged / "trees"
    if tsrc.is_dir():
        tdst = args.canonical / "trees"
        staging = args.canonical / ".incoming_trees"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(tsrc, staging)
        if tdst.exists():
            shutil.rmtree(tdst)
        staging.replace(tdst)
        print(f"imported trees/ ({len(list(tdst.glob('*.npz')))} score vectors)")

    shutil.copy2(args.staged / MANIFEST, args.canonical / MANIFEST)
    print("IMPORT COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
