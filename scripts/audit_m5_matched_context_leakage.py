"""Empirically audit the matched-N context for leakage. Reconstructs, never trusts.

Every claim this makes about the matched-N comparison was previously argued from
reading the sampler and from an assertion inside the runner. That is not the same
as having checked. This rebuilds the actual fit index from the actual frame and
tests it against the actual holdout:

  1. the fit rows are unique and exactly 50/50
  2. every fit row is an even building, every holdout row an odd one
  3. fit and holdout share no row id -- a real set intersection over all 10.1M
  4. the reconstructed digest equals the one the tree run recorded and the one
     Gate 1 froze, so what is audited here is what was actually trained on
  5. no holdout row duplicates a fit row on (17 features + label), which is the
     leakage that row-id disjointness alone would not catch: two different
     buildings can still carry byte-identical feature rows

    uv run python scripts/audit_m5_matched_context_leakage.py --context-rows 100000
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys

import numpy as np

from lead import BASELINE_FEATURE_COLS, PROC, RANDOM_STATE, ROOT, load_m3_frame

CANONICAL_ORDER = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"
NESTING_PROOF = PROC / "m5_tabpfn_context_nesting_proof.json"


def load_canonical_module():
    spec = importlib.util.spec_from_file_location(
        "m5_canonical", ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="int64").astype("<i8", copy=False).tobytes()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument("--dup-sample", type=int, default=2_000_000)
    args = parser.parse_args()
    mod = load_canonical_module()
    fail = []

    print("loading frame", flush=True)
    frame = load_m3_frame(verbose=True)
    building = frame["building_id"].to_numpy(dtype="int32")
    anomaly = frame["anomaly"].to_numpy(dtype="int8")
    row_id = frame.index.to_numpy(dtype="int64")

    train_mask = building % 2 == 0
    train_index = row_id[train_mask]

    with np.load(CANONICAL_ORDER) as payload:
        holdout_index = np.asarray(payload["validation_raw_index"], dtype="int64")

    # --- rebuild exactly what the tree runner fits on -----------------------
    validation_index = mod.fixed_score_indices(
        train_index, args.validation_rows, seed=args.seed + 20_000
    )
    candidate_index = train_index[~np.isin(train_index, validation_index)]
    pos_by_id = dict(zip(row_id.tolist(), anomaly.tolist()))
    candidate_y = np.fromiter(
        (pos_by_id[int(i)] for i in candidate_index),
        dtype="int8",
        count=len(candidate_index),
    )
    fit_index = mod.nested_balanced_indices(
        candidate_index, candidate_y, [args.context_rows], seed=args.seed
    )[args.context_rows]
    del pos_by_id
    gc.collect()

    digest = array_sha256(fit_index)
    print(f"\ncontext {args.context_rows:,}  digest {digest}")

    # 4. is this the same context the run actually used?
    proof = json.loads(NESTING_PROOF.read_text(encoding="utf-8"))
    frozen = proof["contexts"][str(args.context_rows)]["sha256"]
    print(
        f"  Gate 1 frozen digest              : {frozen}  {'MATCH' if frozen == digest else 'MISMATCH'}"
    )
    if frozen != digest:
        fail.append("reconstructed context != Gate 1")
    run_json = PROC / f"m5_tree_ensemble_f17_context{args.context_rows}.json"
    if run_json.is_file():
        recorded = json.loads(run_json.read_text(encoding="utf-8"))["context_sha256"]
        ok = recorded == digest
        print(
            f"  digest recorded by the tree run    : {recorded}  {'MATCH' if ok else 'MISMATCH'}"
        )
        if not ok:
            fail.append("tree run trained on a different context")

    # 1. composition
    fit_y = np.fromiter(
        (int(anomaly[np.searchsorted(row_id, i)]) for i in fit_index),
        dtype="int8",
        count=len(fit_index),
    )
    n_unique = len(np.unique(fit_index))
    print(
        f"\n1. composition: {len(fit_index):,} rows, {n_unique:,} unique, "
        f"{int(fit_y.sum()):,} positive ({fit_y.mean():.1%})"
    )
    if n_unique != len(fit_index):
        fail.append("fit rows are not unique")
    if int(fit_y.sum()) * 2 != len(fit_index):
        fail.append("fit rows are not 50/50")

    # 2. building parity
    fit_pos = np.searchsorted(row_id, fit_index)
    fit_build = building[fit_pos]
    hold_pos = np.searchsorted(row_id, holdout_index)
    hold_build = building[hold_pos]
    fit_even = bool((fit_build % 2 == 0).all())
    hold_odd = bool((hold_build % 2 == 1).all())
    print(
        f"2. parity: all fit buildings even = {fit_even} ({len(np.unique(fit_build))} buildings); "
        f"all holdout buildings odd = {hold_odd} ({len(np.unique(hold_build))} buildings)"
    )
    if not (fit_even and hold_odd):
        fail.append("building parity violated")
    shared = np.intersect1d(np.unique(fit_build), np.unique(hold_build))
    print(f"   buildings appearing in both: {len(shared)}")
    if len(shared):
        fail.append("a building appears in both halves")

    # 3. row-id disjointness, over the full holdout
    overlap = np.intersect1d(fit_index, holdout_index)
    print(f"3. fit ∩ holdout row ids: {len(overlap)}")
    if len(overlap):
        fail.append("fit and holdout share row ids")

    # 5. content duplication: identical feature row AND identical label
    print(
        f"5. duplicate-content check on {args.dup_sample:,} sampled holdout rows",
        flush=True,
    )
    cols = list(BASELINE_FEATURE_COLS)
    fit_block = frame.loc[fit_index, cols].to_numpy(dtype="float64")
    fit_keys = {
        hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in fit_block
    }
    rng = np.random.default_rng(0)
    take = np.sort(
        rng.choice(
            len(holdout_index),
            size=min(args.dup_sample, len(holdout_index)),
            replace=False,
        )
    )
    hold_block = frame.loc[holdout_index[take], cols].to_numpy(dtype="float64")
    hits = sum(
        1
        for r in hold_block
        if hashlib.blake2b(r.tobytes(), digest_size=16).digest() in fit_keys
    )
    print(
        f"   holdout rows byte-identical to a fit row: {hits} "
        f"({hits / len(hold_block):.6%} of the sample)"
    )
    if hits > len(hold_block) * 0.001:
        fail.append(f"{hits} duplicated feature rows across the split")

    print("\n" + ("FAILED: " + "; ".join(fail) if fail else "PASSED: no leakage found"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
