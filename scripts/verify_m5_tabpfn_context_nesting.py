"""Gate 1 of the context curve: prove every context is a prefix of the 100k one.

The whole comparison rests on one property. ``canonical_contract`` builds the
context by calling ``nested_balanced_indices`` with a *single* budget, and that
function shuffles the full per-class arrays under a fixed seed before taking the
first ``budget // 2`` of each. The shuffle therefore does not depend on the
budget, so separate single-budget calls come out nested -- the 5k context is
literally the first 5,000 rows of the 100k context, same rows, same order, same
50/50 interleaving.

If that ever stopped holding, the five points on the curve would differ by both
context size *and* which rows were sampled, and every difference between them
would be uninterpretable. Nothing downstream can detect it: each context would
still fit, score and merge cleanly.

So it is checked here, on the real candidate rows, against the recorded
``context_sha256`` of the frozen 100k run -- not on a synthetic stand-in, and not
by re-reading the same reasoning.

Run before trusting any fit:

    uv run python scripts/verify_m5_tabpfn_context_nesting.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lead import PROC, RANDOM_STATE, ROOT, load_m3_frame, write_json_with_provenance

CANONICAL = ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"

# The frozen 100k context, recorded by the fit that produced both published
# TabPFN lines. Reproducing this digest is what proves the reconstruction below
# is the real context and not merely a self-consistent one.
FROZEN_100K_SHA256 = "e9ffe0cffd2e0cf304d213a02e68f2d7ef092172efc0343e680f982a2d688cbe"
DEFAULT_CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)


def load_canonical_module():
    spec = importlib.util.spec_from_file_location("m5_canonical_nesting", CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="int64").astype("<i8", copy=False).tobytes()
    ).hexdigest()


def build_candidates(frame: Any, mod: Any, *, seed: int, validation_rows: int):
    """Reproduce canonical_contract's candidate pool exactly."""
    train_mask = (frame["building_id"] % 2 == 0).to_numpy()
    train_index = frame.index[train_mask].to_numpy(dtype="int64")
    validation_index = mod.fixed_score_indices(
        train_index, validation_rows, seed=seed + 20_000
    )
    candidate_index = train_index[~np.isin(train_index, validation_index)]
    candidate_y = frame.loc[candidate_index, "anomaly"].to_numpy(dtype="int8")
    return candidate_index, candidate_y, validation_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=int, nargs="*", default=list(DEFAULT_CONTEXTS)
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    parser.add_argument(
        "--out", type=Path, default=PROC / "m5_tabpfn_context_nesting_proof.json"
    )
    args = parser.parse_args(argv)

    contexts = sorted(set(int(c) for c in args.contexts))
    if 100_000 not in contexts:
        raise SystemExit("100000 must be included: it is the reference context")

    mod = load_canonical_module()
    frame = load_m3_frame(verbose=True)
    candidate_index, candidate_y, validation_index = build_candidates(
        frame, mod, seed=args.seed, validation_rows=args.validation_rows
    )
    print(
        f"candidate pool: {len(candidate_index):,} rows "
        f"({int(candidate_y.sum()):,} positive), "
        f"{len(validation_index):,} validation rows held out"
    )

    built = {
        context: mod.nested_balanced_indices(
            candidate_index, candidate_y, [context], seed=args.seed
        )[context]
        for context in contexts
    }
    # Nesting runs smaller-inside-larger, so the reference is the largest size
    # asked for, not 100k. The tree arm of the curve goes well past 100k (up to
    # the 1,353,634 balanced ceiling), and against a 100k reference every one of
    # those would "fail" for the trivial reason that it is longer.
    largest = contexts[-1]
    reference = built[largest]
    label_of = dict(zip(candidate_index.tolist(), candidate_y.tolist()))

    failures: list[str] = []

    frozen_sha = array_sha256(built[100_000])
    if frozen_sha != FROZEN_100K_SHA256:
        failures.append(
            "reconstructed 100k context does not match the frozen run: "
            f"{frozen_sha} != {FROZEN_100K_SHA256}"
        )
    print(
        f"\n100k context digest {'matches' if not failures else 'DIFFERS FROM'} "
        f"the frozen run   (nesting reference: {largest:,})"
    )

    print(f"\ncontext          prefix_of_{largest:<9,}  positive  disjoint  sha256")
    records: dict[str, Any] = {}
    for context in contexts:
        index = built[context]
        is_prefix = np.array_equal(index, reference[:context])
        positives = int(sum(label_of[int(row)] for row in index))
        balanced = positives * 2 == context
        unique = len(np.unique(index)) == len(index)
        disjoint = not np.intersect1d(index, validation_index).size
        digest = array_sha256(index)

        if not is_prefix:
            failures.append(
                f"context {context} is not a prefix of the {largest} context"
            )
        if not balanced:
            failures.append(f"context {context} is not 50/50 balanced")
        if not unique:
            failures.append(f"context {context} contains duplicate rows")
        if not disjoint:
            failures.append(f"context {context} overlaps the fixed validation rows")

        print(
            f"{context:>9,}  {str(is_prefix):>14}  {positives:>8,}  "
            f"{str(disjoint):>17}  {digest[:16]}"
        )
        records[str(context)] = {
            "rows": int(len(index)),
            "positive": positives,
            "prefix_of_largest": bool(is_prefix),
            "balanced_50_50": bool(balanced),
            "unique_rows": bool(unique),
            "disjoint_from_validation": bool(disjoint),
            "sha256": digest,
            "first_10": [int(v) for v in index[:10]],
            "last_10": [int(v) for v in index[-10:]],
        }

    if failures:
        print("\nGATE 1 FAILED:")
        for message in failures:
            print(f"  - {message}")
        return 1

    write_json_with_provenance(
        args.out,
        {
            "schema_version": 1,
            "gate": "context_nesting",
            "seed": args.seed,
            "validation_rows": args.validation_rows,
            "candidate_rows": int(len(candidate_index)),
            "candidate_positive": int(candidate_y.sum()),
            "frozen_100k_sha256": FROZEN_100K_SHA256,
            "reconstructed_100k_sha256": frozen_sha,
            "nesting_reference_rows": largest,
            "contexts": records,
        },
        root=ROOT,
    )
    print(
        f"\nGATE 1 PASSED: every context is an exact prefix of the {largest:,} "
        f"context, and 100k reproduces the frozen digest.\nWrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
