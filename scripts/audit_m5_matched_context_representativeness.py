"""Are the chosen rows representative of the class they were drawn from?

The balance is not the question -- 50/50 is the protocol, declared and applied to
both models. The question is whether *these particular* 50,000 positives and
50,000 negatives are an ordinary draw or a special one. A sampler can be
uniform-looking in code and still be handed a pre-sorted pool, and the frozen
digest proves only that the draw was fixed early, not that it was typical.

So this compares the drawn rows against the full class they came from, on
everything that could plausibly make a row easy: which site it belongs to, which
building, and where its meter_reading falls. A uniform draw reproduces the parent
distribution to within sampling noise; a curated one does not. Nothing here is
conditioned on any model output.

    uv run python scripts/audit_m5_matched_context_representativeness.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np

from lead import PROC, RANDOM_STATE, ROOT, load_m3_frame

CANONICAL_ORDER = PROC / "m6_site_transfer_b2_a0_pos677077_seed42_predictions.npz"


def load_canonical_module():
    spec = importlib.util.spec_from_file_location(
        "m5_canonical", ROOT / "scripts" / "run_m5_tabpfn_canonical_full_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare(name, drawn, parent, quantiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    finite_d = drawn[np.isfinite(drawn)]
    finite_p = parent[np.isfinite(parent)]
    print(f"  {name}: drawn n={len(finite_d):,}  parent n={len(finite_p):,}")
    print(f"    {'q':>6} {'drawn':>16} {'parent':>16} {'rel.diff':>10}")
    worst = 0.0
    for q in quantiles:
        a, b = np.quantile(finite_d, q), np.quantile(finite_p, q)
        denom = max(abs(b), 1e-9)
        rel = (a - b) / denom
        worst = max(worst, abs(rel))
        print(f"    {q:>6.2f} {a:>16.4f} {b:>16.4f} {rel:>+9.2%}")
    print(
        f"    mean drawn {finite_d.mean():.4f} vs parent {finite_p.mean():.4f}; "
        f"largest quantile deviation {worst:.2%}"
    )
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--validation-rows", type=int, default=4_000)
    args = parser.parse_args()
    mod = load_canonical_module()

    frame = load_m3_frame(verbose=True)
    row_id = frame.index.to_numpy(dtype="int64")
    order = np.argsort(row_id)
    building = frame["building_id"].to_numpy(dtype="int32")
    site = frame["site_id"].to_numpy(dtype="int16")
    anomaly = frame["anomaly"].to_numpy(dtype="int8")
    reading = frame["meter_reading"].to_numpy(dtype="float64")
    del frame

    def positions(ids):
        return order[np.searchsorted(row_id, ids, sorter=order)]

    train_index = row_id[building % 2 == 0]
    validation_index = mod.fixed_score_indices(
        train_index, args.validation_rows, seed=args.seed + 20_000
    )
    candidate_index = train_index[~np.isin(train_index, validation_index)]
    cand_pos = positions(candidate_index)
    cand_y = anomaly[cand_pos]

    fit_index = mod.nested_balanced_indices(
        candidate_index, cand_y, [args.context_rows], seed=args.seed
    )[args.context_rows]
    fit_pos = positions(fit_index)
    fit_y = anomaly[fit_pos]

    for label, cls in (("POSITIVES", 1), ("NEGATIVES", 0)):
        drawn = fit_pos[fit_y == cls]
        parent = cand_pos[cand_y == cls]
        print(
            f"\n=== {label}: {len(drawn):,} drawn from {len(parent):,} available "
            f"({len(drawn) / len(parent):.2%}) ==="
        )

        print("  site composition (drawn vs parent, within class):")
        d_share = np.bincount(site[drawn], minlength=16) / len(drawn)
        p_share = np.bincount(site[parent], minlength=16) / len(parent)
        worst_site, worst_s = 0.0, -1
        for s in range(16):
            if p_share[s] == 0:
                continue
            rel = (d_share[s] - p_share[s]) / p_share[s]
            if abs(rel) > worst_site:
                worst_site, worst_s = abs(rel), s
            print(
                f"    site {s:>2}: drawn {d_share[s]:>7.3%}  parent {p_share[s]:>7.3%}  {rel:>+7.2%}"
            )
        print(
            f"    largest within-class site deviation: site {worst_s}, {worst_site:.2%}"
        )

        nb_d, nb_p = len(np.unique(building[drawn])), len(np.unique(building[parent]))
        print(f"  buildings: {nb_d} of {nb_p} carrying this class appear in the draw")
        compare("meter_reading", reading[drawn], reading[parent])

    print(
        "\nA uniform draw of this size has a standard error of roughly "
        f"{1 / np.sqrt(args.context_rows / 2):.2%} on a class share, so deviations of "
        "a few percent are sampling noise, not curation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
