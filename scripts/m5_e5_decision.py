"""Apply E5's pre-declared replication verdict.

The vocabulary and every threshold were frozen in `e5_protocol.json` before any
192-row score existed. This script reads them from the protocol rather than
restating them, so a threshold cannot drift between the freeze and the verdict.

Two verdicts are produced independently: whether the response replicated, and
whether it replicated as something specific to TabPFN rather than shared with
the matched fixed tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np  # noqa: F401  (used by the synthetic-input tests)

EFFECT = "negative_support_main_effect"
ARMS = ("cell_specific", "frozen_reference")
CLUSTERS = ("building", "segment")
SEEDS = ("42", "123", "999")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def gather(factorial: dict, clustered: dict, model: str, endpoints: list[str]) -> dict:
    """Every quantity the pre-declared rules refer to, per endpoint and arm."""
    out: dict = {}
    for ep in endpoints:
        per_arm = {}
        for arm in ARMS:
            pt = factorial[ep][arm][model]["effects"][EFFECT]
            block = {
                "overall": pt["overall_equal_weight_mean"],
                "per_seed": pt["per_seed"],
                "seeds_positive": sum(1 for s in SEEDS if pt["per_seed"][s] > 0),
                "range": pt["range"],
                "sample_sd": pt["sample_sd"],
                "sign_consistency": pt["sign_consistency"],
                "clustered": {},
            }
            for c in CLUSTERS:
                ci = clustered[f"{ep}__{c}"]["contrasts"][f"{model}__{arm}__{EFFECT}"]
                block["clustered"][c] = {
                    "q025": ci["q025"],
                    "q975": ci["q975"],
                    "excludes_zero": ci["excludes_zero"],
                    "excludes_zero_positive": bool(ci["q025"] > 0),
                }
            per_arm[arm] = block
        out[ep] = per_arm
    return out


def evaluate(data: dict, endpoints: list[str]) -> dict:
    """The frozen rules, each recorded with the value that decided it."""
    overall_positive = {
        ep: all(data[ep][a]["overall"] > 0 for a in ARMS) for ep in endpoints
    }
    all_seeds_positive = {
        ep: all(data[ep][a]["seeds_positive"] == 3 for a in ARMS) for ep in endpoints
    }
    two_of_three_positive = {
        ep: all(data[ep][a]["seeds_positive"] >= 2 for a in ARMS) for ep in endpoints
    }
    arms_positive = {
        ep: all(data[ep][a]["overall"] > 0 for a in ARMS) for ep in endpoints
    }
    intervals_exclude_zero = {
        ep: all(
            data[ep][a]["clustered"][c]["excludes_zero_positive"]
            for a in ARMS
            for c in CLUSTERS
        )
        for ep in endpoints
    }

    replicated = (
        all(overall_positive.values())
        and all(all_seeds_positive.values())
        and all(arms_positive.values())
        and all(intervals_exclude_zero.values())
    )
    not_replicated = (
        any(not overall_positive[ep] for ep in endpoints)
        or any(not arms_positive[ep] for ep in endpoints)
        or any(not two_of_three_positive[ep] for ep in endpoints)
    )
    directional = (
        not replicated
        and not not_replicated
        and all(overall_positive.values())
        and all(arms_positive.values())
        and all(two_of_three_positive.values())
    )

    if replicated:
        verdict = "REPLICATED"
    elif not_replicated:
        verdict = "NOT_REPLICATED"
    elif directional:
        verdict = "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE"
    else:
        verdict = "NOT_REPLICATED"
    return {
        "verdict": verdict,
        "criteria": {
            "overall_positive_on_both_endpoints": all(overall_positive.values()),
            "three_of_three_seeds_positive": all(all_seeds_positive.values()),
            "at_least_two_of_three_seeds_positive": all(two_of_three_positive.values()),
            "both_arms_positive": all(arms_positive.values()),
            "both_clustered_intervals_exclude_zero_upward": all(
                intervals_exclude_zero.values()
            ),
        },
        "per_endpoint": {
            ep: {
                "overall_positive": overall_positive[ep],
                "seeds_positive_per_arm": {
                    a: data[ep][a]["seeds_positive"] for a in ARMS
                },
                "intervals_exclude_zero": intervals_exclude_zero[ep],
            }
            for ep in endpoints
        },
        "detail": data,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    args = ap.parse_args()

    proto = read_json(args.canonical / "e5_protocol.json")["protocol"]
    rules = proto["decision_rules"]
    endpoints = list(rules["co_primary_endpoints"])
    factorial = read_json(args.canonical / "e5_factorial.json")["contrasts"]
    clustered = read_json(args.canonical / "e5_clustered.json")["results"]
    summary = read_json(args.canonical / "e5_summary.json")
    cov = summary["coverage"]

    incomplete = [
        k
        for k, ok in (
            ("states_reloaded", cov["states_reloaded"] == 24),
            ("same_process_repeats", cov["same_process_repeats"] == 192),
            ("score_vector_length", cov["score_vector_length"] == 192),
            ("tree_score_vectors", cov["tree_score_vectors"] == 24),
            ("fits_performed_is_zero", cov["fits_performed"] == 0),
            ("effective_n_estimators", cov["effective_n_estimators_"] == [8]),
            ("scaler_verified_exact", cov["scaler_verified_exact"] is True),
        )
        if not ok
    ]

    response = evaluate(gather(factorial, clustered, "tabpfn", endpoints), endpoints)
    specific = evaluate(
        gather(factorial, clustered, "tabpfn_minus_tree", endpoints), endpoints
    )
    tree = gather(factorial, clustered, "tree", endpoints)

    if incomplete:
        response["verdict"] = "EXECUTION_INCOMPLETE"
        specific["verdict"] = "EXECUTION_INCOMPLETE"

    decision = {
        "schema": "m5_e5_decision_v1",
        "generated": time.time(),
        "protocol_sha256": summary["protocol_sha256"],
        "replication_target": rules["primary_replication_target"],
        "co_primary_endpoints": endpoints,
        "vocabulary": rules["vocabulary"],
        "thresholds_frozen_before_scoring": True,
        "coverage": cov,
        "coverage_failures": incomplete,
        "A_response_replication": response,
        "B_tabpfn_specific_replication": specific,
        "fixed_tree_response": tree,
        "e4_reference": {
            "negative_support_main_effect_auc": 0.4118,
            "negative_support_main_effect_margin": 0.6312,
            "tabpfn_minus_tree_auc": 0.1237,
            "tabpfn_minus_tree_margin": 0.1522,
            "note": "E4 values on the 352-row screening query, for orientation "
            "only; E5's verdict is decided by E5's own frozen rules",
        },
        "interval_containing_zero_is_not_proof_of_absence": True,
        "authorises": [],
    }
    digest = atomic_json(args.canonical / "e5_decision.json", decision)
    print(f"wrote e5_decision.json sha256={digest}\n")
    print(f"  A response replication      : {response['verdict']}")
    print(f"  B TabPFN-specific replication: {specific['verdict']}")
    if incomplete:
        print(f"  coverage failures           : {incomplete}")
    for ep in endpoints:
        for arm in ARMS:
            d = response["detail"][ep][arm]
            print(
                f"    {ep[:44]:<44} {arm:<17} overall={d['overall']:+.5f} "
                f"seeds+={d['seeds_positive']}/3 "
                f"bldg=[{d['clustered']['building']['q025']:+.4f},"
                f"{d['clustered']['building']['q975']:+.4f}] "
                f"seg=[{d['clustered']['segment']['q025']:+.4f},"
                f"{d['clustered']['segment']['q975']:+.4f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
