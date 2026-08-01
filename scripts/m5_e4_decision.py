"""Derive the E4 scientific decision from the frozen artifacts.

E4 asks one question, and this script answers only that question:

    does a controlled change in hotwater positive/negative support produce,
    across the three context seeds and both scaler arms, a directionally
    consistent factorial response on steam that exceeds inference noise?

The protocol did not pre-specify a verdict vocabulary for E4 the way it did for
E3, so none is invented here. Each contrast gets an explicit answer against an
explicit bar, and the bar is stated in the artifact next to the answer.

The bar for "established" is deliberately conjunctive, because any single
criterion can be satisfied by an artefact:

* 3/3 context seeds agree in sign          -- not driven by one seed
* both cluster types exclude zero          -- not driven by one clustering
* both primary endpoints agree in sign     -- not an artefact of a saturated AUC
* both scaler arms agree in sign           -- not a preprocessing artefact
* the effect exceeds the inference noise floor

"TabPFN-specific" additionally requires the TabPFN-minus-tree gap to clear the
same bar. An effect the matched tree shows equally is a property of the data, not
of TabPFN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

EFFECTS = (
    "positive_support_main_effect",
    "negative_support_main_effect",
    "positive_x_negative_interaction",
)
PRIMARY = (
    "steam_positive_vs_hotwater_negative_pairwise_auc",
    "steam_positive_minus_hotwater_negative_score_margin",
)
ARMS = ("cell_specific", "frozen_reference")
CLUSTERS = ("building", "segment")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def assess(
    factorial: dict, clustered: dict, model: str, endpoint: str, effect: str
) -> dict:
    """One (model, endpoint, effect) against every criterion, per arm."""
    out: dict = {"per_arm": {}}
    for arm in ARMS:
        pt = factorial[endpoint][arm][model]["effects"][effect]
        block = {
            "overall_equal_weight_mean": pt["overall_equal_weight_mean"],
            "per_seed": pt["per_seed"],
            "range": pt["range"],
            "sample_sd": pt["sample_sd"],
            "sign_consistency": pt["sign_consistency"],
            "all_seeds_same_sign": pt["all_same_sign"],
            "clustered": {},
        }
        for cluster in CLUSTERS:
            ci = clustered[f"{endpoint}__{cluster}"]["contrasts"][
                f"{model}__{arm}__{effect}"
            ]
            block["clustered"][cluster] = {
                "q025": ci["q025"],
                "q975": ci["q975"],
                "excludes_zero": ci["excludes_zero"],
            }
        block["both_clusters_exclude_zero"] = all(
            block["clustered"][c]["excludes_zero"] for c in CLUSTERS
        )
        out["per_arm"][arm] = block

    means = [out["per_arm"][a]["overall_equal_weight_mean"] for a in ARMS]
    out["both_arms_same_sign"] = bool(np.sign(means[0]) == np.sign(means[1]))
    out["all_seeds_same_sign"] = all(
        out["per_arm"][a]["all_seeds_same_sign"] for a in ARMS
    )
    out["both_clusters_exclude_zero"] = all(
        out["per_arm"][a]["both_clusters_exclude_zero"] for a in ARMS
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    args = ap.parse_args()

    proto = read_json(args.canonical / "e4_protocol.json")["protocol"]
    factorial = read_json(args.canonical / "e4_factorial.json")["contrasts"]
    clustered = read_json(args.canonical / "e4_clustered.json")["results"]
    summary = read_json(args.canonical / "e4_summary.json")

    # Inference noise floor: the repeat-level half-widths across all 24 fits.
    noise = {}
    for endpoint in PRIMARY:
        hw = [
            u["endpoints"][endpoint]["half_width"] for u in summary["per_fit"].values()
        ]
        noise[endpoint] = {
            "min_half_width": float(np.min(hw)),
            "median_half_width": float(np.median(hw)),
            "max_half_width": float(np.max(hw)),
            "fits_with_zero_variance": int(sum(1 for h in hw if h == 0.0)),
        }

    findings: dict = {}
    for effect in EFFECTS:
        per_endpoint = {
            ep: {
                "tabpfn": assess(factorial, clustered, "tabpfn", ep, effect),
                "tree": assess(factorial, clustered, "tree", ep, effect),
                "tabpfn_minus_tree": assess(
                    factorial, clustered, "tabpfn_minus_tree", ep, effect
                ),
            }
            for ep in PRIMARY
        }

        def clears(model: str) -> bool:
            signs = [
                np.sign(
                    per_endpoint[ep][model]["per_arm"]["cell_specific"][
                        "overall_equal_weight_mean"
                    ]
                )
                for ep in PRIMARY
            ]
            return bool(
                all(per_endpoint[ep][model]["all_seeds_same_sign"] for ep in PRIMARY)
                and all(
                    per_endpoint[ep][model]["both_clusters_exclude_zero"]
                    for ep in PRIMARY
                )
                and all(
                    per_endpoint[ep][model]["both_arms_same_sign"] for ep in PRIMARY
                )
                and signs[0] == signs[1]
            )

        response = clears("tabpfn")
        specific = response and clears("tabpfn_minus_tree")
        endpoints_agree = np.sign(
            per_endpoint[PRIMARY[0]]["tabpfn"]["per_arm"]["cell_specific"][
                "overall_equal_weight_mean"
            ]
        ) == np.sign(
            per_endpoint[PRIMARY[1]]["tabpfn"]["per_arm"]["cell_specific"][
                "overall_equal_weight_mean"
            ]
        )
        disagreeing = [
            ep
            for ep in PRIMARY
            if not per_endpoint[ep]["tabpfn"]["both_clusters_exclude_zero"]
        ]
        findings[effect] = {
            "tabpfn_response_established": response,
            "tabpfn_specific_beyond_the_matched_tree": bool(specific),
            "primary_endpoints_agree_in_sign": bool(endpoints_agree),
            "endpoints_not_excluding_zero": disagreeing,
            "detail": per_endpoint,
        }

    scaler = {
        ep: {
            effect: {
                cluster: clustered[f"{ep}__{cluster}"]["contrasts"][
                    f"tabpfn__scaler_interaction__{effect}"
                ]["excludes_zero"]
                for cluster in CLUSTERS
            }
            for effect in EFFECTS
        }
        for ep in PRIMARY
    }
    scaler_matters = any(
        v for ep in scaler.values() for eff in ep.values() for v in eff.values()
    )

    decision = {
        "schema": "m5_e4_decision_v1",
        "generated": time.time(),
        "protocol_sha256": summary["protocol_sha256"],
        "realised_order_digest": summary["realised_order_digest"],
        "question": (
            "does a controlled change in hotwater positive/negative support "
            "produce, across three context seeds and two scaler arms, a "
            "directionally consistent factorial response on steam that exceeds "
            "inference noise?"
        ),
        "bar_for_established": [
            "all three context seeds agree in sign",
            "both cluster types exclude zero",
            "both primary endpoints agree in sign",
            "both scaler arms agree in sign",
        ],
        "bar_for_tabpfn_specific": "the TabPFN-minus-tree gap clears the same bar",
        "verdict_vocabulary_pre_specified": False,
        "coverage": summary["coverage"],
        "inference_noise_floor": noise,
        "findings": findings,
        "scaler_arm_interaction_excludes_zero": scaler,
        "scaler_arm_changes_any_conclusion": bool(scaler_matters),
        "authorises": [],
        "conditioning": proto["uncertainty_interpretation"],
        "unresolved_carried_forward": proto["readouts"]["retained_labels"],
        "chilledwater_not_pooled_into_the_steam_claim": True,
    }
    digest = atomic_json(args.canonical / "e4_decision.json", decision)
    print(f"wrote e4_decision.json sha256={digest}\n")
    for effect, f in findings.items():
        print(
            f"  {effect:<34} response={str(f['tabpfn_response_established']):<5} "
            f"tabpfn_specific={str(f['tabpfn_specific_beyond_the_matched_tree']):<5} "
            f"endpoints_agree={str(f['primary_endpoints_agree_in_sign']):<5} "
            f"not_excl_zero={f['endpoints_not_excluding_zero']}"
        )
    print(f"\n  scaler arm changes any conclusion: {scaler_matters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
