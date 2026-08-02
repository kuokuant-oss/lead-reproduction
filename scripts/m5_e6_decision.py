"""M5 E6 decision: apply the frozen rules to the frozen analysis.

The rules were fixed before any holdout row was scored and are read from
`e6_decision_rules.json` rather than restated here, so this module cannot
quietly become a second, more convenient copy of them.

Two things this deliberately refuses to do. It does not convert "the interval
excludes zero" into "the magnitude reproduced" -- the E6/E5 ratio and the E6-E5
difference are reported beside every effect precisely so a shrunken effect
cannot pass as a reproduced one. And it does not read an interval that contains
zero as evidence of absence; with no minimum practical effect threshold set,
such an interval is uninformative about magnitude in both directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

CELLS_ENDPOINT = {
    "auc": "steam_positive_vs_hotwater_negative_pairwise_auc",
    "margin": "steam_positive_minus_hotwater_negative_score_margin",
}
FAMILY_PRIOR = {"tabpfn": "tabpfn", "tree": "tree", "gap": "tabpfn_minus_tree"}
SEEDS = ("42", "123", "999")
ARMS = ("cell_specific", "frozen_reference")
PRIMARY = "negative_support_main_effect"


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


def prior_effect(prior: dict, ep: str, family: str, effect: str) -> dict:
    """One stage's effect, averaged over the two arms exactly as E6 averages."""
    c = prior["contrasts"][CELLS_ENDPOINT[ep]]
    per_arm, per_seed = {}, {}
    for arm in ARMS:
        node = c[arm][FAMILY_PRIOR[family]]["effects"][effect]
        per_arm[arm] = node["overall_equal_weight_mean"]
        for s, v in node["per_seed"].items():
            per_seed.setdefault(s, []).append(v)
    return {
        "overall": sum(per_arm.values()) / len(per_arm),
        "per_arm": per_arm,
        "per_seed": {s: sum(v) / len(v) for s, v in per_seed.items()},
    }


def compare(e4: float, e5: float, e6: float) -> dict:
    """The comparison columns every main table must carry."""
    return {
        "e4_effect": e4,
        "e5_effect": e5,
        "e6_effect": e6,
        "e6_over_e5_ratio": (e6 / e5) if e5 != 0 else None,
        "e6_minus_e5_difference": e6 - e5,
        "e6_over_e4_ratio": (e6 / e4) if e4 != 0 else None,
        "e6_minus_e4_difference": e6 - e4,
    }


def evaluate(analysis: dict, family: str, effect: str) -> dict:
    """Every response-confirmation condition, endpoint by endpoint."""
    point = analysis["point"]
    iv = analysis["intervals"]
    out = {}
    for ep in ("auc", "margin"):
        eff = point[f"{family}|{ep}"]["effects"]
        key = f"x|overall|{effect}"
        out[ep] = {
            "overall": eff["overall"][effect],
            "overall_positive": eff["overall"][effect] > 0,
            "per_seed": {s: eff["per_seed"][f"seed{s}"][effect] for s in SEEDS},
            "seeds_positive": sum(
                eff["per_seed"][f"seed{s}"][effect] > 0 for s in SEEDS
            ),
            "per_arm": {a: eff["per_arm"][a][effect] for a in ARMS},
            "arms_positive": sum(eff["per_arm"][a][effect] > 0 for a in ARMS),
            "building_interval": iv[f"building|{family}|{ep}"][key],
            "segment_interval": iv[f"segment|{family}|{ep}"][key],
        }
    conditions = {
        "auc_overall_positive": out["auc"]["overall_positive"],
        "margin_overall_positive": out["margin"]["overall_positive"],
        "both_positive_in_3_of_3_seeds": (
            out["auc"]["seeds_positive"] == 3 and out["margin"]["seeds_positive"] == 3
        ),
        "both_scaler_arms_positive": (
            out["auc"]["arms_positive"] == 2 and out["margin"]["arms_positive"] == 2
        ),
        "building_interval_excludes_zero_both": (
            out["auc"]["building_interval"]["excludes_zero"]
            and out["margin"]["building_interval"]["excludes_zero"]
        ),
        "segment_interval_excludes_zero_both": (
            out["auc"]["segment_interval"]["excludes_zero"]
            and out["margin"]["segment_interval"]["excludes_zero"]
        ),
    }
    return {
        "endpoints": out,
        "conditions": conditions,
        "all_met": all(conditions.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--e4-factorial", type=Path, required=True)
    ap.add_argument("--e5-factorial", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--execution-incomplete",
        action="store_true",
        help="record EXECUTION_INCOMPLETE without inspecting the numbers",
    )
    args = ap.parse_args()

    rules = read_json(args.protocol_root / "e6_decision_rules.json")
    analysis = read_json(args.analysis)
    e4 = read_json(args.e4_factorial)
    e5 = read_json(args.e5_factorial)

    if args.execution_incomplete:
        digest = atomic_json(
            args.out / "e6_decision.json",
            {
                "schema": "m5_e6_decision_v1",
                "decision": "EXECUTION_INCOMPLETE",
                "reason": "the run did not complete; no scientific claim is made",
                "generated": time.time(),
            },
        )
        print(f"decision = EXECUTION_INCOMPLETE  sha256={digest}")
        return 0

    tabpfn = evaluate(analysis, "tabpfn", PRIMARY)
    gap = evaluate(analysis, "gap", PRIMARY)
    tree = evaluate(analysis, "tree", PRIMARY)

    tables = {}
    for family in ("tabpfn", "tree", "gap"):
        for ep in ("auc", "margin"):
            for effect in (
                "negative_support_main_effect",
                "positive_support_main_effect",
                "positive_x_negative_interaction",
            ):
                p4 = prior_effect(e4, ep, family, effect)
                p5 = prior_effect(e5, ep, family, effect)
                p6 = analysis["point"][f"{family}|{ep}"]["effects"]
                row = compare(p4["overall"], p5["overall"], p6["overall"][effect])
                row["building_interval"] = analysis["intervals"][
                    f"building|{family}|{ep}"
                ][f"x|overall|{effect}"]
                row["segment_interval"] = analysis["intervals"][
                    f"segment|{family}|{ep}"
                ][f"x|overall|{effect}"]
                row["per_seed"] = {
                    "e4": p4["per_seed"],
                    "e5": p5["per_seed"],
                    "e6": {s: p6["per_seed"][f"seed{s}"][effect] for s in SEEDS},
                }
                row["per_arm"] = {
                    "e4": p4["per_arm"],
                    "e5": p5["per_arm"],
                    "e6": {a: p6["per_arm"][a][effect] for a in ARMS},
                }
                tables[f"{family}|{ep}|{effect}"] = row

    if tabpfn["all_met"] and gap["all_met"]:
        decision = "NATURAL_PREVALENCE_CONFIRMED"
    elif (
        tabpfn["endpoints"]["auc"]["overall_positive"]
        and (tabpfn["endpoints"]["margin"]["overall_positive"])
    ):
        decision = "DIRECTIONALLY_CONSISTENT_BUT_INCONCLUSIVE"
    else:
        decision = "NOT_CONFIRMED"

    primary = tables[f"tabpfn|auc|{PRIMARY}"]
    magnitude = {
        "threshold": rules["minimum_practical_effect_threshold"],
        "rule": rules["magnitude_claim_rule"],
        "auc_e6_over_e5_ratio": primary["e6_over_e5_ratio"],
        "margin_e6_over_e5_ratio": tables[f"tabpfn|margin|{PRIMARY}"][
            "e6_over_e5_ratio"
        ],
        "claim_permitted": "direction and sign only; with no minimum practical "
        "effect threshold set, neither an interval excluding zero nor one "
        "containing zero settles whether the magnitude reproduced",
    }

    payload = {
        "schema": "m5_e6_decision_v1",
        "generated": time.time(),
        "decision": decision,
        "vocabulary": rules["vocabulary"],
        "primary_contrast": PRIMARY,
        "co_primary_endpoints": rules["co_primary_endpoints"],
        "decision_rules_sha256": analysis["decision_rules_sha256"],
        "analysis_sha256": hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        "tabpfn_conditions": tabpfn,
        "tabpfn_minus_tree_conditions": gap,
        "tree_conditions": tree,
        "comparison_tables": tables,
        "magnitude": magnitude,
        "segment_degeneracy_disclosure": analysis["segment_degeneracy_disclosure"],
        "estimand_note": analysis["estimand_note"],
        "mechanism_limitation": (
            "cell 00 removes hotwater support and simultaneously flips the meter "
            "feature from numerical to categorical, so this confirms the "
            "negative-support intervention as a whole and does not isolate the "
            "hotwater-normal reference as the sole mechanism"
        ),
        "interval_containing_zero_is_not_proof_of_absence": True,
    }
    digest = atomic_json(args.out / "e6_decision.json", payload)

    print(f"decision = {decision}")
    for name, block in (("tabpfn", tabpfn), ("tabpfn_minus_tree", gap)):
        print(f"\n  {name}:")
        for k, v in block["conditions"].items():
            print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  primary contrast ({PRIMARY}, tabpfn, auc):")
    print(
        f"    E4 {primary['e4_effect']:+.6f}   E5 {primary['e5_effect']:+.6f}   "
        f"E6 {primary['e6_effect']:+.6f}"
    )
    r = primary["e6_over_e5_ratio"]
    print(f"    E6/E5 ratio {r:.4f}   E6-E5 {primary['e6_minus_e5_difference']:+.6f}")
    print(f"\ndecision sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
