"""Record the human override on where the fixed tree comparator may be scored.

`e5_protocol.json` is not rewritten. This is a supplemental artifact: the
protocol said all scientific scoring happens on gpu-host, execution showed that
rule cannot be satisfied for the tree comparator, and a human ruled on it. Both
the original rule and the evidence that broke it are recorded here so the
override can be audited rather than inferred from a changed protocol.

The override relaxes one engineering constraint and nothing else. TabPFN still
scores on gpu-host, no tree is refit, and no gpu-host tree output may enter the
analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(r"C:\Users\tonykuo\projects\lead-reproduction")
E5_ROOT = ROOT / "data" / "processed" / "m5_e5_independent_replication"
FACTORIAL = ROOT / "data" / "processed" / "m5_hotwater_label_factorial"

SOURCE_FILES = [
    "scripts/m5_e5_tree_override.py",
    "scripts/m5_e5_tree_local.py",
    "scripts/m5_e5_protocol.py",
    "scripts/m5_e5_runner.py",
    "scripts/m5_e5_guard.py",
    "scripts/m5_e5_analysis.py",
    "scripts/m5_e5_decision.py",
    "scripts/m5_e4_endpoints.py",
    "scripts/m5_e4_clustered.py",
]

QUERY192_FEATURE_SHA = (
    "e6b44c9ccc902cd6dfa6f1fce07ad98d9af1af52dab32faaf36b148d12ab0482"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.replace(tmp, path)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if sha256_file(path) != digest:
        raise AssertionError(f"{path.name} does not match its body digest")
    return digest


def override() -> dict:
    return {
        "schema": "m5_e5_tree_execution_override_v1",
        "issued": "2026-08-02",
        "authority": "human operator ruling",
        "supplements": "e5_protocol.json",
        "protocol_history_rewritten": False,
        "original_rule": (
            "all E5 scientific scoring executes on gpu-host, in a fresh clean "
            "clone; the laptop is forbidden from scoring the 192-row query"
        ),
        "observed_hard_failure": {
            "where": "m5_e5_trees.py comparator identity gate on gpu-host",
            "first_unit": "seed42__cell11__frozen_reference",
            "message": "the reloaded tree does not reproduce E4's 352-row comparator",
            "ruled_out_by_diagnosis": [
                "scaler: recovery, frozen and refitted scalers all gave the "
                "identical error, so the scaler is not the variable",
                "query matrix: the E4 cached 352-row matrix is bit-identical to "
                "a fresh build",
                "threshold flipping: all 352 rows differ and none are exact, "
                "which is systemic rather than a few rows crossing a split",
                "tree reliability: E4's own reproduction gate passed 24/24 tree "
                "rows; its 13 failures were all TabPFN",
            ],
        },
        "gpu_host_reproduction": {
            "host": "gpu-host, WSL2 Ubuntu, Python 3.12.13, Linux",
            "max_abs_difference": 1.245e-01,
            "mean_abs_difference": 8.149e-03,
            "bit_exact_rows": 0,
            "rows": 352,
            "verdict": "cannot reproduce E4's frozen comparator",
        },
        "laptop_bit_exact_evidence": {
            "host": "laptop, Windows-11-10.0.26200-SP0, Python 3.13.13",
            "units_tested_before_the_ruling": [
                "seed42__cell11__cell_specific",
                "seed42__cell11__frozen_reference",
                "seed999__cell00__cell_specific",
            ],
            "max_abs_difference": 0.0,
            "bit_exact_rows": "352/352 on every unit tested",
            "verdict": "reproduces E4's frozen comparator exactly",
        },
        "root_cause": (
            "the tree ensembles were fitted in the laptop environment recorded "
            "in recovery/environment_provenance_trees.json (Windows 11, Python "
            "3.13.13, torch 2.12.1+cu126); gradient-boosting inference is not "
            "bit-reproducible across that and gpu-host's Linux / Python 3.12.13 "
            "build, and the compiled lightgbm/xgboost/catboost packages on "
            "gpu-host were reinstalled on 2026-08-01"
        ),
        "human_decision": "OPTION_A",
        "tabpfn_execution_host": "gpu-host",
        "tree_execution_host": "original laptop environment",
        "no_refit": True,
        "comparator_identity_requirement": "bit_exact_on_e4_352_query",
        "comparator_gate": {
            "units_required": 24,
            "max_abs_diff_required": 0.0,
            "exact_rows_required": "352/352 per unit",
            "sampling_not_permitted": True,
            "tolerance_not_permitted": True,
            "on_failure": "stop the whole of E5",
        },
        "gpu_host_tree_outputs_prohibited": True,
        "scientific_design_unchanged": True,
        "decision_rules_unchanged": True,
        "endpoints_unchanged": True,
        "states_unchanged": True,
        "clustered_estimator_unchanged": True,
        "tabpfn_specific_threshold_not_lowered": True,
        "shared_input_requirement": {
            "base_192_row_feature_matrix": "one artifact, used by both hosts",
            "sha256": QUERY192_FEATURE_SHA,
            "shape": [192, 137],
            "laptop_may_not_build_its_own": True,
        },
        "reporting_requirement": (
            "this is an execution-provenance limitation and must not be "
            "described as a new scientific factor"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()

    payload = override()
    digest = atomic_json(args.out / "e5_tree_execution_override.json", payload)

    sources = {}
    for rel in SOURCE_FILES:
        p = args.repo / rel
        if not p.exists():
            raise SystemExit(f"source file missing: {rel}")
        sources[rel] = sha256_source(p)

    inputs_path = args.out / "e5_input_manifest.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    inputs["tree_execution_override_sha256"] = digest
    inputs["source_digests"] = sources
    inputs["base_192_row_feature_sha256"] = QUERY192_FEATURE_SHA
    inputs["tree_execution_host"] = payload["tree_execution_host"]
    inputs["tabpfn_execution_host"] = payload["tabpfn_execution_host"]
    inputs["updated"] = time.time()
    updated = atomic_json(inputs_path, inputs)

    print(f"override        sha256 = {digest}")
    print(f"input manifest  sha256 = {updated}")
    print(f"base 192 feature sha256 = {QUERY192_FEATURE_SHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
