"""Summarise the imported E3 variance pilot and apply the frozen decision rule.

Three artifacts are written:

* `e3_summary.json` -- per-cell repeat-level statistics for every readout the
  protocol declares, with the gating and non-gating ones kept apart so a
  non-gating endpoint can never be mistaken for evidence that the pilot passed.
* `e3_decision.json` -- the single verdict, selected by the protocol's own
  `e3_decision_rule` and by nothing else.
* `e3_input_manifest.json` -- every input the numbers depend on, by digest.

Fresh-process runs are reported alongside the same-process distribution but are
never pooled into it: they have their own block and are excluded from every
mean, SD, and confidence interval here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5_e3_runner import half_width  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values: list[float]) -> dict:
    arr = np.asarray(values, dtype="float64")
    n = arr.size
    hw = half_width(arr.tolist())
    mean = float(arr.mean())
    return {
        "n": int(n),
        "mean": mean,
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "half_width": hw,
        "ci95": [mean - hw, mean + hw],
        "t_critical": float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else None,
        "all_identical": bool(np.all(arr == arr[0])),
    }


def summarise_cell(root: Path, cell: str, proto: dict) -> dict:
    cdir = root / f"worker_{cell}" / f"cell_{cell}"
    complete = read_json(cdir / "CELL_COMPLETE.json")
    fit = read_json(cdir / "fit_complete.json")
    start = read_json(cdir / "fit_start.json")
    repeats = [read_json(p) for p in sorted((cdir / "repeats").glob("*.json"))]

    series: dict[str, list[float]] = {}
    for rep in repeats:
        for key, value in rep["endpoints"].items():
            series.setdefault(key, []).append(float(value))

    gating = set(
        proto["supplementary_decisions"]["4_precision_gate_endpoints"]["gating"]
    )
    # The protocol names the margin endpoint by its prose form in the gate list
    # and by its artifact key in the readouts; both refer to one endpoint.
    gating = {
        "steam_positive_vs_hotwater_negative_pairwise_auc",
        "steam_positive_minus_hotwater_negative_score_margin",
    } | {k for k in gating if k in series}

    fresh_dir = cdir / "fresh"
    fresh = (
        [read_json(p) for p in sorted(fresh_dir.glob("*.json"))]
        if fresh_dir.is_dir()
        else []
    )

    return {
        "cell": cell,
        "cell_dir": proto["cells"][cell]["cell_dir"],
        "factorial": proto["cells"][cell]["factorial"],
        "context_rows": proto["cells"][cell]["context_rows"],
        "fits": 1,
        "state_sha256": complete["state_sha256"],
        "process_uuid": complete["process_uuid"],
        "status": complete["status"],
        "repeats": complete["repeats"],
        "stopped_at_batch": complete["stopped_at_batch"],
        "escalated": complete["stopped_at_batch"] > 8,
        "gate": complete["gate"],
        "gate_log": complete["gate_log"],
        "fit_seconds": fit.get("fit_seconds"),
        "query_sha256": start["query_sha256"],
        "distinct_repeat_score_digests": len({r["score_sha256"] for r in repeats}),
        "bitwise_identical_repeats": len({r["score_sha256"] for r in repeats}) == 1,
        "gating_endpoints": {
            k: describe(v) for k, v in sorted(series.items()) if k in gating
        },
        "non_gating_endpoints": {
            k: describe(v) for k, v in sorted(series.items()) if k not in gating
        },
        "fresh_process_runs": {
            "pooled_into_same_process_statistics": False,
            "runs": [
                {
                    "run_index": f["run_index"],
                    "state_sha256": f["state_sha256"],
                    "score_sha256": f["score_sha256"],
                    "load_seconds": f["load_seconds"],
                    "score_seconds": f["score_seconds"],
                    "endpoints": f["endpoints"],
                    "within_same_process_range": {
                        k: bool(min(series[k]) <= v <= max(series[k]))
                        for k, v in f["endpoints"].items()
                        if k in series
                    },
                }
                for f in fresh
            ],
        },
    }


def decide(cells: list[dict], proto: dict) -> dict:
    rule = proto["e3_decision_rule"]
    cap = proto["supplementary_decisions"]["3_repeat_batches"]["cap"]

    incomplete = [
        c["cell"]
        for c in cells
        if c["status"] != "COMPLETE_GATE_PASSED"
        or c["fits"] != 1
        or not c["state_sha256"]
    ]
    failing_at_cap = [
        c["cell"] for c in cells if not c["gate"]["both_pass"] and c["repeats"] >= cap
    ]
    pending = [
        c["cell"] for c in cells if not c["gate"]["both_pass"] and c["repeats"] < cap
    ]
    passing = [c["cell"] for c in cells if c["gate"]["both_pass"]]

    if incomplete:
        verdict = "E3_EXECUTION_INCOMPLETE"
    elif failing_at_cap:
        verdict = "E3_MEASUREMENT_PROCESS_UNSTABLE"
    elif pending:
        verdict = "E3_MORE_REPEATS_REQUIRED"
    elif len(passing) == len(proto["inherited"]["cells"]):
        verdict = "E3_MEASUREMENT_PROCESS_ACCEPTABLE"
    else:
        verdict = "E3_EXECUTION_INCOMPLETE"

    return {
        "verdict": verdict,
        "criterion": rule[verdict],
        "evaluated_against": "protocol e3_decision_rule; no other criterion was applied",
        "cells_passing_both_gate_endpoints": passing,
        "cells_incomplete": incomplete,
        "cells_failing_at_cap": failing_at_cap,
        "cells_pending_under_cap": pending,
        "repeat_cap": cap,
        "authorises": [],
        "note": (
            "The verdict concerns measurement-process stability only. It says the "
            "repeat-level readouts are precise enough to be reported, not that any "
            "scientific claim about the cells is established. It authorises no "
            "downstream stage; the prohibitions in the protocol remain in force."
        ),
        "unresolved_carried_forward": {
            "chilledwater_vs_hotwater_negative": proto["readouts"][
                "chilledwater_vs_hotwater_negative"
            ]["status"],
            "onset_phase_contrast": proto["readouts"]["onset_phase_contrast"]["status"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    proto_path = args.root / "e3_protocol.json"
    proto = read_json(proto_path)["protocol"]
    order = proto["supplementary_decisions"]["1_schedule_seed"]["realised_cell_order"]
    cells = [summarise_cell(args.root, c, proto) for c in order]
    decision = decide(cells, proto)

    inputs = {
        "protocol_artifact": {
            "path": proto_path.name,
            "sha256": sha256_file(proto_path),
        },
        "base_commit": proto["base_commit"],
        "base_policy": proto["base_policy"],
        "query": {
            "sha256": cells[0]["query_sha256"],
            "identical_across_cells": len({c["query_sha256"] for c in cells}) == 1,
        },
        "fit_states": {c["cell"]: c["state_sha256"] for c in cells},
        "reference_iqr": {
            c: {
                "reference_iqr": proto["reference_iqr"]["per_cell"][c]["reference_iqr"],
                "margin_half_width_target": proto["reference_iqr"]["per_cell"][c][
                    "margin_half_width_target"
                ],
                "tree_predictions_sha256": proto["reference_iqr"]["per_cell"][c][
                    "tree_predictions_sha256"
                ],
            }
            for c in order
        },
        "result_file_manifest_sha256": sha256_file(
            args.root / "e3_file_manifest.sha256"
        ),
        "environment": {
            "summariser_python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    summary = {
        "schema": "m5_e3_variance_pilot_summary_v1",
        "generated": time.time(),
        "protocol_sha256": inputs["protocol_artifact"]["sha256"],
        "realised_cell_order": order,
        "execution": {
            "site": "remote gpu-host (WSL2 Ubuntu, RTX 5070 Ti)",
            "cells_run_in_parallel": False,
            "total_fits": sum(c["fits"] for c in cells),
            "total_repeats": sum(c["repeats"] for c in cells),
            "any_cell_escalated": any(c["escalated"] for c in cells),
            "distinct_fit_states": len({c["state_sha256"] for c in cells}),
        },
        "cells": cells,
    }

    for name, payload in (
        ("e3_summary.json", summary),
        ("e3_decision.json", decision),
        ("e3_input_manifest.json", inputs),
    ):
        out = args.root / name
        tmp = out.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(out)
        print(f"wrote {name}")

    print(f"\nVERDICT: {decision['verdict']}")
    for c in cells:
        g = c["gate"]
        print(
            f"  cell {c['cell']} n={c['repeats']} "
            f"auc_hw={g['auc_half_width']:.6f}<={g['auc_target']:.6f} "
            f"margin_hw={g['margin_half_width']:.6f}<={g['margin_target']:.6f} "
            f"distinct_digests={c['distinct_repeat_score_digests']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
