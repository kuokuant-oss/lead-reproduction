from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "m5_e7_full_capacity_tree_strategy"


def test_final_manifest_has_eight_frozen_slots_and_exact_pairs():
    data = json.loads((OUT / "e7_final_training_pool_manifest.json").read_text())
    records = {record["slot"]: record for record in data["records"]}
    assert set(records) == {"s00", "s01", "s10", "s11", "n00", "n01", "n10", "n11"}
    for cell in ("00", "01", "10", "11"):
        support, neutral = records[f"s{cell}"], records[f"n{cell}"]
        assert (
            support["sampled_row_count"],
            support["positive_count"],
            support["negative_count"],
        ) == (
            neutral["sampled_row_count"],
            neutral["positive_count"],
            neutral["negative_count"],
        )
        assert support["odd_rows_used"] == neutral["odd_rows_used"] == 0
        assert support["odd_labels_read"] == neutral["odd_labels_read"] == 0


def test_s11_matches_historical_realised_count_and_amendment_is_pre_execution():
    records = json.loads((OUT / "e7_final_training_pool_manifest.json").read_text())[
        "records"
    ]
    assert (
        next(record for record in records if record["slot"] == "s11")[
            "sampled_row_count"
        ]
        == 2708308
    )
    amendment = json.loads((OUT / "e7_protocol_amendment_001.json").read_text())
    assert amendment["formal_fits_before_amendment"] == 0
    assert amendment["odd_predictions_before_amendment"] == 0
