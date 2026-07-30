from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_story_ae_probe import discover_manifests, validate_feature_matrix
from lead.m5_context import (
    M5_FIT_RULE,
    M5_HOLDOUT_RULE,
    array_sha256,
    build_context_manifest,
    build_query_artifact,
    context_indices,
    feature_names,
    parse_context_tag,
    protocol_source,
    validate_context_manifest,
)


def synthetic_frame() -> pd.DataFrame:
    rows = 4_800
    return pd.DataFrame(
        {
            "building_id": np.repeat(np.arange(48), 100),
            "site_id": np.repeat(np.arange(4), 1_200),
            "meter": np.tile(np.arange(4), 1_200),
            "anomaly": np.tile([0, 1], rows // 2),
        }
    )


class TestM5ContextConstructionProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        cls.source = protocol_source(cls.frame, seed=42, validation_rows=20)

    def test_feature_ladder_is_frozen(self) -> None:
        self.assertEqual(len(feature_names("F0")), 17)
        self.assertEqual(len(feature_names("F4")), 137)
        self.assertEqual(
            parse_context_tag("meter_heavy:hotwater:0.5"),
            {"kind": "meter_heavy", "target": "hotwater", "proportion": 0.5},
        )

    def test_context_is_deterministic_balanced_and_fit_only(self) -> None:
        first = context_indices(
            self.frame,
            context_rows=400,
            context_tag="pooled_reference",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        second = context_indices(
            self.frame,
            context_rows=400,
            context_tag="pooled_reference",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 400)
        self.assertEqual(len(np.unique(first)), 400)
        self.assertEqual(int(self.frame.iloc[first]["anomaly"].sum()), 200)
        self.assertTrue(set(first).isdisjoint(self.source["holdout_rows"]))

    def test_interventions_change_composition_and_exclusion_is_hard(self) -> None:
        balanced = context_indices(
            self.frame,
            context_rows=400,
            context_tag="meter_balanced",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        heavy = context_indices(
            self.frame,
            context_rows=400,
            context_tag="meter_heavy:3:0.8",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        excluded = context_indices(
            self.frame,
            context_rows=400,
            context_tag="meter_excluded:3",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        self.assertGreater(
            int((self.frame.iloc[heavy]["meter"] == 3).sum()),
            int((self.frame.iloc[balanced]["meter"] == 3).sum()),
        )
        self.assertEqual(int((self.frame.iloc[excluded]["meter"] == 3).sum()), 0)
        self.assertEqual(int(self.frame.iloc[excluded]["anomaly"].sum()), 200)

    def test_manifest_records_and_validates_identity_contract(self) -> None:
        indices = context_indices(
            self.frame,
            context_rows=400,
            context_tag="pooled_reference",
            seed=42,
            candidate_rows=self.source["candidate_rows"],
        )
        manifest = build_context_manifest(
            self.frame,
            indices,
            story="A_E_composition",
            context_tag="pooled_reference",
            context_rows=400,
            context_seed=42,
            model_seed=42,
            feature_tag="F4",
            split={"fit_rule": M5_FIT_RULE, "holdout_rule": M5_HOLDOUT_RULE},
        )
        self.assertEqual(manifest["feature_count"], 137)
        self.assertEqual(manifest["raw_index_sha256"], array_sha256(indices))
        validate_context_manifest(
            self.frame, manifest, holdout_rows=self.source["holdout_rows"]
        )
        broken = dict(manifest)
        broken["raw_index"] = list(reversed(broken["raw_index"]))
        with self.assertRaisesRegex(AssertionError, "digest mismatch"):
            validate_context_manifest(
                self.frame, broken, holdout_rows=self.source["holdout_rows"]
            )

    def test_query_artifact_is_ordered_holdout_subset(self) -> None:
        manifest, indices = build_query_artifact(
            self.frame, holdout_rows=self.source["holdout_rows"], rows_per_cell=2
        )
        self.assertEqual(manifest["raw_index_sha256"], array_sha256(indices))
        self.assertTrue(set(indices) <= set(self.source["holdout_rows"]))
        self.assertEqual(len(indices), len(np.unique(indices)))
        self.assertEqual(set(manifest["sentinel_sites"]), {"0", "2", "6", "9"})

    def test_probe_discovers_both_feature_cells_without_canonical_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = root / "manifests" / "A_E" / "pooled" / "n20000"
            cell.mkdir(parents=True)
            common = {
                "context_tag": "pooled_reference",
                "context_rows": 20_000,
                "context_seed": 42,
            }
            for feature_tag in ("F0", "F4"):
                payload = {**common, "feature_tag": feature_tag}
                (cell / f"seed42_{feature_tag.lower()}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            (cell / "seed42.json").write_text(
                json.dumps({**common, "feature_tag": "F0"}), encoding="utf-8"
            )
            paths = discover_manifests(root)
            self.assertEqual(
                [path.name for path in paths],
                ["seed42_f0.json", "seed42_f4.json"],
            )

    def test_probe_accepts_canonical_nan_but_rejects_infinity(self) -> None:
        stats = validate_feature_matrix(
            np.asarray([[1.0, np.nan]], dtype="float32"),
            matrix_name="fixture",
        )
        self.assertEqual(stats["nan_count"], 1)
        with self.assertRaisesRegex(AssertionError, "contains infinities"):
            validate_feature_matrix(
                np.asarray([[1.0, np.inf]], dtype="float32"),
                matrix_name="fixture",
            )


if __name__ == "__main__":
    unittest.main()
