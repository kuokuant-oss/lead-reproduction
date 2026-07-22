from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "run_m5_tabpfn_canonical_full_test.py"


def load_script():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("m5_canonical_full_test", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class TestTabPFNCanonicalFullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_script()

    def make_args(self, directory: Path):
        return self.m.parse_args(
            [
                "--context-rows",
                "200",
                "--validation-rows",
                "200",
                "--query-microbatch-size",
                "128",
                "--min-query-microbatch-size",
                "64",
                "--checkpoint-rows",
                "512",
                "--out",
                str(directory / "summary.json"),
                "--state-out",
                str(directory / "state.json"),
                "--events-out",
                str(directory / "events.jsonl"),
                "--work-dir",
                str(directory / "work"),
                "--predictions-out",
                str(directory / "predictions.npz"),
                "--smoke",
            ]
        )

    def test_contract_uses_exact_even_train_and_odd_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            frame = self.m.synthetic_frame(rows_per_building=40)
            contract = self.m.canonical_contract(frame, args)
            context_buildings = frame.loc[
                contract["context_index"], "building_id"
            ].to_numpy()
            test_buildings = frame.loc[contract["test_index"], "building_id"].to_numpy()
            self.assertTrue(np.all(context_buildings % 2 == 0))
            self.assertTrue(np.all(test_buildings % 2 == 1))
            self.assertEqual(
                len(np.intersect1d(contract["context_index"], contract["test_index"])),
                0,
            )
            np.testing.assert_array_equal(
                contract["test_y"],
                frame.loc[contract["test_index"], "anomaly"].to_numpy(dtype="int8"),
            )

    def test_checkpoint_rejects_row_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npz"
            index = np.arange(10, dtype="int64")
            self.m.save_checkpoint(
                path,
                raw_index=index,
                y=index.astype("int8") % 2,
                score=np.linspace(0, 1, len(index)),
                site_id=np.zeros(len(index), dtype="int8"),
                building_id=np.ones(len(index), dtype="int16"),
            )
            self.assertIsNotNone(self.m.load_checkpoint(path, index))
            with self.assertRaisesRegex(AssertionError, "identity drifted"):
                self.m.load_checkpoint(path, index[::-1])

    def test_contract_rejects_canonical_artifact_row_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            args.smoke = False
            args.canonical_m3_predictions = root / "m3.npz"
            args.canonical_site_predictions = root / "site.npz"
            args.canonical_baseline_predictions = root / "baseline.npz"
            frame = self.m.synthetic_frame(rows_per_building=40)
            test = frame.loc[frame["building_id"] % 2 == 1]
            raw_index = test.index.to_numpy(dtype="int64")
            y = test["anomaly"].to_numpy(dtype="int8")
            scores = np.linspace(0, 1, len(test), dtype="float32")
            np.savez(
                args.canonical_m3_predictions,
                anomaly=y,
                m3_1_lightgbm=scores,
                lightgbm=scores,
                ensemble=scores,
            )
            drifted_index = raw_index.copy()
            drifted_index[-1] = drifted_index[0]
            np.savez(
                args.canonical_site_predictions,
                validation_raw_index=drifted_index,
                anomaly=y,
                site_id=test["site_id"].to_numpy(dtype="int8"),
                building_id=test["building_id"].to_numpy(dtype="int16"),
                ensemble=scores,
            )
            np.savez(
                args.canonical_baseline_predictions,
                anomaly=y,
                site_id=test["site_id"].to_numpy(dtype="int8"),
                ensemble=scores,
            )
            with self.assertRaisesRegex(AssertionError, "not unique"):
                self.m.canonical_contract(frame, args)

    def test_contract_uses_a0_order_that_aligns_all_four_m3_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            args.smoke = False
            args.canonical_m3_predictions = root / "m3.npz"
            args.canonical_site_predictions = root / "site.npz"
            args.canonical_baseline_predictions = root / "baseline.npz"
            frame = self.m.synthetic_frame(rows_per_building=40)
            test = frame.loc[frame["building_id"] % 2 == 1]
            raw_index = test.index.to_numpy(dtype="int64")
            permutation = np.random.RandomState(42).permutation(len(test))
            y = test["anomaly"].to_numpy(dtype="int8")[permutation]
            site_id = test["site_id"].to_numpy(dtype="int8")[permutation]
            scores = np.linspace(0, 1, len(test), dtype="float32")
            np.savez(
                args.canonical_m3_predictions,
                anomaly=y,
                m3_1_lightgbm=scores,
                lightgbm=scores,
                ensemble=scores,
            )
            np.savez(
                args.canonical_site_predictions,
                validation_raw_index=raw_index[permutation],
                anomaly=y,
                site_id=site_id,
                building_id=test["building_id"].to_numpy(dtype="int16")[permutation],
                ensemble=scores,
            )
            np.savez(
                args.canonical_baseline_predictions,
                anomaly=y,
                site_id=site_id,
                ensemble=scores,
            )
            contract = self.m.canonical_contract(frame, args)
            self.assertTrue(
                contract["metadata"]["canonical_m3_scores_aligned_to_a0_order"]
            )
            np.testing.assert_array_equal(
                contract["test_index"], raw_index[permutation]
            )

    def test_fit_state_is_saved_then_loaded_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            args.work_dir.mkdir(parents=True)
            frame = self.m.synthetic_frame()
            contract = self.m.canonical_contract(frame, args)
            heartbeat = self.m.Heartbeat(None, args.context_rows)
            first, _, first_action, _ = self.m.fit_or_load(
                frame, contract, args, heartbeat
            )
            saved_offset = first.offset_
            del first
            second, _, second_action, verified = self.m.fit_or_load(
                frame, contract, args, heartbeat
            )
            self.assertEqual(first_action, "fitted")
            self.assertEqual(second_action, "loaded")
            self.assertEqual(second.offset_, saved_offset)
            self.assertEqual(verified["effective_context_rows"], args.context_rows)
            self.assertTrue((args.work_dir / "fake_model.joblib").is_file())
            self.assertTrue((args.work_dir / "fit_manifest.json").is_file())

    def test_full_smoke_rerun_loads_fit_and_reuses_all_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            self.assertEqual(self.m.worker(args), 0)
            first = json.loads(
                (args.work_dir / "worker_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["fit_action"], "fitted")
            self.assertEqual(first["validation_action"], "predicted")
            checkpoint_mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in (args.work_dir / "chunks").glob("*.npz")
            }
            self.assertEqual(self.m.worker(args), 0)
            second = json.loads(
                (args.work_dir / "worker_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(second["fit_action"], "loaded")
            self.assertEqual(second["validation_action"], "loaded")
            self.assertEqual(second["completed_checkpoints"], len(checkpoint_mtimes))
            self.assertEqual(
                checkpoint_mtimes,
                {
                    path.name: path.stat().st_mtime_ns
                    for path in (args.work_dir / "chunks").glob("*.npz")
                },
            )

    def test_prediction_heartbeat_uses_global_position_offset(self) -> None:
        positions: list[int] = []
        heartbeat = SimpleNamespace(
            update=lambda _stage, position=0: positions.append(position)
        )
        model = SimpleNamespace(
            predict_proba=lambda matrix: np.column_stack(
                [np.zeros(len(matrix)), np.ones(len(matrix))]
            )
        )
        self.m.batched_predict(
            model,
            np.zeros((5, 2)),
            initial_batch_size=2,
            minimum_batch_size=2,
            stop_requested=lambda: False,
            heartbeat=heartbeat,
            stage="test",
            position_offset=20_000,
        )
        self.assertEqual(positions, [20_000, 20_002, 20_004])

    def test_parent_has_no_top_level_torch_or_tabpfn_import(self) -> None:
        self.assertFalse(self.m.parent_has_forbidden_imports())

    def test_new_invocation_clears_only_transient_worker_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            args.work_dir.mkdir(parents=True)
            transient = [
                args.work_dir / "worker_result.json",
                args.work_dir / "heartbeat.json",
                args.work_dir / "stop.json",
            ]
            durable = [
                args.work_dir / "model.tabpfn_fit",
                args.work_dir / "scaler.joblib",
                args.work_dir / "validation.npz",
                args.work_dir / "chunks" / "chunk_000000.npz",
            ]
            for path in [*transient, *durable]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"saved")

            self.m.prepare_worker_invocation(args.work_dir)

            self.assertTrue(all(not path.exists() for path in transient))
            self.assertTrue(all(path.is_file() for path in durable))


if __name__ == "__main__":
    unittest.main()
