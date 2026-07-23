from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTabPFNPortableShard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worker = load_script("run_m5_tabpfn_portable_shard")
        cls.exporter = load_script("export_m5_tabpfn_colab_tail")
        cls.merger = load_script("merge_m5_tabpfn_distributed_predictions")

    def test_reverse_checkpoint_spans_cover_every_row_once(self) -> None:
        spans = self.worker.checkpoint_spans(45, 20, "reverse")
        self.assertEqual(spans, [(40, 45), (20, 40), (0, 20)])
        rows = [value for start, end in spans for value in range(start, end)]
        self.assertEqual(sorted(rows), list(range(45)))
        self.assertEqual(len(rows), len(set(rows)))

    def test_metadata_requires_contiguous_global_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.npz"
            np.savez(
                path,
                raw_index=np.array([11, 13]),
                anomaly=np.array([0, 1]),
                site_id=np.array([1, 2]),
                building_id=np.array([893, 895]),
                global_position=np.array([100, 102]),
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                self.worker.load_metadata(path)

    def test_checkpoint_rejects_row_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npz"
            np.savez(
                path,
                raw_index=np.array([3, 5]),
                anomaly=np.array([0, 1]),
                score=np.array([0.2, 0.8]),
                site_id=np.array([1, 1]),
                building_id=np.array([893, 893]),
            )
            self.assertIsNotNone(
                self.worker.load_saved_checkpoint(path, np.array([3, 5]))
            )
            with self.assertRaisesRegex(AssertionError, "identity drifted"):
                self.worker.load_saved_checkpoint(path, np.array([5, 3]))

    def test_relocated_archive_changes_only_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tabpfn_fit"
            destination = root / "portable.tabpfn_fit"
            original = {
                "__class_name__": "TabPFNClassifier",
                "model_path": "C:\\old\\model.ckpt",
                "fit_mode": "low_memory",
            }
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("init_params.json", json.dumps(original))
                archive.writestr("executor_state.joblib", b"executor")
                archive.writestr("fitted_attrs.joblib", b"attributes")

            remote = PurePosixPath("/content/lead_tabpfn_tail/model.ckpt")
            self.exporter.relocate_fitted_archive(source, destination, remote)

            with zipfile.ZipFile(destination) as archive:
                relocated = json.loads(archive.read("init_params.json"))
                self.assertEqual(archive.read("executor_state.joblib"), b"executor")
                self.assertEqual(archive.read("fitted_attrs.joblib"), b"attributes")
            self.assertEqual(relocated["model_path"], str(remote))
            self.assertEqual(relocated["fit_mode"], original["fit_mode"])
            self.assertEqual(relocated["__class_name__"], original["__class_name__"])

    def test_exporter_accepts_forward_head_bounds(self) -> None:
        args = self.exporter.parse_args(
            [
                "--global-start",
                "0",
                "--global-end",
                "5060000",
                "--shard",
                "head",
                "--direction",
                "forward",
            ]
        )
        self.assertEqual(args.global_start, 0)
        self.assertEqual(args.global_end, 5_060_000)
        self.assertEqual(args.shard, "head")
        self.assertEqual(args.direction, "forward")

    def test_worker_has_no_top_level_torch_or_tabpfn_import(self) -> None:
        self.assertFalse(self.worker.parent_has_forbidden_imports())

    def test_merge_reconstructs_exact_head_and_tail_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            head = root / "head"
            tail = root / "tail"
            head.mkdir()
            tail.mkdir()
            rows = 45
            raw_index = np.arange(rows, dtype="int64") * 2 + 1
            y = (np.arange(rows) % 2).astype("int8")
            site = (np.arange(rows) % 4).astype("int8")
            building = (np.arange(rows) // 5 * 2 + 1).astype("int16")
            canonical = root / "canonical.npz"
            np.savez(
                canonical,
                validation_raw_index=raw_index,
                anomaly=y,
                site_id=site,
                building_id=building,
            )
            expected_score = np.linspace(0, 1, rows, dtype="float32")
            for start in (0, 10):
                end = start + 10
                np.savez(
                    self.merger.head_checkpoint_path(head, start, 10),
                    raw_index=raw_index[start:end],
                    y=y[start:end],
                    score=expected_score[start:end],
                    site_id=site[start:end],
                    building_id=building[start:end],
                )
            for start, end in ((20, 30), (30, 40), (40, 45)):
                np.savez(
                    self.merger.tail_checkpoint_path(tail, start, end),
                    raw_index=raw_index[start:end],
                    anomaly=y[start:end],
                    score=expected_score[start:end],
                    site_id=site[start:end],
                    building_id=building[start:end],
                )
            output = root / "merged.npz"
            self.assertEqual(
                self.merger.main(
                    [
                        "--canonical",
                        str(canonical),
                        "--head-chunks",
                        str(head),
                        "--tail-chunks",
                        str(tail),
                        "--boundary",
                        "20",
                        "--checkpoint-rows",
                        "10",
                        "--out",
                        str(output),
                    ]
                ),
                0,
            )
            with np.load(output) as merged:
                np.testing.assert_array_equal(merged["raw_index"], raw_index)
                np.testing.assert_array_equal(merged["tabpfn"], expected_score)


if __name__ == "__main__":
    unittest.main()
