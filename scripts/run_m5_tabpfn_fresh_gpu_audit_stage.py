"""Run one immutable fresh-process GPU lifecycle stage from frozen sentinel inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from lead import ROOT


AUDIT = (
    ROOT
    / "data"
    / "processed"
    / "m5_hotwater_label_factorial"
    / "deterministic_execution_audit"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("P6_fresh_gpu_reload_1", "P7_fresh_gpu_reload_2"),
        required=True,
    )
    args = parser.parse_args()
    started_epoch = time.time()
    target = AUDIT / "fresh_gpu" / f"{args.stage}.npz"
    meta = target.with_suffix(".json")
    if target.exists() or meta.exists():
        raise FileExistsError(f"immutable stage already exists: {target}")
    from tabpfn.model_loading import load_fitted_tabpfn_model
    import torch

    with np.load(AUDIT / "inputs.npz") as payload:
        query = np.asarray(payload["scaled_query"], dtype="float32")
        raw_index = np.asarray(payload["query_raw_index"], dtype="int64")
    model = load_fitted_tabpfn_model(AUDIT / "model.tabpfn_fit", device="cuda")
    proba = np.asarray(model.predict_proba(query), dtype="float32")
    raw_logits = (
        model._raw_predict(query, return_logits=False, return_raw_logits=True)
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )
    temperature = float(
        getattr(model, "softmax_temperature_", model.softmax_temperature)
    )
    per_probability = (
        torch.softmax(torch.from_numpy(raw_logits) / temperature, dim=-1)
        .numpy()
        .astype("float32")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".tmp").open("wb") as stream:
        np.savez_compressed(
            stream,
            raw_index=raw_index,
            aggregate_probability=proba,
            positive_score=proba[:, 1],
            raw_logits=raw_logits,
            per_estimator_probability=per_probability,
        )
    os.replace(target.with_name(target.name + ".tmp"), target)
    metadata = {
        "stage": args.stage,
        "status": "completed",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "command": sys.argv,
        "device": "cuda",
        "n_estimators": int(model.n_estimators_),
        "state_sha256": digest(AUDIT / "model.tabpfn_fit"),
        "inputs_sha256": digest(AUDIT / "inputs.npz"),
        "scaler_sha256": digest(AUDIT / "scaler.joblib"),
        "query_dtype": str(query.dtype),
        "query_sha256": hashlib.sha256(query.tobytes()).hexdigest(),
    }
    temporary_meta = meta.with_name(meta.name + ".tmp")
    temporary_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_meta, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
