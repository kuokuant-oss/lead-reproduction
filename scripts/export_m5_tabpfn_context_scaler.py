"""Export one fit's StandardScaler as a portable two-array npz.

The context curve scores the same rows at five context sizes, and each context
refits its own scaler. Shipping a separately standardised copy of the feature
matrix per context costs 5.17 GiB each for the 137-feature line; shipping one
unscaled matrix plus these 2 x n_features numbers costs the same 5.17 GiB once.

``run_m5_tabpfn_portable_shard.py --scaler`` consumes the output and applies it
per microbatch, so the arithmetic is identical -- ``(X - mean) / scale`` either
way, just moved to where it is cheap to vary.

    uv run python scripts/export_m5_tabpfn_context_scaler.py \
        --work-dir data/processed/m5_tabpfn_137_full_test_context20000_n8.work
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <work-dir>/scaler.npz",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(
        (args.work_dir / "fit_manifest.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(args.work_dir / "scaler.joblib")
    mean = np.asarray(scaler.mean_, dtype="float32")
    scale = np.asarray(scaler.scale_, dtype="float32")

    expected = len(manifest["feature_names"])
    if mean.shape != (expected,) or scale.shape != (expected,):
        raise AssertionError(
            f"{args.work_dir} scaler covers {mean.shape} features, "
            f"but the fit names {expected}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise AssertionError("scaler contains non-finite values")
    if (scale == 0).any():
        # sklearn already rewrites zero variance to 1.0, so a zero here means
        # the object was tampered with rather than merely fit on constant data.
        raise AssertionError("scaler has a zero scale; refusing to export")

    out = args.out or (args.work_dir / "scaler.npz")
    np.savez(out, mean=mean, scale=scale)
    print(
        json.dumps(
            {
                "work_dir": str(args.work_dir.resolve()),
                "context_rows": manifest["context_rows"],
                "n_features": expected,
                "out": str(out.resolve()),
                "sha256": sha256_file(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
