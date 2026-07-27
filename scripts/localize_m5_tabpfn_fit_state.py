"""Point an exported portable fit at a local checkpoint, without touching the original.

The exported shards carry the *pod's* checkpoint path in `init_params.json`
(`/workspace/lead_tabpfn_.../tabpfn-v3-classifier-v3_default.ckpt`). Running one
of them on this machine finds nothing there, and TabPFN's response to a missing
checkpoint is not an error about the path -- it decides the weights need
downloading and raises `TabPFNLicenseError`, a traceback entirely about licensing.
That has cost this project a debugging session already, so the fix is explicit.

A copy is written rather than an edit in place: the originals are the verified
remote-ready artifacts, and the whole grid's provenance rests on their digests.
Only the fit state is copied -- a few MB -- because the worker takes the feature
matrix and metadata as separate paths and can read those 2.6 GiB files where they
already are.

    uv run python scripts/localize_m5_tabpfn_fit_state.py \
        --shard-dir data/processed/m5_tabpfn_137_distributed_context5000_n8/head \
        --out data/processed/m5_tabpfn_local_c5000_f137_head/model.portable.tabpfn_fit
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

DEFAULT_CKPT = Path(".tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt")
INIT = "init_params.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args(argv)

    source = args.shard_dir / "model.portable.tabpfn_fit"
    if not source.is_file():
        raise SystemExit(f"no portable fit in {args.shard_dir}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if INIT not in members:
        raise SystemExit("portable fit lacks init_params.json")
    params = json.loads(members[INIT])
    was = params.get("model_path")
    params["model_path"] = str(checkpoint)
    members[INIT] = json.dumps(params).encode("utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, args.out)

    # Read it back: the point of this script is that a wrong path here fails
    # later as something that does not mention paths at all.
    with zipfile.ZipFile(args.out, "r") as archive:
        got = json.loads(archive.read(INIT))["model_path"]
    if got != str(checkpoint) or not Path(got).is_file():
        raise SystemExit(f"rewrite did not take: {got}")
    print(f"{source}\n  was: {was}\n  now: {got}\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
