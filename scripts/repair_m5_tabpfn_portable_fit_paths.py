"""Rewrite the checkpoint path embedded in exported portable TabPFN fit states.

A portable fit is a zip whose ``init_params.json`` carries ``model_path``, the
absolute path TabPFN uses to find the foundation checkpoint at load time. If that
path does not exist on the box, TabPFN does not fail loudly -- it decides the
weights are missing and tries to *download* them, which on a box with
``TABPFN_NO_BROWSER=1`` surfaces as ``TabPFNLicenseError``. The traceback names
licensing and says nothing about paths, so the real cause is easy to miss.

Three ways the exported shards acquired wrong paths:

* The 137-feature exporter still emits Colab roots (``/content/lead_tabpfn_137_head``)
  and, worse, omits the context from the name, so every context shares one path.
  The head shards only worked because a leftover ``/content/lead_tabpfn_137_head``
  directory happened to survive on the pod; the tail had no counterpart and died.
* ``--remote-prefix /workspace`` passed through Git Bash was rewritten by MSYS path
  translation into ``C:/Program Files/Git/workspace/...``, which is meaningless on
  the Linux pod. This hit 17-feature 50k and 5k.
* Only 17-feature 10k and 20k came out correct.

Rewriting the one JSON entry is enough: the fitted tensors are untouched, so this
costs seconds instead of re-exporting multi-GiB feature matrices.

    uv run python scripts/repair_m5_tabpfn_portable_fit_paths.py --dry-run
    uv run python scripts/repair_m5_tabpfn_portable_fit_paths.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

PROC = Path("data/processed")
CKPT_NAME = "tabpfn-v3-classifier-v3_default.ckpt"
INIT = "init_params.json"


def shard_dirs() -> list[tuple[Path, int, int, str]]:
    """Every exported shard directory with its (context, line, shard) identity."""
    found: list[tuple[Path, int, int, str]] = []
    for line, pattern in (
        (137, "m5_tabpfn_137_distributed_context*_n8"),
        (17, "m5_tabpfn_f17_batch0_context*_n8"),
    ):
        for root in sorted(PROC.glob(pattern)):
            context = int(root.name.split("context")[1].split("_")[0])
            # 100k is the finished, published line; it ran on Colab and its
            # artifacts are already committed. Never rewrite a completed run.
            if context == 100_000:
                continue
            for shard in ("head", "tail"):
                if (root / shard / "model.portable.tabpfn_fit").is_file():
                    found.append((root / shard, context, line, shard))
    return found


def expected_path(context: int, line: int, shard: str) -> str:
    """The remote root gputw_tabpfn_shard.sh actually hydrates and runs from."""
    return f"/workspace/lead_tabpfn_c{context}_b0_{shard}_f{line}_n8/{CKPT_NAME}"


def read_model_path(archive: Path) -> str | None:
    with zipfile.ZipFile(archive) as zf:
        return json.loads(zf.read(INIT)).get("model_path")


def rewrite(archive: Path, new_path: str) -> None:
    """Copy the archive through, replacing only init_params.json.

    Rebuilt rather than edited in place: a zip entry cannot grow without shifting
    every later offset, and a partially rewritten fit state would be a far worse
    failure than the one being fixed. The temp file is swapped in only after it
    closes cleanly and reads back with the intended path.
    """
    tmp = archive.with_suffix(".repair-tmp")
    with (
        zipfile.ZipFile(archive) as src,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == INIT:
                params = json.loads(data)
                params["model_path"] = new_path
                data = json.dumps(params).encode()
            dst.writestr(item, data)
    if read_model_path(tmp) != new_path:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"verification failed for {archive}")
    shutil.move(tmp, archive)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--dry-run", action="store_true", help="report only (default)")
    args = ap.parse_args()
    apply = args.apply and not args.dry_run

    changed = 0
    for directory, context, line, shard in shard_dirs():
        archive = directory / "model.portable.tabpfn_fit"
        current = read_model_path(archive)
        want = expected_path(context, line, shard)
        tag = f"{line:>3}f c{context:<6} {shard:<4}"
        if current == want:
            print(f"  ok   {tag}")
            continue
        changed += 1
        print(f"  FIX  {tag}\n         from {current}\n         to   {want}")
        if apply:
            rewrite(archive, want)

    print(f"\n{changed} archive(s) {'repaired' if apply else 'need repair'}")
    if changed and not apply:
        print("re-run with --apply to write, then re-upload the fit states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
