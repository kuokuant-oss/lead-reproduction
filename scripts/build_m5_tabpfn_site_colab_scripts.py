"""Generate the per-shard remote Colab scripts for the estimator sweep.

The official run ships one hand-written trio of remote scripts per shard. The
sweep needs four shards live at once (two sites x head/tail) and will reuse the
same trio for each estimator value, so the trio is generated from the shard
manifest instead of copied by hand: paths, the remote root and the worker flags
all come from ``manifest.json``, which keeps the launcher honest about which
fitted state and estimator count it is running.

Emits, under ``.scratch/site<k>_<shard>/``:

* ``create_dirs.py``   -- rebuild the remote work/chunks tree
* ``reassemble.py``    -- join uploaded parts and verify four SHA-256 digests
* ``launch.py``        -- start the detached worker with --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lead import ROOT

TEMPLATE_DIRS = """from pathlib import Path

root = Path("{remote_root}")
(root / "work" / "chunks").mkdir(parents=True, exist_ok=True)
print(str(root / "work" / "chunks"))
"""

TEMPLATE_REASSEMBLE = """from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("{remote_root}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assemble(name: str) -> None:
    parts = sorted(ROOT.glob(f"{{name}}.part*"))
    if not parts:
        return
    temporary = ROOT / f"{{name}}.assembling"
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    temporary.replace(ROOT / name)


manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
expected = {{
    "features.float32.npy": manifest["features"]["sha256"],
    "metadata.npz": manifest["metadata"]["sha256"],
    "model.portable.tabpfn_fit": manifest["fit_state"]["sha256"],
    "tabpfn-v3-classifier-v3_default.ckpt": (
        manifest["foundation_checkpoint"]["sha256"]
    ),
}}
for filename in ("features.float32.npy", "tabpfn-v3-classifier-v3_default.ckpt"):
    assemble(filename)
observed = {{name: digest(ROOT / name) for name in expected}}
if observed != expected:
    raise RuntimeError(
        "SHA-256 mismatch: "
        + json.dumps({{"expected": expected, "observed": observed}})
    )
print(json.dumps({{"sha256": observed, "verified": True}}, sort_keys=True))
"""

TEMPLATE_LAUNCH = """from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("{remote_root}")
work = root / "work"
work.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    str(root / "run_m5_tabpfn_portable_shard.py"),
    "--features",
    str(root / "features.float32.npy"),
    "--metadata",
    str(root / "metadata.npz"),
    "--fit-state",
    str(root / "model.portable.tabpfn_fit"),
    "--work-dir",
    str(work),
    "--context-rows",
    "100000",
    "--n-features",
    "{n_features}",
    "--n-estimators",
    "{n_estimators}",
    "--query-microbatch-size",
    "{microbatch}",
    "--min-query-microbatch-size",
    "64",
    "--checkpoint-rows",
    "20000",
    "--direction",
    "{direction}",
    "--resume",
]
log_path = work / "worker.log"
with log_path.open("ab", buffering=0) as log:
    log.write(f"\\n=== launch {{time.time()}} site{site} {shard} n={n_estimators} ===\\n".encode())
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=root,
        start_new_session=True,
    )
payload = {{
    "command": command,
    "launched_at_unix": time.time(),
    "pid": process.pid,
}}
temporary = work / "launcher.json.tmp"
temporary.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
os.replace(temporary, work / "launcher.json")
print(json.dumps(payload, sort_keys=True))
"""


TEMPLATE_CALIBRATE = """from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Detached, exactly like the worker, for two independent reasons:
#  * the Colab kernel keeps the base-image numpy in memory, so an in-kernel
#    import hits that stale build instead of the pinned runtime on disk;
#  * a blocking exec dies with the kernel. A calibration that has to complete
#    one full microbatch of real inference runs for minutes, and losing it to a
#    kernel recycle leaves the caller waiting on a reply that never comes.
# Progress is therefore read from the JSON file, not from this call.
root = Path("{remote_root}")
work = root / "work"
work.mkdir(parents=True, exist_ok=True)
out_path = work / "microbatch_calibration_n{n_estimators}.json"
command = [
    sys.executable,
    str(root / "calibrate_m5_tabpfn_microbatch.py"),
    "--features",
    str(root / "features.float32.npy"),
    "--fit-state",
    str(root / "model.portable.tabpfn_fit"),
    "--out",
    str(out_path),
    "--n-estimators",
    "{n_estimators}",
    "--candidates",
    {candidate_args}
    "--max-seconds-per-candidate",
    "{max_seconds}",
]
log_path = work / "calibration.log"
with log_path.open("ab", buffering=0) as log:
    log.write(f"\\n=== calibrate {{time.time()}} n={n_estimators} ===\\n".encode())
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=root,
        start_new_session=True,
    )
print(json.dumps({{"pid": process.pid, "out": str(out_path)}}, sort_keys=True))
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard", choices=("head", "tail"), required=True)
    parser.add_argument(
        "--query-microbatch-size",
        type=int,
        required=True,
        help="calibrated on the target GPU before the real run",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--calibration-seconds", type=float, default=90.0)
    parser.add_argument(
        "--calibration-candidates",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
    )
    args = parser.parse_args(argv)

    manifest = json.loads(
        (args.shard_root / args.shard / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["shard"] != args.shard:
        raise AssertionError("manifest shard name disagrees with --shard")
    if args.query_microbatch_size > 20_000:
        raise ValueError("microbatch cannot exceed the 20,000-row checkpoint size")
    if args.query_microbatch_size < 64:
        raise ValueError("microbatch cannot go below the contract minimum 64")

    # Site shards carry "site"; the remaining-holdout batches carry "batch".
    # Only used for labelling the generated scripts and the log line.
    if "site" in manifest:
        site = int(manifest["site"])
    else:
        site = int(manifest["batch"])
    fields = {
        "remote_root": manifest["remote_root"],
        "n_estimators": int(manifest["fit_state"]["n_estimators"]),
        "microbatch": args.query_microbatch_size,
        "direction": manifest["direction"],
        "site": site,
        "shard": args.shard,
        # Must come from the manifest: the 137-feature line reuses this
        # generator, and a hard-coded 17 makes the worker refuse to start.
        "n_features": int(manifest["n_features"]),
        "max_seconds": args.calibration_seconds,
        "candidate_args": "".join(
            f'"{value}",\n    ' for value in sorted(args.calibration_candidates)
        ),
    }
    out_dir = args.out_dir or (ROOT / ".scratch" / f"site{site}_{args.shard}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, template in (
        ("create_dirs.py", TEMPLATE_DIRS),
        ("reassemble.py", TEMPLATE_REASSEMBLE),
        ("launch.py", TEMPLATE_LAUNCH),
        ("calibrate.py", TEMPLATE_CALIBRATE),
    ):
        path = out_dir / name
        path.write_text(template.format(**fields), encoding="utf-8")
        written.append(str(path))
    print(
        json.dumps(
            {
                "site": site,
                "shard": args.shard,
                "remote_root": fields["remote_root"],
                "n_estimators": fields["n_estimators"],
                "direction": fields["direction"],
                "query_microbatch_size": fields["microbatch"],
                "written": written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
