from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


root = Path("/content/lead_tabpfn_tail")
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
    "--query-microbatch-size",
    "1024",
    "--min-query-microbatch-size",
    "64",
    "--checkpoint-rows",
    "20000",
    "--direction",
    "reverse",
    "--resume",
]
log_path = work / "worker.log"
with log_path.open("ab", buffering=0) as log:
    log.write(f"\n=== launch {time.time()} exact-runtime ===\n".encode())
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=root,
        start_new_session=True,
    )
payload = {
    "command": command,
    "launched_at_unix": time.time(),
    "pid": process.pid,
}
temporary = work / "launcher.json.tmp"
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, work / "launcher.json")
print(json.dumps(payload, sort_keys=True))
