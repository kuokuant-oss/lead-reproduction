from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


work = Path("/content/lead_tabpfn_head/work")
launcher_path = work / "launcher.json"
payload: dict[str, object] = {
    "chunk_count": len(list((work / "chunks").glob("rows_*.npz"))),
}
if launcher_path.exists():
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    pid = int(launcher["pid"])
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    payload.update(alive=alive, pid=pid)
    payload["process"] = subprocess.run(
        ["ps", "-o", "pid=,ppid=,stat=,etime=,cmd=", "-p", str(pid)],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip()
else:
    payload.update(alive=False, pid=None)
for name in ("heartbeat.json", "progress.json", "result.json"):
    path = work / name
    if path.exists():
        try:
            payload[name] = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as error:
            payload[name] = {"read_error": repr(error)}
log_path = work / "worker.log"
if log_path.exists():
    payload["log_tail"] = log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()[-20:]
print(json.dumps(payload, sort_keys=True))
gpu_status = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    check=False,
    text=True,
)
print(gpu_status.stdout.strip())
