#!/usr/bin/env bash
# Prepare a freshly rented gputw.ai box, and prove it can run this workload
# BEFORE any multi-GiB upload.
#
# Run it on the instance:
#   ssh user@<ip> 'bash -s' < scripts/gputw_bootstrap.sh
#
# The RTX 5090 is Blackwell (compute capability 12.0). A PyTorch build without
# sm_120 kernels imports fine, reports the GPU by name, and only fails when it
# tries to launch a kernel -- so "torch.cuda.is_available() is True" proves
# nothing here. The check below does a real matmul, which is the cheapest way to
# find out before spending an hour uploading shards to a box that cannot run
# them.

set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== torch / Blackwell kernel check ==="
python - <<'PY'
import sys
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA unavailable; CPU fallback is forbidden for this workload")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"device {name}  capability {major}.{minor}  {gib:.1f} GiB")

try:
    x = torch.randn(4000, 4000, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    print("fp32 matmul ok:", float((x @ x).sum()))
    h = x.half()
    print("fp16 matmul ok:", float((h @ h).float().sum()))
except RuntimeError as error:
    print(f"\nKERNEL LAUNCH FAILED: {error}")
    sys.exit(
        "This torch build has no kernels for sm_"
        f"{major}{minor}. Install a matching build, e.g.\n"
        "  pip install --upgrade torch "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
PY

echo
echo "=== tabpfn ==="
python -c "import tabpfn; print('tabpfn', tabpfn.__version__)" 2>/dev/null \
  || pip install 'tabpfn>=8,<9'
python -c "import tabpfn; print('tabpfn', tabpfn.__version__)"

echo
echo "=== workspace ==="
mkdir -p /workspace
df -h /workspace | tail -1
# Persistence across a stop is undocumented for this provider. Leave a marker,
# stop the instance, start it again, and look for this file before trusting a
# long run to /workspace.
date -Is > /workspace/PERSISTENCE_PROBE.txt
echo "wrote /workspace/PERSISTENCE_PROBE.txt -- check it survives a stop/start"

echo
echo "bootstrap OK"
