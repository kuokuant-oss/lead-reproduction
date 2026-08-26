#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
ROOT="/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k"
PYTHON="/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python"
MODEL="/home/kuant_kuo/.cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt"
FORMAL_ROOT="$ROOT/data/processed/m5_building_curve/v5_fixed_50k"
VALIDATION_ROOT="$ROOT/data/processed/m5_building_curve/NON_SCIENTIFIC_VALIDATION_v5_fixed_50k"
IMPLEMENTATION_COMMIT="8d604c91ebfc69a3fd1234df74b89c67304ba82c"

if [[ "$MODE" != "--check" && "$MODE" != "--run" ]]; then
    echo "usage: $0 [--check|--run]" >&2
    exit 2
fi

cd "$ROOT"

if pgrep -af 'scripts/run_m5_building_count_v5.py|run_m5_building_count_v5_(tree|tabpfn)_cell.py' \
    | grep -v 'pgrep -af' >/dev/null; then
    echo "BLOCKED: an M5 V5 scheduler/model process already exists" >&2
    exit 3
fi
if [[ -e "$FORMAL_ROOT/supervisor/FAILED.json" ]]; then
    echo "BLOCKED: formal supervisor has FAILED.json; do not auto-retry" >&2
    exit 4
fi
if [[ ! -x "$PYTHON" || ! -f "$MODEL" ]]; then
    echo "BLOCKED: Python environment or TabPFN checkpoint is missing" >&2
    exit 5
fi
if ! git merge-base --is-ancestor "$IMPLEMENTATION_COMMIT" HEAD; then
    echo "BLOCKED: V5 implementation commit is not an ancestor of HEAD" >&2
    exit 6
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "BLOCKED: tracked worktree is not clean" >&2
    exit 7
fi
if [[ ! -f "$VALIDATION_ROOT/matched_context_gate.json" ]]; then
    echo "BLOCKED: bounded model-validation gate is missing" >&2
    exit 8
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
env PYTHONPATH=src:scripts "$PYTHON" \
    scripts/prepare_m5_building_count_v5_fixed_50k.py --mode check
env PYTHONPATH=src:scripts "$PYTHON" -m unittest \
    tests.test_m5_building_count_v5
env PYTHONPATH=src:scripts "$PYTHON" \
    scripts/run_m5_building_count_v5.py --mode plan >/dev/null

echo "PRECHECK PASSED: 42 contexts / 84 model cells; no formal model launched"
if [[ "$MODE" == "--check" ]]; then
    exit 0
fi

exec env PYTHONPATH=src:scripts "$PYTHON" \
    scripts/run_m5_building_count_v5.py --mode formal --authorize-formal
