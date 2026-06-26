# M5 Phase C handoff: TabPFN local metric complete

## Scope

This slice completed issue #30 for the LEAD/M3-only TabPFN feasibility spike.
No BDG2 data was downloaded or processed. The run preserved the frozen
`src/lead` path: M3.2 80/20 mod5 split, offline row-offset value-change regime,
M3 downsampling seeds, shared scaler, and the same reduced feature table for the
GBDT and TabPFN models.

## What Changed

+ Added `.tabpfn-cache/` and `*.ckpt` to `.gitignore`.
+ Copied the accepted checkpoint to
  `.tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt`; the checkpoint remains
  ignored and untracked.
+ Added `--model-path` support to `scripts/run_m5_phaseC_tabpfn_spike.py` so
  local weights can be used without browser login, token access, or online
  download.
+ Updated `tests/test_m5_tabpfn_spike.py` to keep guarding the frozen M3 path
  and cover the local-checkpoint branch.
+ Regenerated `data/processed/m5_phaseC_tabpfn_spike.json` with a real TabPFN
  metric and latency.

## Commands

```powershell
New-Item -ItemType Directory -Force -Path .\.tabpfn-cache
Copy-Item "C:\Users\tonykuo\Downloads\tabpfn-v3-classifier-v3_default.ckpt" `
  ".\.tabpfn-cache\tabpfn-v3-classifier-v3_default.ckpt"

$env:UV_CACHE_DIR='C:\Users\tonykuo\projects\lead-reproduction\.uv-cache'
$env:UV_PYTHON_INSTALL_DIR='C:\Users\tonykuo\projects\lead-reproduction\.uv-python'
$env:TABPFN_MODEL_CACHE_DIR='C:\Users\tonykuo\projects\lead-reproduction\.tabpfn-cache'
$env:TABPFN_NO_BROWSER='1'
$env:TABPFN_DISABLE_TELEMETRY='1'
.\.venv\Scripts\python.exe scripts\run_m5_phaseC_tabpfn_spike.py `
  --max-fit-rows 1000 --max-val-rows 1000 --tabpfn-batch-size 256 `
  --model-path ".\.tabpfn-cache\tabpfn-v3-classifier-v3_default.ckpt"
```

An explicit CUDA torch install was attempted for CUDA 12.x. The command timed
out, but the environment did land on `torch==2.7.1+cu128` and verified
`cuda_available=true`.

## Evidence

+ Full M3 downsample shape: `4,285,104 x 137`, above the documented TabPFN-3
  `1,000,000 x 200` limit.
+ Reduced local table: `1,000 x 137` train rows and `1,000` validation rows.
+ Device: `NVIDIA GeForce RTX 4070 Laptop GPU`, `8187` MiB VRAM,
  `torch==2.7.1+cu128`, `cuda_available=true`.
+ GBDT anchor: AUC `0.986955266955267`, precision `0.5196078431372549`, recall
  `0.9636363636363636`, F1 `0.6751592356687898`, fit+predict
  `0.8137212998699397` seconds.
+ TabPFN: AUC `0.9904377104377105`, precision `0.5196078431372549`, recall
  `0.9636363636363636`, F1 `0.6751592356687898`, fit+predict
  `8.195373600116` seconds.
+ Result archive: `data/processed/m5_phaseC_tabpfn_spike.json`.

The run emitted a nonfatal Windows background `UnicodeDecodeError` from a
dependency subprocess after the JSON was saved. The result file was written
successfully and validated by later gates.

## Current State

Unknowns 19-21 are updated with evidence. ADR 0015 is Accepted. README and the
M5 tracker mark Phase C complete. Issue #30 is closed by the commit message.

## Next Step

Phase D is issue #31 and is docs-only in this commit: plan BDG2 ingestion,
few-shot transfer, minimal-feature comparison, and causal discipline before any
BDG2 data or dependency work starts.
