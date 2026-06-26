# M5 Phase C handoff: TabPFN local feasibility spike

## Scope

This slice opened issue #30 and started the LEAD/M3-only TabPFN feasibility
spike. No BDG2 data was downloaded or processed. The spike uses the frozen
`src/lead` API, the M3.2 80/20 mod5 split, the offline row-offset
value-change regime, and the existing M3 downsampling helper.

## What changed

+ Added an optional `m5` uv dependency group:
  `torch>=2.3` and `tabpfn>=3.0`.
+ Added `scripts/run_m5_phaseC_tabpfn_spike.py`.
+ Added `tests/test_m5_tabpfn_spike.py` to guard that the runner imports the
  frozen `lead` helpers and keeps the M3 split/sample/regime path.
+ Archived the result at `data/processed/m5_phaseC_tabpfn_spike.json`.

## Install and run commands

```powershell
$env:UV_CACHE_DIR='C:\Users\tonykuo\projects\lead-reproduction\.uv-cache'
$env:UV_PYTHON_INSTALL_DIR='C:\Users\tonykuo\projects\lead-reproduction\.uv-python'
uv sync --group m5
.\.venv\Scripts\python.exe scripts\run_m5_phaseC_tabpfn_spike.py --max-fit-rows 1000 --max-val-rows 1000 --tabpfn-batch-size 256 --allow-tabpfn-failure
```

`uv sync --group m5` resolved `tabpfn==8.0.8` and `torch==2.12.1+cpu` in this
environment.

## Evidence

+ Full M3 downsample shape: `4,285,104 x 137`.
+ Documented TabPFN-3 fit: full M3 downsample does not fit the `1,000,000 x 200`
  limit.
+ Local reduced table: `1,000 x 137` train rows, `1,000` validation rows,
  batch size `256`, CPU.
+ Hardware: `nvidia-smi` sees `NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB`,
  but installed torch is CPU-only (`2.12.1+cpu`, `cuda_available=false`).
+ GBDT anchor on the local reduced table: AUC `0.986955266955267`, precision
  `0.5196078431372549`, recall `0.9636363636363636`, F1
  `0.6751592356687898`, fit+predict `0.7817001000512391` seconds.
+ TabPFN metric and latency: not measured. `TABPFN_TOKEN` was not set, and the
  noninteractive runner does not launch the browser license flow.

## Current state

Unknown 19 is partially resolved: the full downsample shape exceeds documented
TabPFN-3 limits, while the reduced CPU spike table fits. Unknown 20 is
partially resolved: local hardware and package state are recorded, but license
setup is missing. Unknown 21 remains blocked on license/token setup because no
TabPFN fit+predict wall-clock was produced.

ADR 0015 remains Proposed until a local TabPFN run completes.

## Next step

Accept the Prior Labs license and provide `TABPFN_TOKEN` or cached local weights,
then rerun the same script. If GPU execution is required, install a CUDA-enabled
torch build in the optional M5 environment before rerunning.
