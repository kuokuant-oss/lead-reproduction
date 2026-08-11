# M5 Building-Count V2 Seeds 47–51 Runbook

## Purpose

This runbook is the handoff contract for running only building seeds 47–51 on
a separate computer. The bundle preserves the seed 42–46 scientific protocol
while changing only `building_seed`.

Do not regenerate or modify the selected building identities before the formal
run. The tracked ladder manifests are the authoritative inputs.

## Frozen experiment contract

- Building seeds: `47 48 49 50 51`
- Budgets: `10 20 50 100`
- Formal order: every seed at K10, then every seed at K20, K50, and K100
- Models per seed/K pair: Tree first, then TabPFN
- Sampling: NumPy PCG64 site-stratified random sampling without replacement
- Feasibility: whole-ladder rejection sampling; no greedy correction
- Identity inputs: `building_id` and `site_id`
- Feasibility-only inputs: meter presence
- Anomaly labels, anomaly rate, row count, size, and Tree role do not choose identity
- `row_seed=42`, `model_seed=42`, `role_seed=None`
- `average_building_cap=500`, maximum context rows `50,000`
- Tree: frozen fixed iterations, no early stopping
- TabPFN: 8 estimators
- Tree and TabPFN must pass the matched-context gate for every pair

The formal queue contains 20 seed/K pairs and 40 model units.

## Tracked files supplied by this bundle

- `experiments/m5_building_count_v2_seed47_51/audit/`
  - five JSON manifests and five human-readable ladder CSV files
  - `summary.json`, composition, overlap, and prefix audits
  - the frozen candidate profile used to reproduce the ladders
- `experiments/m5_building_count_v2_seed47_51/canonical_holdout_identity.npz`
  - compact canonical holdout identity arrays required by every model cell
- `scripts/prepare_m5_building_count_v2_seed47_51.py`
  - validates the bundle and atomically installs the canonical holdout identity
- `scripts/run_m5_building_count_v2_seed47_51.py`
  - dedicated safe launcher; defaults to plan mode
- `docs/reports/m5-building-count-experiment_V2_seed47-51.md`
  - separate tracked progress report used by automatic publication

Existing generic Tree, TabPFN, checkpoint, aggregation, and supervisor scripts
remain shared with V2. The dedicated launcher pins their arguments.

## Required raw data

The following files must already exist on the target computer:

```text
data/raw/m3/train.csv
data/raw/m3/bad_meter_readings.csv
data/raw/m3/building_metadata.csv
data/raw/m3/weather_train.csv
```

The large historical M6 prediction artifact is not required. The bundle carries
only its four canonical identity arrays, without historical model scores. The
preparation script verifies their holdout-row SHA256 and installs the compact
artifact at the path consumed by the V2 model cells.

## 1. Clone and install

From the repository root:

```bash
git pull --ff-only
uv sync --frozen
```

Confirm CUDA and the Python environment before doing any model work:

```bash
nvidia-smi
.venv/bin/python -c "import numpy, pandas, sklearn, tabpfn; print('imports ok')"
```

## 2. Prepare and verify prerequisites

This command does not launch a model. If the canonical processed artifact is
missing, it copies the tracked compact identity artifact using an atomic write.
If an artifact already exists, it validates it and does not overwrite it.

```bash
.venv/bin/python scripts/prepare_m5_building_count_v2_seed47_51.py \
  --mode prepare-canonical
```

Expected final JSON fields include:

```text
"status": "ready"
"building_seeds": [47, 48, 49, 50, 51]
"budgets": [10, 20, 50, 100]
"holdout_row_sha256": "6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893"
```

Run the focused tests:

```bash
.venv/bin/python -m unittest \
  tests.test_m5_building_count_v2_seed47_51 \
  tests.test_m5_building_count_v2_overnight \
  tests.test_m5_building_count_v2
```

## 3. Inspect the non-launching plan

```bash
.venv/bin/python scripts/run_m5_building_count_v2_seed47_51.py --mode plan
```

The plan must report `pair_order_policy: budget-major`, 20 pairs, and 40 units.
The pair order must be exactly:

```text
K10:  seed47 seed48 seed49 seed50 seed51
K20:  seed47 seed48 seed49 seed50 seed51
K50:  seed47 seed48 seed49 seed50 seed51
K100: seed47 seed48 seed49 seed50 seed51
```

Within each pair, Tree runs before TabPFN.

## 4. Run bounded non-scientific validation

Validation uses isolated outputs and deterministic 200-row caps for context and
holdout. It does not publish results.

```bash
.venv/bin/python scripts/run_m5_building_count_v2_seed47_51.py \
  --mode validation \
  --validation-context-rows 200 \
  --validation-holdout-rows 200
```

Do not start the formal run unless validation exits successfully and its
matched-context gate checks all 20 seed/K pairs.

## 5. Choose a Git publication strategy

The formal supervisor can commit and push the separate progress report after
every completed pair. Do not let two computers publish to the same branch at
the same time.

If the seed 42–46 computer is still publishing to `main`, use a dedicated
results branch on the seed 47–51 computer:

```bash
git switch -c m5-v2-seed47-51-results
git push -u origin m5-v2-seed47-51-results
```

If seed 42–46 has completely finished and `main` is synchronized, publication
may remain on `main`. In either case, the tracked worktree must be clean before
formal launch. Unrelated user files must not be staged or deleted.

```bash
git status --short --branch
```

## 6. Formal launch

Formal execution requires an explicit instruction from the operator after the
bounded validation is reviewed. If an AI agent only discovers this runbook but
has not been told to start the formal run, it must stop here and report that it
is ready.

After authorization, run in a persistent terminal or `tmux` session:

```bash
.venv/bin/python scripts/run_m5_building_count_v2_seed47_51.py \
  --mode formal \
  --publish-results
```

The launcher forces `budget-major`; do not replace it with the generic
overnight command unless the same `--pair-order budget-major`, audit root,
output root, and report path are supplied.

## 7. Monitoring

Primary status:

```text
data/processed/m5_building_curve/v2/
  building_seed_sweep_47-48-49-50-51/overnight/status.json
```

Active cell heartbeat:

```text
data/processed/m5_building_curve/v2/
  building_seed_sweep_47-48-49-50-51/model_runs/
  building_seed<SEED>/<MODEL>_k<K>_f137/heartbeat.json
```

Supervisor events and failure markers:

```text
.../overnight/events.jsonl
.../overnight/failed_stages/
.../overnight/failed_publications/
.../overnight/FAILED.json
```

Read these files without editing them. Do not delete checkpoints, partial
chunks, failure markers, or `COMPLETE.json` files to force progress.

## 8. Resume after interruption

Rerun the exact same formal command. Every cell receives `--resume`; completed
valid checkpoints and pairs are reused. Provenance mismatches hard-fail rather
than silently mixing results.

Never use `git reset --hard`, remove model output directories, or restart a
still-running model process.

## 9. Completion criteria

The extension is complete only when all of the following exist and validate:

- 40 model-unit `COMPLETE.json` files
- 20 pair-gate JSON records with `passed: true`
- `overnight/COMPLETE.json` with no failed units or publications
- the final aggregate report and matched-context gate
- the tracked seed 47–51 report committed and pushed to the intended branch

The final ten-seed analysis combines this extension with the existing seed
42–46 sweep; raw model artifacts remain in their separate sweep roots.
