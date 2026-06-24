# M4 Plan: Importable Pipeline Foundation

**Status**: M4.0/M4.1 done; M4.2-M4.5 pending
**Started**: 2026-06-24
**Reference**:

+ M3 exit line: `docs/m3-plan.md`
+ Current foundation review: duplicated M3 logic in `scripts/` and empty
  `src/lead/`
+ Golden metrics: `data/processed/*.json`, `docs/*.json`
+ Design rule: when paper, docs, and executable code disagree, executable code
  is the reproduction authority unless an ADR explicitly changes it.

---

## M4 Goal

M4 turns the current notebook/script-centered pipeline into an importable
`src/lead/` package. The package must preserve M3 behavior first, then create
the stable surface needed for M5 FDD on BDG2.

M4 is a foundation milestone, not a new modeling milestone. M4.0 and M4.1 lock
the current behavior before any semantic fixes. M4.2 and M4.3 are where the
known label-join and timestamp-shift defects may be corrected behind explicit
regression gates.

Golden regression targets:

| Metric | Target |
|---|---:|
| M3.2 LightGBM 80/20 offline AUC | 0.9920 |
| M3.4 4-model ensemble 80/20 offline AUC | 0.9928 |
| 50/50 offline ensemble AUC | 0.9921 |
| 50/50 causal ensemble AUC | 0.9911 |
| Site-held-out ensemble AUC | 0.9774 |
| Steam meter AUC | 0.9553 |
| Regression noise floor | +/- 0.0005 |

---

## Milestones

### M4.0: Baseline Lock

**Status**: Done

**What**: Archive the current golden metrics in a tracked test fixture with
source filenames and commit provenance.

**Done when**:

+ [x] `tests/golden_metrics.json` exists and records M3.2, M3.4, 50/50,
  site-held-out, and steam AUC targets.
+ [x] Each metric has `value`, `source_file`, and `source_commit`.
+ [x] The fixture records the regression noise floor `0.0005`.
+ [x] No pipeline code is changed in this slice.

**Out of scope**:

+ Refactoring scripts or `src/lead/`
+ Re-running experiments
+ Changing any metric value by interpretation

**Labels**: `m4`, `foundation`, `regression-gate`
**Priority**: P0
**Depends on**: M3 closed artifacts

---

### M4.1: Extract Importable `src/lead` Package

**Status**: Done

**What**: Move duplicated M3 helper logic into importable modules while
preserving current numeric behavior.

**Done when**:

+ [x] `src/lead/data.py`, `features.py`, `split.py`, `sample.py`,
  `evaluate.py`, and `io.py` exist.
+ [x] `src/lead/__init__.py` exports the public API used by scripts.
+ [x] `scripts/run_m3_4_ensemble.py`, `scripts/run_m3_split_causality.py`,
  `scripts/run_m3_50_50_ensemble.py`, and `scripts/run_m3_3_budslab.py` import
  helpers from `lead` instead of carrying local copies.
+ [x] `add_value_change_features(df, shifts)` is the single public signature.
+ [x] M3.2 and M3.4 reruns differ from `tests/golden_metrics.json` by less
  than +/- `0.0005`.
+ [x] `data/processed/m4_1_refactor_check.json` records the regression
  comparison and provenance.
+ [x] `tests/test_refactor_regression.py` asserts golden metric compatibility.

**Out of scope**:

+ Fixing positional label join
+ Replacing `groupby().shift()` with timestamp merge
+ Removing `StandardScaler`
+ Changing downsample semantics
+ BDG2 data ingestion

**Labels**: `m4`, `refactor`, `src-lead`, `behavior-preserving`
**Priority**: P0
**Depends on**: M4.0

---

### M4.2: Key-Aligned Label Join

**Status**: Pending

**What**: Replace positional M3 label assignment with an explicit key-aligned
join or a documented invariant check that proves positional files remain safe.

**Done when**:

+ [ ] Label alignment no longer relies only on equal row count.
+ [ ] M3.2 regression is measured against M4.0 golden metrics.
+ [ ] Any AUC movement beyond +/- `0.0005` is explained in an ADR/update.
+ [ ] A focused test fails on row-order mismatch before the fix and passes
  after the fix.

**Out of scope**:

+ Timestamp value-change changes
+ Model hyperparameter changes
+ BDG2 labels

**Labels**: `m4`, `data-integrity`, `label-join`
**Priority**: P1
**Depends on**: M4.1

---

### M4.3: Timestamp-Based Value-Change Regime

**Status**: Pending

**What**: Add timestamp-merge value-change features as an explicit regime and
evaluate whether the current row-offset approximation moves M3.2 beyond the
noise floor.

**Done when**:

+ [ ] `add_value_change_features` supports the current row-offset regime and a
  timestamp-merge regime.
+ [ ] M3.2 AUC change is measured against `0.9920` with +/- `0.0005` gate.
+ [ ] The M4 unknown about timestamp merge is resolved or carried with
  evidence.
+ [ ] Offline and causal regime naming remains compatible with ADR 0007.

**Out of scope**:

+ Label-join changes beyond M4.2
+ Model changes
+ BDG2 transfer

**Labels**: `m4`, `feature-semantics`, `timestamp-merge`
**Priority**: P1
**Depends on**: M4.2

---

### M4.4: Dead-Code and Sampling Semantics Review

**Status**: Pending

**What**: Decide whether to remove or preserve StandardScaler and the current
positive-duplication downsample pattern.

**Done when**:

+ [ ] The current `StandardScaler` behavior is either removed with regression
  proof or documented as intentionally preserved compatibility code.
+ [ ] The downsample pattern `[negs1, pos, negs2, pos]` is documented as
  current behavior or replaced behind a regression gate.
+ [ ] Any behavior change has M3.2 and M3.4 regression evidence.

**Out of scope**:

+ Timestamp value-change
+ New sampling strategies for BDG2
+ Hyperparameter tuning

**Labels**: `m4`, `cleanup`, `sampling`, `dead-code`
**Priority**: P2
**Depends on**: M4.3

---

### M4.5: M5 Readiness Gate

**Status**: Pending

**What**: Freeze the importable API and document the extension points needed by
M5 FDD on BDG2.

**Done when**:

+ [ ] `src/lead` public API is listed in a handoff or README section.
+ [ ] M3.2/M3.4 golden gates still pass.
+ [ ] Remaining M4 unknowns are either resolved or explicitly out of M5 scope.
+ [ ] M5 entry criteria name the expected data, label, split, and evaluation
  interfaces.

**Out of scope**:

+ Downloading or processing BDG2
+ Running FDD experiments
+ Changing M3 headline metrics

**Labels**: `m4`, `m5-readiness`, `api-freeze`
**Priority**: P1
**Depends on**: M4.4

---

## Out of M4 Scope

+ Downloading, preprocessing, or modeling BDG2
+ FDD-specific feature engineering
+ New model families
+ Hyperparameter tuning
+ Changing the M3.4 headline result
+ Rewriting notebooks for presentation
+ Closing M4.2+ issues during M4.0/M4.1

---

## Issue Tracker Map (M4)

| Slice | Proposed issue | Status |
|---|---|---|
| M4.0 baseline lock | TBD | Done |
| M4.1 extract `src/lead` | TBD | Done |
| M4.2 key-aligned label join | TBD | Pending |
| M4.3 timestamp value-change | TBD | Pending |
| M4.4 dead-code/sampling review | TBD | Pending |
| M4.5 M5 readiness gate | TBD | Pending |

---

## M4 Exit Criteria

+ [ ] All duplicated M3 foundation helpers are removed from tracked scripts.
+ [ ] At least one tracked test asserts golden metric compatibility.
+ [ ] Known P1-P4 issues are either fixed with evidence or explicitly deferred
  with ADR/unknown coverage.
+ [ ] `src/lead` is importable without path hacks from the project root.
+ [ ] M5 can import data, feature, split, sample, and evaluation helpers without
  reading notebook cells.
