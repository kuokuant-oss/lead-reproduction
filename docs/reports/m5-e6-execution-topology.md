# M5 E6 — execution topology

## Unitization: two candidates

### A. State-major

One fresh process loads one state and streams the whole holdout through it, then
commits.

- 24 processes, 24 reloads
- one state's 10.1M scores come from **one process and one realization** — the
  clean stochastic semantics
- a failure at 90% of a state costs up to 8.5 h of that state
- output is naturally one vector per state

### B. Row-shard-major

One row shard is scored by all 24 states in turn.

- 24 × N_shards reloads (12 shards → 288 reloads)
- each shard is self-contained across cells, seeds and arms, so a shard can be
  analysed alone
- a failure costs at most one shard × one state
- but **a state's full vector is then stitched from 12 separate processes**

## The decisive difference

Under B, a state's 10.1M-row vector is assembled from 12 process realizations.
E3, E4 and E5 all established that repeated inference on one fitted TabPFN state
is not bitwise reproducible, so those 12 pieces are 12 different draws from the
same state's inference distribution. Concatenating them produces a vector that
is **no single realization**, while looking exactly like one.

That is precisely what the protocol forbids: a resume rule may not stitch parts
of one state, scored in different unrecorded processes, into something presented
as a single realization.

B can be made legitimate — by recording the process identity of every batch and
declaring the result a *mosaic* rather than a realization — but that changes the
estimand, and it changes it silently unless every downstream report repeats the
qualification.

**Recommendation: A, state-major**, with per-microbatch atomic checkpoints
recording the owning process UUID. A state's batches must all carry the same
UUID, and the importer must refuse a state whose batches do not. That keeps
"one state, one realization" true by construction instead of by convention,
which is what makes it checkable.

The failure cost is the honest objection: up to 8.5 h lost. It is bounded by
resuming *within* a state only if the resumed process re-scores every batch it
did not itself write — that is, a resume restarts the state. At 8.5 h per state
and 20,000-row checkpoints for progress reporting, that is acceptable; the
alternative buys cheaper restarts with a permanently ambiguous estimand.

## Machine topology

### TabPFN — one GPU host, recommended

gpu-host, TabPFN 8.0.8, RTX 5070 Ti, torch 2.12.1+cu130, CUDA 13.0, checkpoint
`d0d865d5…f3988`, CPU fallback prohibited.

Multi-machine is **not** recommended as the default, and the reason is the same
one that decided E4. If more than one GPU host is used:

- shard by **row**, never by state; a machine that scored only some states makes
  "machine" confounded with every state contrast
- every machine scores **all 24 states** for its shard
- the laptop and gpu-host differ in CUDA build (cu126 vs cu130) and GPU
  architecture (Ada vs Blackwell), and E5 demonstrated that such differences are
  real, not hypothetical: the same tree ensembles differed by a mean of 8.1e−03
  across those two environments
- before merging shards, an **overlap-shard gate** is required: give two machines
  the same 20,000-row overlap shard, score all 24 states on both, and require
  agreement at a stated tolerance. Without that gate, cross-shard merging has no
  evidence behind it

Until such a gate has been run and passed, the recommendation stays single-host.
At 8.5 days for R1 that is tolerable; buying 4 days by introducing an unverified
cross-machine seam into the final confirmation stage is not a good trade.

### Trees — the laptop, as in E5

The fixed comparators reproduce E4's frozen 352-row vector bit for bit only in
the environment that fitted them: Windows 11, Python 3.13.13, lightgbm 4.6.0,
xgboost 3.2.0, catboost 1.2.10, sklearn 1.8.0, numpy 2.4.6, joblib 1.5.3.

The same 24/24 bit-exact gate E5 used must pass before any holdout tree scoring,
with `max_abs_diff == 0` and 352/352 rows exact. No refit. No gpu-host tree
output.

Measured: **309,483 rows/s** on synthetic input, so 24 comparators over the
whole holdout is about **13 minutes**, with under 1 GB RSS at a 500,000-row
batch. The tree half is not a scheduling concern.

## Streaming and scaler flow

```
raw F4/137 batch (float32)
  -> the unit's exact scaler   (frozen_reference: load + verify digest)
                               (cell_specific: rebuild, then verify against
                                the state's scaled X_train)
  -> float32                   (any float64 upcast is a hard failure)
  -> state or tree inference
  -> atomic batch output
```

Two dtype lessons from E5 are carried in as hard failures rather than comments:
E4 scaled float32, so an upcast copy does not reproduce it in the last bit; and
a gate that certifies one dtype path while scoring runs another produces vectors
the gate never checked.

A transformed batch may be cached by scaler identity, but caching may never
merge, skip or substitute a state. Cell 11's two arms share a scaler, so they
share a transform — and are still scored as two separate states, because they
are two separate states.

## Resume and fail-closed

- per-microbatch atomic writes, each recording the owning process UUID
- a state whose batches carry more than one UUID is refused at import
- `INTERRUPTED_INCOMPLETE` on failure; no silent retry, no reload-backfill
- a partially scored holdout may not produce any scientific decision — the
  existing batch plan shows per-batch prevalence ranging from 3.1% to 15.6%, so
  partial coverage is not a representative sample
