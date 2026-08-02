# M5 E6 — design audit

Read-only audit of the inputs E6 would run on. Nothing was fitted, and no model
was asked to predict on any holdout row.

## What E6 would ask

> Does the steam negative-support response established in E4 and independently
> replicated in E5 still hold on the complete odd-building holdout at natural
> prevalence?

Two verdicts, as in E5: response confirmation, and TabPFN-specific confirmation.
The primary contrast, the two co-primary endpoints, the cell coding, the effect
formulas and signs, the three context seeds, the two scaler arms, the 24
persisted states, the matched fixed trees, and both cluster definitions are all
inherited unchanged. No model-seed factor, no refit, no endpoint re-selection.

## Full-holdout identity — verified

| | |
|---|---|
| rows | 10,137,155, all unique |
| buildings | 724, **every one odd** |
| sites | 16 |
| anomaly rows | 637,397 |
| natural prevalence | 6.288% |
| sorted `raw_index` SHA-256 | `f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76` |
| disjoint from the fit half | yes — the split rule is `building_id % 2`, and no even building appears |
| agreement across artifacts | **10 full-test artifacts share one row set and one order** |

The stored order is *not* sorted `raw_index`. That stored order is the canonical
position, and all ten artifacts agree on it, so E6 must adopt it rather than
re-sorting.

Only `raw_index`, `building_id`, `site_id`, `anomaly` and the `meter` feature
column were read. The `tabpfn` score column present in those artifacts was never
opened, and the row manifest records that explicitly.

### Required wording

E6 is a **natural-prevalence factorial confirmation using new factorial states
on previously characterised holdout rows**. It is not an untouched holdout, not
a first contact, and not a previously unseen row set: these rows already carry
context-curve predictions. What is new in E6 is the states.

## Strata at natural prevalence

| Meter | Rows | Anomaly | Prevalence | Buildings | Sites |
|---|---:|---:|---:|---:|---:|
| electricity | 6,035,071 | 356,679 | 5.91% | 709 | 16 |
| chilledwater | 2,115,354 | 141,139 | 6.67% | 252 | 10 |
| steam | 1,350,609 | 48,888 | 3.62% | 162 | 6 |
| hotwater | 636,121 | 90,691 | 14.26% | 73 | 7 |

**The co-primary contrast needs only a small part of the holdout:**

| | |
|---|---:|
| steam positives | 48,888 (150 buildings, 6 sites) |
| hotwater negatives | 545,430 (73 buildings, 7 sites) |
| co-primary subset | **594,318 rows = 5.86% of the holdout** |
| buildings in the subset | 215 |

This is the single most consequential fact in the audit. Scoring is a 10.1M-row
problem; the co-primary *statistics* are a 594k-row problem.

## Existing feature artifacts are not reusable

`m5_tabpfn_137_distributed_context100000/{head,tail}/features.float32.npy`
(5,060,000 × 137 and 5,077,155 × 137, 5.3 GB, digest-verified) are **already
scaled** with the context-100000 scaler: the `meter` column holds
−0.7625 / 0.3011 / 1.3647 / 2.4283 rather than 0 / 1 / 2 / 3.

E6 applies 24 different per-unit scalers, so it needs the **raw** F4/137
matrix. That must be rebuilt: 10,137,155 × 137 float32 = **5.56 GB**, built once
and shared by all 24 states. The existing artifacts remain useful for row
identity and metadata, which is what this audit used them for.

## Cluster structure, and a problem to rule on

On the co-primary subset:

| Clustering | Clusters | Note |
|---|---:|---|
| building | **215** | as in E4/E5 |
| segment | **594,297** | of which **545,430 (91.8%) are singletons** |

Segments are contiguous anomaly runs; every non-anomaly row is its own cluster.
All 545,430 hotwater negatives are non-anomaly, so at natural prevalence the
segment estimator is **91.8% singletons on this contrast** and behaves close to
a row bootstrap on the negative side. Its interval will be far narrower than the
building interval and will not represent within-building correlation.

The estimator still extends exactly — this is not a feasibility problem. It is
an interpretation problem, and per the protocol it is raised for a ruling rather
than silently removed, swapped for a row bootstrap, or replaced with site
clusters. See `m5-e6-statistics-scalability.md`.

## What was not done

No `predict` on any holdout row. No fit, no refit, no protocol freeze, no
launch, no remote deployment, no tmux. The tree throughput figure was measured
on synthetic random rows, which produce no holdout prediction.
