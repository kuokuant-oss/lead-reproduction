# M5 E6 — cost model

Every figure is measured or read from an existing artifact. Nothing is
extrapolated from E5's 192-row inference time, which is dominated by fixed
overhead and would badly misstate a 10.1M-row pass.

## Measured inputs

| Quantity | Value | Source |
|---|---:|---|
| TabPFN throughput, steady state | **330 rows/s** | `m5_tabpfn_137_batch_runner.log` second half, and `throughput_rows_per_second_per_gpu` recorded in `m5_tabpfn_137_remaining_batch_plan.json` |
| TabPFN throughput, all-in | 82.4 rows/s | same log across the whole 7.35 h span, including warm-up and stalls |
| Fixed-tree throughput | **309,483 rows/s** | synthetic 500,000 × 137 float32 rows through the four-model ensemble on the laptop; no real holdout row scored |
| Weighted-AUC sweep | 23 ms per draw | measured on the real 594,318-row co-primary subset with synthetic scores |
| Microbatch / checkpoint | 20,000 rows, **516 per state** | `m5_tabpfn_137_remaining_batch_plan.json`; the 516 is counted from `e6_microbatch_manifest.json` |

**Caveat that matters.** The 330 rows/s anchor was measured at **context
100,000**. E6 uses **context 20,000**. TabPFN attends over its context, so E6
should be faster — but by how much is unmeasured, and this audit refuses to
assume a factor. Treat 330 rows/s as an *upper bound on time*. Pinning the real
rate needs a throughput probe, which can run on replicated non-holdout rows and
produce no holdout prediction. That is an open item.

## Per-state cost

| | Steady (330 rows/s) | All-in (82.4 rows/s) |
|---|---:|---:|
| one state, one full pass | **8.53 h** | 34.17 h |

## Repeat policies

| Policy | `predict_proba` calls | Row scores | Wall clock (steady) | Output (float32) |
|---|---:|---:|---:|---:|
| **R1** — 24 × 1 pass | 12,384 | 243,291,720 | **204.8 h ≈ 8.5 d** | 0.97 GB |
| **R8** — 24 × 8 passes | 99,072 | 1,946,333,760 | **1,638 h ≈ 68.3 d** | 7.79 GB |
| **R1_PLUS_SENTINEL** | 12,384 + 192 sentinel | 243,291,720 | **204.8 h ≈ 8.5 d** | 0.97 GB |

At the all-in rate those become 34 d, 273 d, and 34 d. R8 is not a schedule
anyone will run; it is priced here so the comparison is on the record rather
than asserted.

`R1_PLUS_SENTINEL` adds, after each of the 24 reloads, 8 repeats on a fixed
352-row sentinel — about 70 s in total across the whole run, which does not move
the wall clock. Sentinel results are kept out of every full-holdout endpoint.

### What each policy can and cannot support

| | R1 | R8 | R1_PLUS_SENTINEL |
|---|---|---|---|
| estimates same-state inference variation **on the holdout** | no | yes | no |
| detects inference drift or a degraded reload | no | yes | **yes**, on the sentinel |
| clustered interval conditions on | one realization per state | the mean of 8 realizations | one realization per state |
| matches the E4/E5 estimand | no — E4/E5 averaged 8 repeats per fit | yes | no |
| paper wording permitted | must state the interval covers row resampling only | may state repeats are inside | must state the interval covers row resampling only, with the sentinel as a lifecycle check |

## Other costs

| Item | Cost |
|---|---:|
| raw F4/137 rebuild, 10,137,155 × 137 float32 | **5.56 GB**, built once, shared by all 24 states |
| 24 fixed-tree comparators over the whole holdout | **0.22 h ≈ 13 min** |
| clustered intervals, 1000 draws × 2 clusterings × 24 units | **0.31 h ≈ 19 min** |
| archive of R1 outputs | ~1 GB, minutes to transfer |
| worst-case recompute after a failure | **one complete state pass = 10,137,155 rows ≈ 8.53 h** |

The tree half is about 0.1% of the TabPFN cost. It is not a scheduling concern,
which is worth stating plainly because the E5 override made it look like the
awkward half.

## Recommendation

**R1_PLUS_SENTINEL**, single recommendation, for three reasons.

**Same-state inference variation is already measured, twice.** E3 measured it at
one context on 352 rows; E4 measured it across 24 fits; E5 measured it again
across 24 reloads on an independent query, where the co-primary half-widths were
0.0004–0.0098 (AUC) and 0.0003–0.0064 (margin). None of those runs found it
anywhere near the effect being tested, which is +0.40 AUC and +0.60 margin. A
third measurement at 8× the cost of the entire experiment buys a quantity we
have three independent estimates of.

**E6's open question is prevalence, not inference noise.** E4 and E5 both scored
balanced or stratified queries. What is untested is whether the response holds
when hotwater negatives outnumber steam positives 11:1 and overall prevalence is
6.3%. That is answered by one clean pass per state; repeating the pass eight
times does not make the prevalence question better answered.

**R8 costs 68 days of exclusive GPU time.** At the all-in historical rate it is
nine months. That is not a trade against R1's 8.5 days; it is a different
project.

The sentinel is the part that must not be dropped. R1 alone cannot distinguish a
healthy reload from a degraded one, because there is nothing to compare a single
realization against. Eight repeats on a fixed 352-row sentinel after each reload
costs about 70 seconds total and gives a per-state lifecycle check with a known
expected range from E4/E5. Sentinel results never enter a full-holdout endpoint.

### The wording constraint R1 imposes

With one realization per state, the clustered interval covers **row resampling
given that realization**. It does not contain same-state inference variation,
and the report must say so rather than let the reader assume E4/E5's estimand
carried over. E4/E5 averaged 8 repeats per fit before forming contrasts; E6
under R1 does not, and that difference belongs in the methods, not in a
footnote.

## Corrections to the first audit

Two figures in the original audit were wrong and are superseded here.

**Call census.** The audit divided 10,137,155 by 20,000 and reported 12,165
full-holdout calls. Every one of the 12 shards ends in a short microbatch, so
the true count is a sum of ceilings: **516 microbatches per state, 12,384 calls
across 24 states**, and **99,072** for R8 rather than 97,317. All call counts are
now derived programmatically from `e6_microbatch_manifest.json`, and a test
rejects the superseded numbers by value so they cannot drift back.

**Failure recomputation.** The audit wrote that a failure costs one 20,000-row
microbatch. That is the progress granularity, not the scientific one. A state is
one canonical single-process batched pass, so a process failure quarantines the
partial state and the state restarts from canonical row 0. The correct figure is
**one complete state pass — 10,137,155 rows, about 8.53 h**. Completed states are
skipped on restart; the failed one is not resumed mid-way.
