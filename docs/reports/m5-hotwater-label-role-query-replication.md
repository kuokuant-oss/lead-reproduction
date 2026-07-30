# M5 hotwater factorial independent-query replication

## Frozen sampling contract

Before reading any independent predictions, the query builder excluded every
raw row and every building in the original 352-row query. It sampled three
strata with deterministic raw-index priorities:

| Stratum | Rows | Buildings | Anomaly segments |
| --- | ---: | ---: | ---: |
| HW 0–1 positive | 64 | 34 | 52 |
| HW 0–1 negative | 64 | 34 | 64 |
| Steam positive | 64 | 40 | 58 |

The query has 192 rows and excludes all original-query buildings, with capped
row contribution per building and segment.

## Status

The query is **frozen and unscored**. It cannot select the 8.0.8 pilot's
replicate count, endpoints, direction, analysis method, or stopping rules. The
former deterministic execution gate is no longer a scientific prerequisite;
instead, the query may be considered only after the predeclared repeated-
inference protocol is frozen by the variance pilot. No prediction, metric,
factorial effect, or replication decision has been fabricated.

No Path B, tree refit, new TabPFN version, full holdout, site-transfer, or
paper-manuscript change is authorized by this status.
