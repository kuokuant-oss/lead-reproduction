# M5 hotwater factorial independent-query replication

## Frozen sampling contract

Before reading any independent predictions, the query builder excluded every
raw row and every building in the original 352-row query. It then sampled three
strata from odd buildings using deterministic raw-index priorities:

| Stratum | Rows | Buildings | Anomaly segments |
| --- | ---: | ---: | ---: |
| HW 0–1 positive | 64 | 34 | 52 |
| HW 0–1 negative | 64 | 34 | 64 |
| Steam positive | 64 | 40 | 58 |

Every stratum has a maximum of two rows per building and two rows per anomaly
segment. The query has 192 rows and excludes all 142 original-query buildings.
Its manifest and row audit are in
`data/processed/m5_hotwater_label_factorial/independent_query/`.

## Current gate lock

The first-round runner lacked portable states, so the authorized exact-design
recovery refit rebuilt and saved them. That refit was then checked on the
original 352-row query before this query could be used. Its TabPFN reproduction
gate failed: all 24 TabPFN cells missed the predeclared score gate and 16 also
missed a primary-estimand gate, while trees passed. Therefore this independent
query remains unscored.

No independent-query prediction, metric, factorial effect, or replication
decision has been fabricated. The only remaining active diagnostic is a no-fit
reload verification of the saved states against the recovery's own screening
scores. Until reproducibility is resolved, Path B remains deferred and
`recall@FPR=.001` is excluded from independent-query readouts.

The no-fit reload verification has now completed and shows that tree state
reload is exact but TabPFN portable-state inference is not stable enough for
the predeclared gate. Consequently this query is permanently unscored for the
current Path-A round. Its sampling contract and artifact remain available for a
future, separately authorized deterministic-execution protocol; they must not
be reused to make a post-screening claim now.
