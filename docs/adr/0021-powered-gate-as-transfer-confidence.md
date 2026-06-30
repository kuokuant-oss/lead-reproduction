# Treat powered BDG2 transfer strata as confidence, not entry

## Status

Accepted (2026-06-30)

## Context

ADR 0019 selected GEPIII-trained detector transfer to BDG2 as the Phase E
evaluation paradigm. Because BDG2 has no native per-row anomaly labels, ADR 0019
also prohibited supervised BDG2 AUC, PR-AUC, precision, recall, and F1 claims;
required BDG2-only and GEPIII-overlap separation; and rejected `site_id % k`
inside one dataset as cross-dataset evidence.

Phase E Step 4 then implemented a powered-entry gate for chilledwater: a
BDG2-only sufficient-observation stratum had to reach at least 5 buildings and
17,544 rows before the next transfer step could proceed. The corrected
chilledwater pilot found 1 Fox BDG2-only sufficient-observation building, while
the pooled raw fallback found 3 BDG2-only sufficient-observation buildings and
52,632 rows. The result was reported as `underpowered_even_pooled` and stopped.

ADR 0020 subsequently reframed the next BDG2 output as within-context ranked
evidence packets and audit-yield measurement. Under that frame, the powered
multi-building threshold is useful as a confidence label, but it conflicts with
the packet path if treated as a blocking entry gate. A single BDG2-only building
can still produce reviewable within-`(building, meter)` or within-context
evidence, as long as the output does not become an absolute cross-dataset top-K
or supervised accuracy claim.

## Decision

Supersede only the powered-entry-gate clause of ADR 0019.

The 5-building / 17,544-row powered check remains measured, but it is now an
after-the-fact reporting-confidence dimension named
`multi_building_transfer_stability`. It does not block the within-context
evidence-packet path when BDG2-only sufficient-observation evidence exists.

The updated gate semantics are:

+ score plumbing failures, such as incomplete score coverage, still stop and
  require diagnosis;
+ absent BDG2-only sufficient-observation evidence still stops because there is
  no BDG2-only packet path to review;
+ one or more BDG2-only sufficient-observation buildings allow the
  within-context packet path to proceed;
+ the powered threshold is reported as `powered=true` or `powered=false`, with
  observed buildings and rows, so readers can distinguish single-building
  evidence from multi-building transfer stability.

This does not change the transfer paradigm. GEPIII-trained detector transfer to
BDG2 remains fixed.

This does not touch the M3 numeric line. `load_m3_frame` defaults, M3.2/M3.4
golden values, the regression gate, downsample semantics, and the StandardScaler
fit path remain unchanged.

This preserves and carries TabPFN forward. ADR 0020's TabPFN roles remain:
offline audit re-ranking, model-disagreement diagnostics, active-learning
audit-set selection, and few-shot calibration after a human review set exists.
Demoting the powered gate makes those audit roles reachable sooner because real
within-context candidates can be generated before multi-building transfer
stability is available.

All other ADR 0019 constraints remain in force:

+ no BDG2 supervised ground-truth AUC, PR-AUC, precision, recall, or F1;
+ no `confirmed` BDG2 fault status;
+ no absolute cross-dataset top-K;
+ BDG2-only and GEPIII-overlap outputs remain separated;
+ raw/cleaned pseudo-label metrics, if used later, remain secondary and must be
  explicitly labeled as pseudo-label metrics.

## Rationale

The prior powered-entry gate mixed two questions:

+ Can Phase E produce honest, reviewable BDG2 within-context evidence?
+ Does that evidence have enough multi-building breadth to support a stronger
  transfer-stability statement?

ADR 0020 answers the first question through evidence packets and matched
audit-yield evaluation. Multi-building breadth informs confidence in transfer
stability, but it is not required to create a review queue from a single
BDG2-only building. Keeping the threshold as metadata preserves the measurement
without blocking the first usable BDG2 FDD output.

## Consequences

+ Phase E is no longer stuck solely because chilledwater did not meet the
  multi-building powered bar.
+ Existing chilledwater Step 4 results remain valid as measurements:
  `underpowered_even_pooled` becomes a confidence description, not a dead-end
  verdict.
+ Future M6 evidence packets can proceed from single-building BDG2-only
  sufficient-observation evidence, while reporting whether
  `multi_building_transfer_stability.powered` is true or false.
+ A later slice must still choose the entry meter and raw/cleaned scoring path;
  this ADR does not make the A2 electricity decision, the A4 raw-first decision,
  the A5 value-change regime decision, or any M6 implementation change.
