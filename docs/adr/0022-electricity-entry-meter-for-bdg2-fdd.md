# Use electricity as the BDG2 FDD entry meter

## Status

Accepted (2026-06-30)

## Context

ADR 0019 fixes the Phase E evaluation paradigm: apply a GEPIII-trained detector
to BDG2 as unlabeled transfer evidence, not BDG2-native supervised detection.
ADR 0020 defines the next usable output as within-context evidence packets and
audit-yield measurement. ADR 0021 demotes the prior powered multi-building gate
to reporting confidence so the packet path can proceed when BDG2-only
sufficient-observation evidence exists.

The prior Phase E Step 4 investigation centered on chilledwater because the
Stage 1 weather-load diagnostic found strong chilledwater temperature-load
alignment. The BDG2 EDA then showed chilledwater is not a good entry meter for
the first within-context transfer/FDD path: only 26 BDG2-only buildings have
chilledwater columns, and only 3 satisfy the sufficient-observation rule.

Electricity has much broader BDG2-only coverage: the EDA records 151 BDG2-only
buildings with electricity and 99 sufficient-observation BDG2-only electricity
buildings. It also has the GEPIII meter-code bridge needed for the
GEPIII-trained detector path.

Miller et al. 2020 Fig 5 adds an important nuance. Electricity weather
sensitivity is heterogeneous: a significant subset of electricity meters are
temperature-correlated, while others are more occupancy-driven or otherwise less
temperature-synchronous. Therefore ADR 0020 Level-3 `weather_response` evidence
is partially usable for electricity on a per-meter basis. It must be evaluated
per meter, not assumed usable or unusable wholesale.

## Decision

Make electricity the entry meter for the first Phase E transfer/FDD
within-context scoring path.

Chilledwater support remains intact. Chilledwater stays in script meter choices,
the loader, and the plans, but it is deferred to a later Level-3
weather-conditioned path rather than used as the first entry meter.

Unknown #25 remains open and is reframed as a per-site/per-meter
weather-feature-validity caveat. It is not an electricity-wide Level-3
disqualifier, and it is not resolved by this ADR.

This does not change the transfer paradigm. GEPIII-trained detector transfer to
BDG2 remains fixed under ADR 0019.

This does not touch the M3 numeric line. `load_m3_frame` defaults, M3.2/M3.4
golden values, the regression gate, downsample semantics, and the StandardScaler
fit path remain unchanged.

This preserves and carries TabPFN forward. ADR 0020's TabPFN roles remain:
offline audit re-ranking, model-disagreement diagnostics, active-learning
audit-set selection, and few-shot calibration after a human review set exists.
Electricity-first within-context candidates make those audit roles reachable
without turning TabPFN into the full-corpus scanner.

## Rationale

A first transfer/FDD scoring path needs enough BDG2-only coverage to produce
reviewable within-context candidates. Electricity has the broadest BDG2-only
coverage and sufficient-observation count among the meter types measured in the
EDA. Chilledwater remains scientifically useful for weather-conditioned evidence,
but its current BDG2-only coverage makes it a poor entry meter.

The Miller et al. Fig 5 nuance prevents the opposite overcorrection. Electricity
is not disqualified from weather-response evidence. Instead, Level-3
weather-response support should be evaluated per meter and attached only where
the local electricity meter shows an adequate temperature relationship.

## Consequences

+ Phase E transfer scripts default to electricity while retaining chilledwater
  as an allowed meter.
+ M6.1 remains blocked on A4's raw-first correctness precondition; A2 does not
  run a full electricity scan.
+ Chilledwater is deferred, not deleted. Later chilledwater work can use the
  Level-3 weather-conditioned path after the entry electricity workflow exists.
+ Unknown #25 stays open as a weather-feature-validity caveat for electricity
  Level-3 evidence.
+ No BDG2 supervised metrics, `confirmed` status, absolute cross-dataset top-K,
  label fabrication, M3 numeric-line changes, or TabPFN full-corpus scanner role
  are introduced by this ADR.
