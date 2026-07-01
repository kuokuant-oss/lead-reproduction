# Use electricity as the BDG2 FDD entry meter

## Status

Accepted (2026-06-30)

Amended by ADR 0025 (2026-07-01): electricity remains the first BDG2 FDD meter,
but it is now scoped as the first labeled supervised-evaluation meter on the
GEPIII-overlap subset rather than only the first unlabeled transfer/audit meter.

## Context

At acceptance time, ADR 0019 framed Phase E as unlabeled transfer evidence, ADR
0020 framed the next output as audit evidence, and ADR 0021 demoted the prior
powered multi-building gate to reporting confidence. ADR 0025 later superseded
that primary paradigm for the GEPIII-overlap, 2016, meters-0-3 subset, where
bridged rank-1 GEPIII annotations make supervised metrics legitimate.

The prior Phase E Step 4 investigation centered on chilledwater because the
Stage 1 weather-load diagnostic found strong chilledwater temperature-load
alignment. The BDG2 EDA then showed chilledwater is not a good first meter for
the initial BDG2 FDD path: only 26 BDG2-only buildings have chilledwater
columns, and only 3 satisfy the sufficient-observation rule.

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

Make electricity the entry meter for the first Phase E transfer/FDD path.

Chilledwater support remains intact. Chilledwater stays in script meter choices,
the loader, and the plans, but it is deferred to a later Level-3
weather-conditioned path rather than used as the first entry meter.

Unknown #25 remains open and is reframed as a per-site/per-meter
weather-feature-validity caveat. It is not an electricity-wide Level-3
disqualifier, and it is not resolved by this ADR.

This paragraph is superseded by ADR 0025 for the primary M6 path. The selected
transfer object remains GEPIII-trained detector scoring on BDG2, but the
GEPIII-overlap, 2016, meters-0-3 subset now has bridged supervised labels.

This does not touch the M3 numeric line. `load_m3_frame` defaults, M3.2/M3.4
golden values, the regression gate, downsample semantics, and the StandardScaler
fit path remain unchanged.

This preserves and carries TabPFN forward. ADR 0025 keeps TabPFN in scope for
supervised M6 comparison and label-scarce/offline evaluation, with the Phase D
latency and license caveats attached.

## Rationale

A first transfer/FDD scoring path needs enough coverage to support a useful
first slice. Electricity has the broadest BDG2-only coverage and
sufficient-observation count among the meter types measured in the EDA, and it
also maps to GEPIII meter code `0` for the supervised overlap bridge.
Chilledwater remains scientifically useful for weather-conditioned evidence,
but its current BDG2-only coverage makes it a poor entry meter.

The Miller et al. Fig 5 nuance prevents the opposite overcorrection. Electricity
is not disqualified from weather-response evidence. Instead, Level-3
weather-response support should be evaluated per meter and attached only where
the local electricity meter shows an adequate temperature relationship.

## Consequences

+ Phase E transfer scripts default to electricity while retaining chilledwater
  as an allowed meter.
+ M6.1 is now the supervised label-bridge integrity slice; A2 did not run a full
  electricity scan.
+ Chilledwater is deferred, not deleted. Later chilledwater work can use the
  Level-3 weather-conditioned path after the entry electricity workflow exists.
+ Unknown #25 stays open as a weather-feature-validity caveat.
+ This ADR itself introduced no metric, label bridge execution, M3 numeric-line
  change, or TabPFN full-corpus scanner role.
