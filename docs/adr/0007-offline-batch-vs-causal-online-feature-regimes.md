# Offline batch vs causal online feature regimes

## Status

Accepted (2026-06-22)

## Context

M3.2 value-change features use both past-looking and future-looking meter
reading shifts. That matches the LEAD reproduction line, where anomaly labeling
is treated as offline batch detection over an already observed time series.

The PI also requested a protocol where train and validation each use about half
of the buildings. While checking that split, M3.2a made the feature-regime
distinction explicit:

+ **Offline**: past + future value-change shifts, matching M3.2 and the LEAD
  reproduction framing.
+ **Causal**: past-only value-change shifts, matching a real-time FDD framing
  where future meter readings are unavailable at prediction time.

Both regimes preserve ADR 0001's building-level train/validation separation.

## Decision

Report both feature regimes when the distinction matters:

+ The **offline** regime is the canonical reproduction line for M3.3 and later:
  80/20 `building_id % 5 == 4`, past + future value-change shifts, LightGBM
  M3.2 validation AUC `0.9920`.
+ The **causal** regime is the real-time-FDD variant: past-only value-change
  shifts, used to quantify the cost of deployability without future readings.

The PI-requested 50/50 building split is documented as an experimental-design
response, not as the canonical M3 optimization line unless the reporting target
changes.

## Rationale

This keeps the reproduction faithful to the already established LEAD-aligned
pipeline while preventing ambiguity about deployability. The M3.2a results show
that the causal restriction has a small but measurable cost:

| Split | Regime | Val AUC | P/R/F1 @ 0.5 |
|---|---|---:|---:|
| 80/20 mod5 | offline | 0.9920 | 0.6409/0.9665/0.7707 |
| 80/20 mod5 | causal | 0.9908 | 0.6237/0.9603/0.7562 |
| 50/50 mod2 | offline | 0.9914 | 0.6878/0.9421/0.7951 |
| 50/50 mod2 | causal | 0.9903 | 0.6646/0.9355/0.7772 |

The 50/50-causal label-shuffle sanity check produced AUC `0.4527`, cleaner than
the earlier M3.2 label-shuffle AUC `0.5669`, so it does not raise a concern.

## Consequences

+ M3.3 uses the canonical 80/20 offline line and compares against M3.2 AUC
  `0.9920`.
+ Reports and handoffs must name the feature regime when discussing
  value-change results.
+ The 50/50 PI protocol remains archived and reproducible, but does not replace
  the 80/20 offline headline unless explicitly chosen later.
+ Real-time FDD claims must use the causal past-only result, not the offline
  reproduction result.
