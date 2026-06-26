# Regression noise-floor gate

## Status

Accepted (2026-06-26)

## Context

M3 reports several metrics rounded to four decimals, while JSON artifacts retain
full precision for many values. Prior M3 work used +/- `0.0005` as the
practical noise floor for distinguishing meaningful movement from rerun or
rounding noise.

## Decision

Use +/- `0.0005` AUC as the M4 regression gate. A refactor passes if M3.2 and
M3.4 AUC deltas are strictly within that band. A semantic change must stop for
review if it moves outside the band unless that movement is the intended and
documented result.

## Rationale

The threshold is small enough to catch unintended behavior changes but tolerant
of rounding and minor stochastic variation around already accepted M3 metrics.

## Consequences

+ Golden metrics must record full available precision and display targets.
+ M4.1 cannot claim success without explicit delta values.
+ M4.2+ changes that exceed the gate require ADR or unknown updates.
+ Commit messages should include the measured regression deltas.
