# M4 evaluation protocol

## Status

Proposed

## Context

M4 is a refactor and foundation milestone. It must preserve successful M3
results while exposing known limitations for later slices. Prior project
workflow uses stage gates, quantified Done when criteria, handoffs, and
machine-readable provenance JSON.

## Decision

M4 uses a three-layer evaluation protocol:

+ Golden regression: M3.2 and M3.4 AUC must remain within +/- `0.0005`.
+ Diagnostic robustness: site-held-out and per-meter metrics remain visible.
+ Readiness review: `src/lead` API and duplicate-helper removal are checked
  before M5 starts.

## Rationale

Separating numeric preservation from readiness prevents a structural refactor
from being mistaken for a modeling improvement. It also keeps documented
limitations visible instead of hiding them behind headline AUC.

## Consequences

+ M4.0 creates tracked golden metric provenance.
+ M4.1 writes a regression result JSON.
+ Later semantic changes must cite their own gate results.
+ Precision, recall, and F1 are useful diagnostics but AUC is the primary
  compatibility gate.
