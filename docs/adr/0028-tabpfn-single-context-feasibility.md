# Isolate large TabPFN single-context feasibility runs

## Status

Accepted.

## Context

Existing M5 comparisons do not establish whether one laptop GPU can predict
with a 500,000-row TabPFN context. External sharding answers a different
question, while CUDA failures can interrupt a long experiment before results
are archived.

## Decision

Add an independent raw-17 experiment with deterministic unique balanced nested
contexts. One disposable worker owns one classifier and one budget. A parent
controller owns atomic state, resume behavior, monitoring, and failure records,
but never imports torch or TabPFN.

The context is accepted only when source/config evidence disables row
subsampling and fitted attributes report the requested row count and one
effective estimator. Fit-only completion is insufficient; validation and test
prediction must also finish.

## Consequences

+ Existing M3/M5 artifacts and golden metrics remain unchanged.
+ Internal row chunking is allowed; external model sharding is forbidden.
+ Query OOM may reduce query batch size, never training rows.
+ Formal 100K--500K execution remains an explicit operator action.
