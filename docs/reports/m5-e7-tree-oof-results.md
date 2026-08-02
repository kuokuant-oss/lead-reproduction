# M5 E7 Tree OOF Results

This document is part of the pre-score protocol freeze. No OOF models have
been fitted and no OOF AP values are reported here.

The frozen OOF census is five building-disjoint folds, eight expert ensembles
per fold, and four components per expert: 40 ensembles and 160 component
fits. Each validation fold is scored only on steam rows; all meters from that
fold's validation buildings are excluded from its training pool.

The selected support and neutral meta-model C values, per-fold `s11` AP,
support-stack AP, neutral-stack AP, and their deltas will be written only after
all 160 digest-validated OOF component units exist. Missing, temporary,
corrupt, or provenance-incompatible units make finalisation fail closed.
