# M5 E7 Final All-Even Training Pools

This pre-execution protocol correction freezes the eight final all-even input
identities omitted from `c4821a0`. It was made before any formal fit or odd
prediction. OOF folds and OOF pool identities are unchanged.

Support pools are generated from all even-building rows using the existing
historical `downsample_indices` path and seeds 10/20. `final_s11` has
2,708,308 rows, matching the historical Full Tree realised row count. The
historical source records sampling provenance and row count, but not a training
raw-index digest; no nonexistent identity evidence is claimed.

Each neutral pool is independently sampled from the all-even full-support pool
using its frozen pair seeds and exactly matches its paired support pool's total,
positive, and negative row counts. The raw-index vectors remain gitignored;
their locations and SHA-256 digests are frozen in the manifest.
