# M5 E6 design-audit artifacts (DRAFT)

Byte-identical copies of the draft artifacts in
`data/processed/m5_e6_protocol_draft/`, which is gitignored with the rest of
`data/processed/`.

**None of these is a launch artifact.** `e6_protocol.DRAFT.json` carries
`not_a_launch_artifact: true` and lists the items still awaiting a human ruling.
No protocol has been frozen and E6 has not been started.

| File | What it is |
|---|---|
| `e6_row_manifest.json` | full-holdout identity, verified; records which columns were read and that the score column was not |
| `e6_shard_manifest.json` | 12 row shards that tile the holdout, each scored by all 24 states |
| `e6_cost_model.json` | measured throughput and the R1 / R8 / R1_PLUS_SENTINEL comparison |
| `e6_protocol.DRAFT.json` | the draft design, the candidate decision rule, and the open items |
