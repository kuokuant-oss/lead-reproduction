# M5 E7 Tree score-freeze handoff — 2026-08-03

## Frozen state

- Protocol commit: `c4821a0899e334a5b1354598742d47fbf992efd5`
- Canonical components: 160 OOF + 32 final = 192
- Meta-model inputs: complete even-building OOF predictions only
- Selected C: support `0.01`; neutral `0.01`
- Odd labels: unopened

## Required evaluation inputs

- `data/processed/m5_e7_full_capacity_tree_strategy/e7_final_score_manifest.json`
- `data/processed/m5_e7_full_capacity_tree_strategy/e7_final_model_manifest.json`
- `data/processed/m5_e7_full_capacity_tree_strategy/e7_score_firewall_audit.json`
- `data/processed/m5_e7_full_capacity_tree_strategy/e7_oof_summary.json`

The next independently authorised evaluation phase may open the odd labels and
perform the approved metrics.  It must not alter the frozen scores, final
models, OOF selection, row order, or protocol.
