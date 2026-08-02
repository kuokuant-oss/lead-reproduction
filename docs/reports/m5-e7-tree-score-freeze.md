# M5 E7 Tree score freeze

The label-free M5 E7 Tree score fields are frozen after 160 OOF and 32 final
all-even component fits (192 canonical fits in total).  The selected
regularisation value is `C=0.01` for both support and neutral meta-models.

Even-building OOF AP by fold:

| Fold | s11 | Support stack | Neutral stack |
| --- | ---: | ---: | ---: |
| 0 | 0.845405 | 0.835695 | 0.863590 |
| 1 | 0.960637 | 0.948992 | 0.966093 |
| 2 | 0.887338 | 0.879189 | 0.891150 |
| 3 | 0.937568 | 0.935277 | 0.932028 |
| 4 | 0.815031 | 0.792488 | 0.781621 |

The frozen full-holdout artifact has 10,137,155 label-free rows in canonical
order and four hybrid fields: `deployable_refit_hybrid`,
`locked_reference_hybrid`, `deployable_refit_neutral_hybrid`, and
`locked_reference_neutral_hybrid`.  The digests and raw-index identity are
recorded in `e7_final_score_manifest.json`.

No odd labels were read and no odd AP, ROC, bootstrap, LOO, calibration, or
decision analysis was executed.  Any performance evaluation on the frozen
odd-holdout scores belongs to the separately authorised evaluation phase.
