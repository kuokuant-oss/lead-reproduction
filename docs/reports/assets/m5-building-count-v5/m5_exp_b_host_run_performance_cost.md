# Experiment B — Tree vs TabPFN on host-run cells

Generated 2026-09-01 12:02 (Asia/Taipei). All inputs read-only from published `COMPLETE.json` cells.

## Scope

17 matched contexts in which **both** models ran on this host. The 15 TabPFN cells executed on Colab are excluded, together with their paired Tree cells, so every number comes from the same machine.

Colab cells are identified by the absence of a `prediction_chunks` directory: cells computed here stream 206 prediction chunks to disk, Colab cells were published as a finished artifact only.

Excluded (Colab TabPFN): K=400 b3 r0, K=400 b3 r1, K=400 b4 r0, K=400 b4 r1, K=200 b0 r0, K=200 b0 r1, K=200 b1 r0, K=200 b3 r0, K=200 b3 r1, K=200 b4 r0, K=200 b4 r1, K=100 b3 r0, K=100 b3 r1, K=100 b4 r0, K=100 b4 r1.

Hardware: NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120), 24 CPU cores, WSL2 Ubuntu. Every cell uses the same 50,000-row context (25k anomaly / 25k normal), 137 features, model seed 42, and the same frozen 4,102,084-row holdout.

Cost is the span of artifact mtimes inside each cell directory, the same definition for both models. For the four Tree cells the scheduler timed directly this span runs about 4% below the recorded elapsed time, so the ratios are mild underestimates.

Results are reported per budget K. Surviving contexts are unevenly spread across K (K=725: 2, K=400: 6, K=200: 3, K=100: 6), so a context-weighted average would silently weight some budgets more than others. Where a single figure is given it is the mean of the per-budget means, weighting each K equally.

## Computational efficiency

| K | n | Tree mean | TabPFN mean | Ratio | Tree rows/s | TabPFN rows/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 725 | 2 | 523 s | 29,542 s (8.21 h) | 57x | 7,847 | 139 |
| 400 | 6 | 447 s | 29,605 s (8.22 h) | 66x | 9,170 | 139 |
| 200 | 3 | 391 s | 29,359 s (8.16 h) | 75x | 10,499 | 140 |
| 100 | 6 | 400 s | 29,425 s (8.17 h) | 73x | 10,244 | 139 |
| **Equal-weight** | | | | **68x** | | |

**TabPFN costs 68x the wall-clock time of the tree ensemble**, ranging from 57x at K=725 to 75x at K=200.

The ratio narrows as K grows because the two models scale differently. TabPFN run time is essentially independent of K: across all 17 cells it spans 29,320-29,835 s, a range of 1.7%. Tree run time grows with the source pool, from 391 s to 523 s.

The two models also spend their time differently. The tree ensemble fits four boosters in 9-188 s and spends the rest building the holdout feature matrix and scoring it, producing five score arrays. TabPFN fits its context in roughly 530-660 s and spends about 98% of the run streaming 206 prediction chunks, producing one score array.

## Predictive performance

PR-AUC, Tree / TabPFN (difference). Difference is TabPFN minus Tree.

| K | n | Chilled water | Steam | Hot water |
| ---: | ---: | ---: | ---: | ---: |
| 725 | 2 | 0.7856 / 0.8111 (+0.0255) | 0.7620 / 0.7852 (+0.0232) | 0.8280 / 0.8147 (-0.0132) |
| 400 | 6 | 0.7573 / 0.7492 (-0.0081) | 0.7438 / 0.7726 (+0.0288) | 0.8011 / 0.7230 (-0.0781) |
| 200 | 3 | 0.6376 / 0.6625 (+0.0249) | 0.6922 / 0.7575 (+0.0652) | 0.7208 / 0.6422 (-0.0786) |
| 100 | 6 | 0.5923 / 0.6021 (+0.0097) | 0.6947 / 0.6417 (-0.0530) | 0.7206 / 0.6420 (-0.0787) |
| **Equal-weight difference** | | **+0.0130** | **+0.0161** | **-0.0622** |

Per-budget direction:

- **Chilled water**: TabPFN ahead at 3 of 4 budgets (K=725 +0.0255; K=400 -0.0081; K=200 +0.0249; K=100 +0.0097).
- **Steam**: TabPFN ahead at 3 of 4 budgets (K=725 +0.0232; K=400 +0.0288; K=200 +0.0652; K=100 -0.0530).
- **Hot water**: TabPFN ahead at 0 of 4 budgets (K=725 -0.0132; K=400 -0.0781; K=200 -0.0786; K=100 -0.0787).

## Cost-effectiveness

| K | Extra TabPFN time | Chilled water | Steam | Hot water |
| ---: | ---: | ---: | ---: | ---: |
| 725 | 8.06 h | 32 h | 35 h | no gain |
| 400 | 8.10 h | no gain | 28 h | no gain |
| 200 | 8.05 h | 32 h | 12 h | no gain |
| 100 | 8.06 h | 83 h | no gain | no gain |
| **Equal-weight** | 8.07 h | **62 h** | **50 h** | **no gain** |

Hours of extra compute per +0.1 PR-AUC. "No gain" means the extra compute produced a worse detector than the tree ensemble.

## Overall

Across the 4 budgets, TabPFN costs 68x the wall-clock time of the tree ensemble, 8.1 hours more per context. It returns +0.0130 PR-AUC on chilled water and +0.0161 on steam, and loses 0.0622 on hot water. Buying a tenth of a point of PR-AUC therefore costs 62 hours on chilled water and 50 hours on steam, and is not available at any price on hot water.

## Detailed records

### Per-cell timing

| K | b | r | Model | Chunks | Span (s) | Hours | Start | End |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 725 | 725 | 0 | Tree | 21 | 529 | 0.15 | 08-26 11:09 | 08-26 11:18 |
| 725 | 725 | 0 | TabPFN | 206 | 29,634 | 8.23 | 08-26 11:18 | 08-26 19:32 |
| 725 | 725 | 1 | Tree | 21 | 516 | 0.14 | 08-26 19:32 | 08-26 19:41 |
| 725 | 725 | 1 | TabPFN | 206 | 29,451 | 8.18 | 08-26 19:41 | 08-27 03:52 |
| 400 | 0 | 0 | Tree | 21 | 401 | 0.11 | 08-27 03:52 | 08-27 03:59 |
| 400 | 0 | 0 | TabPFN | 206 | 29,393 | 8.16 | 08-27 03:59 | 08-27 12:09 |
| 400 | 0 | 1 | Tree | 21 | 421 | 0.12 | 08-27 12:09 | 08-27 12:16 |
| 400 | 0 | 1 | TabPFN | 206 | 29,835 | 8.29 | 08-27 12:17 | 08-27 20:34 |
| 400 | 1 | 0 | Tree | 21 | 501 | 0.14 | 08-27 17:58 | 08-27 18:06 |
| 400 | 1 | 0 | TabPFN | 206 | 29,780 | 8.27 | 08-27 20:34 | 08-28 04:51 |
| 400 | 1 | 1 | Tree | 21 | 475 | 0.13 | 08-27 18:07 | 08-27 18:15 |
| 400 | 1 | 1 | TabPFN | 206 | 29,537 | 8.20 | 08-28 04:51 | 08-28 13:04 |
| 400 | 2 | 0 | Tree | 21 | 451 | 0.13 | 08-27 18:15 | 08-27 18:23 |
| 400 | 2 | 0 | TabPFN | 206 | 29,634 | 8.23 | 08-28 13:04 | 08-28 21:18 |
| 400 | 2 | 1 | Tree | 21 | 434 | 0.12 | 08-27 18:23 | 08-27 18:30 |
| 400 | 2 | 1 | TabPFN | 206 | 29,452 | 8.18 | 08-28 21:18 | 08-29 05:29 |
| 200 | 1 | 1 | Tree | 21 | 380 | 0.11 | 08-27 19:23 | 08-27 19:30 |
| 200 | 1 | 1 | TabPFN | 206 | 29,368 | 8.16 | 08-29 05:30 | 08-29 13:39 |
| 200 | 2 | 0 | Tree | 21 | 393 | 0.11 | 08-27 19:30 | 08-27 19:37 |
| 200 | 2 | 0 | TabPFN | 206 | 29,346 | 8.15 | 08-29 13:39 | 08-29 21:48 |
| 200 | 2 | 1 | Tree | 21 | 399 | 0.11 | 08-27 19:37 | 08-27 19:44 |
| 200 | 2 | 1 | TabPFN | 206 | 29,362 | 8.16 | 08-29 21:49 | 08-30 05:58 |
| 100 | 0 | 0 | Tree | 21 | 353 | 0.10 | 08-27 20:11 | 08-27 20:17 |
| 100 | 0 | 0 | TabPFN | 206 | 29,357 | 8.15 | 08-30 05:58 | 08-30 14:08 |
| 100 | 0 | 1 | Tree | 21 | 392 | 0.11 | 08-27 20:18 | 08-27 20:24 |
| 100 | 0 | 1 | TabPFN | 206 | 29,390 | 8.16 | 08-30 14:08 | 08-30 22:18 |
| 100 | 1 | 0 | Tree | 21 | 368 | 0.10 | 08-27 20:25 | 08-27 20:31 |
| 100 | 1 | 0 | TabPFN | 206 | 29,320 | 8.14 | 08-30 22:18 | 08-31 06:27 |
| 100 | 1 | 1 | Tree | 21 | 376 | 0.10 | 08-27 20:31 | 08-27 20:38 |
| 100 | 1 | 1 | TabPFN | 206 | 29,488 | 8.19 | 08-31 06:27 | 08-31 14:39 |
| 100 | 2 | 0 | Tree | 21 | 549 | 0.15 | 08-27 20:38 | 08-27 20:47 |
| 100 | 2 | 0 | TabPFN | 206 | 29,592 | 8.22 | 08-31 14:39 | 08-31 22:52 |
| 100 | 2 | 1 | Tree | 21 | 365 | 0.10 | 08-27 20:47 | 08-27 20:53 |
| 100 | 2 | 1 | TabPFN | 206 | 29,401 | 8.17 | 08-31 22:53 | 09-01 07:03 |

### Per-context PR-AUC

| K | b | r | Chilled water | Steam | Hot water |
| ---: | ---: | ---: | --- | --- | --- |
| 725 | 725 | 0 | 0.7829 / 0.8026 (+0.0197) | 0.7713 / 0.8071 (+0.0358) | 0.8330 / 0.8105 (-0.0224) |
| 725 | 725 | 1 | 0.7882 / 0.8196 (+0.0313) | 0.7527 / 0.7633 (+0.0106) | 0.8230 / 0.8190 (-0.0040) |
| 400 | 0 | 0 | 0.7086 / 0.6948 (-0.0137) | 0.7531 / 0.7284 (-0.0246) | 0.8298 / 0.7358 (-0.0940) |
| 400 | 0 | 1 | 0.7145 / 0.7239 (+0.0094) | 0.7113 / 0.7071 (-0.0042) | 0.8162 / 0.7482 (-0.0679) |
| 400 | 1 | 0 | 0.7915 / 0.7840 (-0.0075) | 0.7281 / 0.7951 (+0.0670) | 0.8052 / 0.7128 (-0.0924) |
| 400 | 1 | 1 | 0.7958 / 0.7578 (-0.0380) | 0.7450 / 0.8052 (+0.0603) | 0.7949 / 0.7073 (-0.0877) |
| 400 | 2 | 0 | 0.7693 / 0.7567 (-0.0127) | 0.7708 / 0.7961 (+0.0253) | 0.7722 / 0.7034 (-0.0688) |
| 400 | 2 | 1 | 0.7640 / 0.7777 (+0.0137) | 0.7546 / 0.8036 (+0.0490) | 0.7881 / 0.7303 (-0.0577) |
| 200 | 1 | 1 | 0.7041 / 0.6964 (-0.0077) | 0.6138 / 0.7177 (+0.1039) | 0.7220 / 0.6637 (-0.0583) |
| 200 | 2 | 0 | 0.6024 / 0.6559 (+0.0535) | 0.7341 / 0.7783 (+0.0442) | 0.7184 / 0.6196 (-0.0988) |
| 200 | 2 | 1 | 0.6063 / 0.6352 (+0.0289) | 0.7286 / 0.7763 (+0.0477) | 0.7218 / 0.6433 (-0.0786) |
| 100 | 0 | 0 | 0.5957 / 0.6037 (+0.0080) | 0.6925 / 0.6552 (-0.0374) | 0.7272 / 0.7603 (+0.0331) |
| 100 | 0 | 1 | 0.5711 / 0.5940 (+0.0228) | 0.6418 / 0.6232 (-0.0187) | 0.6832 / 0.7521 (+0.0689) |
| 100 | 1 | 0 | 0.5115 / 0.6307 (+0.1192) | 0.6912 / 0.6178 (-0.0734) | 0.7539 / 0.5732 (-0.1807) |
| 100 | 1 | 1 | 0.5216 / 0.6477 (+0.1261) | 0.6767 / 0.5991 (-0.0776) | 0.7309 / 0.5581 (-0.1728) |
| 100 | 2 | 0 | 0.6786 / 0.5676 (-0.1110) | 0.7307 / 0.6681 (-0.0626) | 0.7127 / 0.5893 (-0.1234) |
| 100 | 2 | 1 | 0.6755 / 0.5686 (-0.1069) | 0.7352 / 0.6871 (-0.0481) | 0.7160 / 0.6187 (-0.0973) |

Each cell reads Tree / TabPFN (difference).
