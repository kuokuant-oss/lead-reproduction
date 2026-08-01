# M5 E4 formal Path A complete — handoff (2026-08-02)

## 0. Status

E4 is **complete and frozen**. Nothing is running. There is no resume command.
The next stage requires explicit human authorisation and is not started here.

| | |
|---|---|
| Branch | `m5-e4-formal-path-a` |
| Base commit | `e7aa8f72a26eceb926038c33c29043ec0c3ce2aa` (E3 completion) |
| Protocol artifact | `6efdc937878a835ad35eb139d38b3fc77c1d12a17e330988c55523f85cdebb10` |
| Realised order digest | `63ca76f1167768252b29992fd791c450ba33447f5908b8938f1b67d0ecc732e3` |
| Result root | `data/processed/m5_e4_formal_path_a/` (269 files) |

## 1. What ran

24 fits (3 context seeds × 4 cells × 2 scaler arms), one fit each, 8
same-process inference repeats each = 192 `predict_proba` calls. Single GPU
worker, strictly sequential, in a randomised block schedule. 2 h 10 m.

Coverage: 24/24 fits, 192/192 repeats, 24 distinct states, 24 distinct process
UUIDs, effective `n_estimators_` = 8 in all 24 (re-read from the persisted
states), 0 stderr bytes, 0 interrupted, 0 stray temp files, 1000/1000 valid
clustered draws in all eight bootstraps.

## 2. The result in one paragraph

Adding **hotwater-negative** support to the context raises steam-positive
separation from hotwater negatives by +0.41 AUC and +0.63 margin, consistently
across all three seeds, both clusterings, both endpoints and both scaler arms —
and by +0.124/+0.152 more than the matched fixed tree, a gap that itself clears
the full bar. Adding **hotwater-positive** support lowers separation by −0.19
AUC and −0.28 margin, equally consistently, but the matched tree shows the same
thing slightly more strongly, so that effect is not TabPFN's. The **interaction
is not established**: the AUC interval excludes zero and the margin interval does
not. The **scaler arm changes nothing** anywhere.

## 3. Five things a successor should not have to rediscover

**The AUC saturates in cell 01.** All six cell-01 fits give exactly 1.000000
with zero repeat variance, in every seed and both arms. A zero half-width there
is a ceiling, not precision. Any future stage that gates on the AUC alone will
be gating on a saturated metric in a quarter of its cells.

**TabPFN's fit is not bitwise deterministic.** At cell 11 the two scaler arms
are the same transform by construction, so their fit inputs are byte-identical,
and their `ensemble_configs_` are identical too — yet all three seed-pairs
produced different persisted states. E3 established non-determinism for
inference; this establishes it for the fit. The resulting arm-to-arm difference
(1.7–2.5e−03) is *smaller* than the within-fit repeat spread (3.74e−03).

**Cell 11 is therefore a free null control for the scaler axis** and should be
used as one in any later design that touches preprocessing.

**Reporting AUC and margin together is not ceremony.** It is the only reason the
interaction was not overclaimed, and the only reason the two main effects can be
trusted despite the cell-01 ceiling.

**The bottleneck is the feature-matrix build, not the GPU.** 89% of wall-clock at
16.2 GB peak RSS with the GPU at 0%; 11% at 2.7 GB with the GPU at 78–97%.
Blocking both scaler arms of a `(seed, cell)` so the matrix is built once made
the second arm 8× faster and 6× smaller — about 57 minutes saved, more than
parallelism could have delivered. Parallelism is not available: 2 × 16.2 GB
exceeds the whole 23 GB VM.

## 4. Measured throughput, for sizing the next stage

| Quantity | Measured |
|---|---:|
| feature-matrix build (cache miss) | 570–579 s, peak RSS 16.22–16.26 GB |
| cached load (cache hit) | 65–70 s, peak RSS 2.70–2.73 GB |
| fit, 20,000 rows, F4_137 | ~1 s |
| one inference repeat, 352 query rows | ~8 s |
| peak VRAM per unit | 1.72 GB (torch allocator), ~3.1 GB device total |
| whole run | 2 h 10 m |
| clustered bootstrap, 1000 draws | 115 s per AUC endpoint per clustering; 2 s per margin |

Read peak RSS from `/proc/PID/status` `VmHWM`, never from a sampler. On this
same workload a 1-minute sampler said 8.0 GB and a finer one 12.7 GB; the true
peak is 16.2 GB. A sampler can only miss downward, so its error biases toward
"this fits, run two."

## 5. E5 and E6, for when they are authorised

Both are re-scoring problems against the 24 persisted states, not refitting
problems.

- **E5** reloads each state and scores the frozen 192-row query. Smaller than one
  E4 repeat; minutes of GPU time.
- **E6** reloads the same states and scores 10,137,155 rows — four orders of
  magnitude more than E4's query. Size it from the existing sharded full-test
  logs, not by extrapolating the 8 s figure, which is dominated by fixed
  overhead at 352 rows.

If E6 is split across machines, **give each machine a row shard and have it
score all 24 states**, never a subset of states. Sharding by state would make
"machine" confounded with every state contrast, which is the mistake E4
deliberately avoided by keeping to one machine. The laptop and the remote differ
in CUDA build (cu126 vs cu130) and GPU architecture (Ada vs Blackwell), so
cross-machine differences are real, not hypothetical.

## 6. Carried forward unresolved

| Item | Status |
|---|---|
| chilledwater positive vs hotwater negative | `RESOLUTION_LIMITED_DIAGNOSTIC` |
| onset / middle / recovery phase contrast | `UNRESOLVED_NOT_EXECUTED` |

Both were settled before any fit. Chilledwater within-meter readouts are two to
three orders of magnitude below the steam effects; the one interval that
excludes zero under building clustering fails under segment clustering.
Chilledwater is not pooled into the steam claim.

## 7. Not authorised by this handoff

E5 frozen 192-row query, E6 complete other-half full test, Path B,
representation ablation, 500k, site transfer, tree refit, TabPFN 8.1.0 as
science, manuscript changes, adding a model-seed factor, and changing N, cells,
seeds or arms. The base policy file was not modified.

**Neither the 192-row query nor the 10,137,155-row holdout was scored in this
round.** The holdout identity audit
(`docs/reports/m5-final-holdout-identity-audit.md`) reads only `raw_index`,
`building_id`, `site_id` and `anomaly`; no score column was opened.

## 8. Reproducing the result

```bash
# re-validate the canonical root against itself (recomputes every endpoint and
# re-reads the effective ensemble size from each persisted state)
python scripts/m5_e4_import.py \
  --staged    data/processed/m5_e4_formal_path_a \
  --canonical data/processed/m5_e4_formal_path_a \
  --repo-root .

# regenerate the summaries, contrasts and clustered intervals from raw scores
python scripts/m5_e4_analysis.py \
  --canonical data/processed/m5_e4_formal_path_a --repo-root . --endpoints all

# re-derive the verdict
python scripts/m5_e4_decision.py --canonical data/processed/m5_e4_formal_path_a
```

The importer is read-only without `--apply`. Clustered draws are addressable —
draw *d* of cluster type *t* always comes from
`SeedSequence([20260730, 4004, code[t], d])` — so any single draw can be
reproduced on its own, and no result depends on loop order.

## 9. Artifact digests

| Artifact | SHA-256 |
|---|---|
| `e4_protocol.json` | `6efdc937878a835ad35eb139d38b3fc77c1d12a17e330988c55523f85cdebb10` |
| `e4_fit_manifest.json` | `8af4a541a3cf8b63b9275dec3a6672a4cb13d84828951134fd0a384454a21939` |
| `e4_repeat_manifest.json` | `220de1c6c99bc64acde2b70db2eed0aa445030fb73a248b0995abf96209778bd` |
| `e4_input_manifest.json` | `e973c6819e747b2e…` (see the artifact) |
| `e4_summary.json` | `e8cf50e132eaa29a00789edc35f8670b0432e2b4bb293ed26edc2e46f40fe439` |
| `e4_factorial.json` | `f41ff6cedcac90b03cdbfeb862341e78696325d9a3b17bc883d4d63f2e352e4f` |
| `e4_clustered.json` | `c810db1658075f2584ba4273a0e6426a30bccaba1803cb6ce18d3719b3224b8f` |
| `e4_decision.json` | `a8477f9eaa276273546d12c24d6238f4a22a8d78f4ed398ed2c1f74a90f3b28b` |
| remote archive | `81bbd57e772ab292a6a15f611519237c047188c371e9ad4dfc2d10f2f41f8612` |
