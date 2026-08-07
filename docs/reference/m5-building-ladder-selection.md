# M5 building ladder selection reference

## 結論先講

M5 的 K 棟建築不是 uniform random sampling，也不是先 shuffle 全部 building
再取前 K。實作採用 representative-balanced greedy selection：每一步都評估
所有尚未選取的 even training buildings，選出加入後最接近 training candidate
pool composition target 的 building。

目前有兩種明確不同的操作模式：

| 模式 | `diversify_candidates` | seed 的實際作用 | 是否提供 meaningful ladder diversity |
| --- | --- | --- | --- |
| Primary M5 protocol | `False` | 只在 selection score 完全相同時作 deterministic tie-break | 通常不會；不同 seed 很可能得到相同 ladder |
| Building-candidate sensitivity pilot | `True` | 只在 score 幾乎同樣好的 acceptable set 內，用 seed-controlled stable hash 選一棟 | 會，但仍受 representative-quality 約束 |

因此，這裡保留的不是執行期 stochastic RNG，而是可重現、受品質邊界控制的
seed diversity。同一 seed、同一 candidate profile 必須產生 byte/digest-identical
ladder；程式沒有呼叫未受控的 random shuffle 或 RNG draw。

## 主要程式碼入口

- Building profile 建立：
  [`build_building_profiles`](../../scripts/m5_building_curve_protocol.py#L78-L137)
- Representative target 與 feature weights：
  [`_design_matrix`](../../scripts/m5_building_curve_protocol.py#L140-L230)
- Seed-controlled stable hash：
  [`stable_priority`](../../scripts/m5_building_curve_protocol.py#L42-L51)
- Greedy ladder 核心：
  [`build_nested_building_ladder`](../../scripts/m5_building_curve_protocol.py#L271-L447)
- 每一步的 score、acceptable set 與最終選擇：
  [selection loop](../../scripts/m5_building_curve_protocol.py#L311-L395)
- K prefix、fit/early-stop roles 與 digest：
  [manifest cell construction](../../scripts/m5_building_curve_protocol.py#L397-L445)
- Even-only、無重複、strict-nested validation：
  [`validate_ladder`](../../scripts/m5_building_curve_protocol.py#L450-L482)
- Primary protocol 呼叫方式：
  [`prepare_m5_building_curve.py`](../../scripts/prepare_m5_building_curve.py#L56-L92)
- Sensitivity canonical/seeded ladders 與 quality gate：
  [`build_sensitivity_audit`](../../scripts/audit_m5_building_candidate_sensitivity.py#L193-L308)
- Sensitivity composition、overlap 與 machine-readable outputs：
  [audit output construction](../../scripts/audit_m5_building_candidate_sensitivity.py#L309-L489)

## 1. Candidate pool 如何形成

Primary preparation 在建立 profile 前先套用：

```text
building_id % 2 == 0
```

程式入口見
[`prepare_m5_building_curve.py` 的 even filter](../../scripts/prepare_m5_building_curve.py#L56-L67)。
Sensitivity audit 也在把 frame 傳給 profile builder 前先過濾 even buildings，見
[`profiles_from_training_frame`](../../scripts/audit_m5_building_candidate_sensitivity.py#L140-L143)。

除此之外還有兩層防線：

1. `build_building_profiles` 收到任何 odd building 會直接 `ValueError`。
2. `validate_ladder` 發現 ladder 內有 odd building 會直接 `AssertionError`。

所以 odd-building canonical holdout 不只是「理論上不該選」，而是在 profile
輸入與 ladder 輸出兩端都有 hard gate。Odd-building labels 不會進入 selection
target。Even training labels 會用來計算 anomaly-related profile；這是 supervised
training-side representativeness，不是查看 holdout labels。

## 2. 每棟 building 的 profile

`build_building_profiles` 對每個 even building 建立一列，內容包括：

- `site_id`
- available row count：`rows`
- anomaly count 與 `anomaly_rate`
- meter 0/1/2/3 的 row count
- meter presence indicator
- meter row-share
- 每個 meter 的 anomaly count/rate
- building 具有幾種 meter：`meter_count`
- 有 anomaly 的 meter 種數：`anomaly_meter_count`
- 是否完全沒有 anomaly：`zero_anomaly`
- anomaly-rate bin
- building-size bin

Anomaly-rate bins 是：

```text
zero
positive_low
positive_mid
positive_high
```

正 anomaly-rate buildings 依 rank percentile 切成三組。Size 先用
`log1p(rows)`，再依 rank percentile 切成四個 quantile bins。實作分別在
[`_positive_rate_bins`](../../scripts/m5_building_curve_protocol.py#L54-L64) 與
[`_quantile_bins`](../../scripts/m5_building_curve_protocol.py#L67-L75)。

`primary_use` 會附加到 profile/audit artifact，但目前 `_design_matrix` 沒有把
`primary_use` 放進 selection score。換句話說，目前 ladder 會報告 primary-use
composition，卻不直接最佳化它。不能把 primary-use audit 誤解成 primary-use
balancing constraint。

## 3. Representative target 是什麼

每棟 building profile 會轉成一個 numeric design vector。Primary 使用的
`sampling_profile="representative"` 以 entire even-building candidate pool 的
empirical mean 作為 target。

| Dimension block | Representative target | Block total weight |
| --- | --- | ---: |
| Site one-hot | Candidate-pool site shares | 2.0 |
| Anomaly-rate-bin one-hot | Candidate-pool bin shares | 2.0 |
| Anomaly-bearing meter count | Candidate-pool distribution | 1.0 |
| Building-size-bin one-hot | Candidate-pool distribution | 1.0 |
| Meter presence | Candidate-pool mean presence | 2.0 |
| Meter row-share | Candidate-pool mean row-share | 2.0 |
| Mean anomaly rate | Candidate-pool mean | 1.0 |
| Zero-anomaly share | Candidate-pool mean | 1.0 |

Categorical block 的 total weight 會平均分配到該 block 的 levels；meter blocks
則平均分配到四個 meters。這些 target/weight 的程式碼在
[`_design_matrix`](../../scripts/m5_building_curve_protocol.py#L140-L230)。

這個 target 是「以 building 為單位的 candidate-pool composition」，不是直接
用所有 raw rows 的 distribution。Meter row-share 本身帶有 building 內的 row
composition，但每棟 building 在 prefix mean 中仍是一個 observation。

## 4. 每一步 greedy score 如何計算

假設已經選了 `p` 棟，profile vector 總和是 `S`，candidate building `c` 的
vector 是 `x_c`，candidate-pool target 是 `t`，各 dimension weight 是 `w`。

加入 `c` 後的 overall prefix mean：

```text
overall_mean(c) = (S + x_c) / (p + 1)
```

Overall discrepancy：

```text
overall_error(c) = sum_j w_j * (overall_mean_j(c) - t_j)^2
```

Role 在選 building 之前已由 position 固定：每五個 positions 中前四個是
`fit`，第五個是 `early_stop`。針對該 position 的 role，另外計算「如果把 c
加入這個 role，該 role prefix 距離 target 多遠」：

```text
role_error(c) = sum_j w_j * (role_mean_j(c) - t_j)^2
```

最終 selection score：

```text
score(c) = overall_error(c) + 0.35 * role_error(c)
```

所以它不只要求整條 ladder representative，也避免 early-stop subset 或 fit
subset 因固定 position pattern 而嚴重偏離。實作見
[`overall_error`、`role_error` 與 `score`](../../scripts/m5_building_curve_protocol.py#L325-L335)。

這是逐步 greedy optimization：每一步找當下加入後最好的 building，但不是
對所有 K-combinations 做 global combinatorial optimization。因此它是可解釋且
計算可行的 local greedy solution，不宣稱是 global optimum。

## 5. Primary M5 到底怎麼選

Primary preparation 沒有傳入 `diversify_candidates=True`，所以使用預設
`False`。候選排序 key 依序為：

```text
selection score
seed-controlled stable priority
building_id
```

由於 score 是第一順位，stable priority 只有在 floating score 完全同分時才會
改變結果。這代表 primary protocol 雖記錄 `building_seed`，但不同 seed 不保證
產生不同 ladder；通常 score 沒有大量 exact ties 時，seed 幾乎不影響選擇。

這是刻意保留的 canonical exact-best greedy 行為。Primary M5 沒有被改成 random
sampling，也沒有被 sensitivity pilot 取代。

## 6. Sensitivity pilot 如何保留 diversity 又不亂抽

Sensitivity pilot 每一步仍先對所有 remaining buildings 計算相同的
representative score。接著才建立 acceptable candidate set：

```text
best = 最低 selection score
acceptable_limit = best * 1.02 + 1e-12
acceptable = score <= acceptable_limit 的 candidates
acceptable = acceptable 中 score 最好的前 4 棟
```

只在這個集合內，才依 `stable_priority(building_id, building_seed)` 由小到大選
下一棟；若 priority 還同分，再用 `building_id`。核心實作見
[`acceptable_limit` 與 stable-hash selection](../../scripts/m5_building_curve_protocol.py#L337-L355)。

因此任何在該步驟：

- score 比 best 差超過 2%，或
- 不在符合 tolerance 的 top 4

的 building 都不可能因 seed 被選中。Seed 不能凌駕 representative score，僅能
在「近乎同樣好」的 candidates 之間提供 deterministic diversity。

`stable_priority` 是一個由 `building_id XOR seed` 開始的 64-bit mixing function。
它沒有 mutable RNG state，也不依賴 input row order。同一 building ID/seed 永遠
得到同一 priority；相同輸入重跑不會抽到另一棟。

用簡化 pseudocode 表示：

```python
for position in range(max_K):
    role = "early_stop" if (position + 1) % 5 == 0 else "fit"

    for candidate in remaining_buildings:
        overall_error = discrepancy(selected + candidate, pool_target)
        role_error = discrepancy(selected_for_role + candidate, pool_target)
        score[candidate] = overall_error + 0.35 * role_error

    if primary_protocol:
        chosen = argmin(score, stable_hash_tie_break, building_id)
    else:
        acceptable = top_4(score <= best_score * 1.02 + 1e-12)
        chosen = argmin(acceptable, stable_hash(building_seed), building_id)

    append chosen once
```

真正執行的 source of truth 仍是
[`build_nested_building_ladder`](../../scripts/m5_building_curve_protocol.py#L271-L447)，
上面 pseudocode 只用於解釋。

## 7. K=10/20/50/100 為何 strict nested

程式只建立一次長度 `max(K)` 的 ordered ladder，然後：

```text
K10  = ladder[:10]
K20  = ladder[:20]
K50  = ladder[:50]
K100 = ladder[:100]
```

不是每個 K 各跑一次 selection。Prefix construction 見
[`ladder.iloc[:budget]`](../../scripts/m5_building_curve_protocol.py#L397-L415)，
strict-superset、digest、無重複與 even-only gates 見
[`validate_ladder`](../../scripts/m5_building_curve_protocol.py#L450-L482)。

Role 也綁定 position，而不是 K：positions 5、10、15、20、… 永遠是
`early_stop`，其他是 `fit`。因此增大 K 不會改變較小 prefix 中 building 的 role。
`role_seed` 明確是 `None`。

## 8. Sensitivity quality gate

Pilot 另外建立 seed 42、`diversify_candidates=False` 的 canonical exact-best
greedy ladder作為 reference。對每個 seeded ladder、每個 K 都重新計算完整
prefix discrepancy，且必須同時滿足：

```text
seeded_discrepancy <= canonical_discrepancy * 1.50 + 1e-12
seeded_discrepancy - canonical_discrepancy <= 0.003
```

實作見
[`canonical ladder` 與 seeded ladders](../../scripts/audit_m5_building_candidate_sensitivity.py#L219-L242)
以及
[`quality comparison`](../../scripts/audit_m5_building_candidate_sensitivity.py#L272-L308)。

第一條限制 relative degradation，第二條限制 absolute degradation。兩條都通過
才可進模型 evaluation。Audit 還要求 seed 42/43/44 在每個 K 的 prefix 都確實
不同，否則 sensitivity pilot 不算 ready。

## 9. Building seed 與 row seed 是分開的

Building selection 完成後才進行 row allocation。正式 row policy 先按每個新增
K block 中 building 的 available rows 比例建立固定 quota，見
[`add_proportional_row_quotas`](../../scripts/m5_building_curve_protocol.py#L517-L566)。

每棟 building 內再用 raw-row identity 與固定 `row_seed` 的 stable priority 取前
`quota` rows，見
[`average_building_capped_indices`](../../scripts/m5_building_curve_protocol.py#L569-L623)。

Sensitivity pilot 固定：

```text
building_seed = 42 / 43 / 44
row_seed      = 42
role_seed     = None
model_seed    = 42
```

所以跨 seed 變化只來自 source buildings，不是 row sampling 或 model randomness
一起改變。

## 10. 這算不算「保持隨機性」

精確說法如下：

- 有 seed-controlled diversity。
- 沒有執行期 stochastic randomness。
- 不是 uniform random sampling。
- 不是 weighted random sampling。
- Seed 不能選 acceptable set 外的 building。
- 同一 seed 可完全重現。
- Primary 模式 seed 只處理 exact ties；sensitivity 模式 seed 才能在近最佳集合內
  改變選擇。

如果「隨機性」指每次執行都可能不同，答案是沒有；這是研究 protocol 所需的
determinism。如果「隨機性」指不同預先指定 seeds 能產生不同但合理的 candidates，
答案是有，而且 diversity 被 score tolerance、top-4 與 prefix quality gate 三層
限制。

## 11. 已知邊界與不可過度解讀處

- `primary_use` 目前只 audit、不進 score。
- Selection 使用 even training labels 建 anomaly profiles；它不是 unsupervised
  building selection，但沒有使用 odd holdout labels。
- Greedy ladder 是逐步 local optimum，不是全域最佳 K-subset 的證明。
- 三個 seeds 是 sensitivity pilot，不足以估計一個隨機抽樣母體分布。
- Quality gate 確保 composition discrepancy 沒有明顯惡化，不保證所有未納入
  score 的 covariates 都相等。
- Row allocation、tree training downsampling 與 model fitting 是 selection 之後的
  獨立階段，不應混稱為 building selection randomness。

## 12. 對應 tests

- 同 seed byte/digest-identical：
  [`test_same_seed_artifacts_are_byte_and_digest_identical`](../../tests/test_m5_building_candidate_sensitivity.py#L50-L62)
- 三 seeds 不同且 K strict nested：
  [`test_three_seeds_are_distinct_and_every_budget_is_a_strict_prefix`](../../tests/test_m5_building_candidate_sensitivity.py#L63-L109)
- Unique、even-only、odd labels ignored：
  [`test_selected_buildings_are_unique_even_and_holdout_labels_are_ignored`](../../tests/test_m5_building_candidate_sensitivity.py#L110-L128)
- Candidate limit 與 prefix quality gate：
  [`test_quality_composition_and_candidate_limits_pass`](../../tests/test_m5_building_candidate_sensitivity.py#L129-L162)
- Fixed row-seed independence：
  [`test_fixed_row_seed_keeps_priority_policy_independent_of_building_seed`](../../tests/test_m5_building_candidate_sensitivity.py#L163-L176)
