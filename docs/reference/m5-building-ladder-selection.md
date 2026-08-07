# M5 source-building ladder selection

## 現行協定

M5 的 source-building identity 現在使用：

> Seeded site-stratified random sampling without replacement, subject only to
> prespecified meter-coverage feasibility constraints.

這個協定要研究的是 source-building 數量 (K) 增加時的模型表現，而不是尋找
composition 最佳的 source subset。舊的 representative-balanced greedy score、
candidate discrepancy、role-prefix discrepancy、anomaly/size weights 與
near-best candidate diversification 均不再參與 building selection。

目前 budgets 固定為：

```text
K = 10, 20, 50, 100
```

每個 `building_seed` 只生成一條長度 100 的 ladder：

```text
K10  = ladder[:10]
K20  = ladder[:20]
K50  = ladder[:50]
K100 = ladder[:100]
```

因此 building sets 與固定 position roles 都是 strict nested。

## 一句話解釋

Source buildings were selected using seeded site-stratified random sampling
without replacement. Budgets were strict nested prefixes of a single random
ladder. Sampling was subject only to prespecified meter-coverage feasibility
constraints, requiring every evaluated meter to be represented at the smallest
budget and to gain at least one additional source building at each subsequent
budget. Other building characteristics were audited after sampling but were not
optimized during selection.

## Selection 可讀取與不可讀取的資料

### Identity selection inputs

抽樣 identity 只使用：

- `building_id`：唯一 identity、even/odd split 與 within-site permutation。
- `site_id`：建立 strata 及 candidate-pool proportional site schedule。

Meter presence columns：

```text
meter_0_present
meter_1_present
meter_2_present
meter_3_present
```

只在完整 random ladder 生成後檢查 feasibility，不做 ranking、weighting、交換或
greedy correction。

Sampler 只抽取上述欄位的程式碼：
[`_sampling_profiles()`](../../scripts/m5_building_curve_protocol.py#L145-L169)。

### 明確排除

下列欄位不會影響 building identity：

- anomaly label、anomaly count/rate/bin、zero-anomaly status；
- anomaly-bearing meter count；
- total rows、building size/bin；
- meter row count/share；
- primary use；
- candidate-pool discrepancy 或任何手調權重；
- fit/early-stop role composition。

完整 profiles 仍可在抽樣完成後輸出 composition diagnostics，但 diagnostics 不會回饋
sampler。測試會直接改寫所有 anomaly diagnostics，確認同 seed 的 building order
完全不變：
[`test_anomaly_diagnostics_cannot_change_building_identity`](../../tests/test_m5_building_candidate_sensitivity.py#L139-L170)。

## Candidate pool

Candidate pool 僅包含：

```python
building_id % 2 == 0
```

Odd-ID canonical holdout 不會傳入 sampler。由 training frame 建 profile 前即先做
even filter：
[`profiles_from_training_frame()`](../../scripts/audit_m5_building_candidate_sensitivity.py#L135-L138)。

核心 sampler 也會再次拒絕任何 odd building、duplicate building 或缺少 site/meter
presence 的輸入：
[`_sampling_profiles()`](../../scripts/m5_building_curve_protocol.py#L145-L169)。

Odd holdout labels 改變而 even profiles 不變的 leakage test：
[`test_only_even_unique_buildings_and_holdout_data_are_ignored`](../../tests/test_m5_building_candidate_sensitivity.py#L117-L137)。

## Seeded random draw

### RNG

每次 draw 使用 NumPy `PCG64`，而不是 stable hash 排序：

```python
SeedSequence([building_seed_low32, building_seed_high32, attempt])
Generator(PCG64(seed_sequence))
```

實作：
[`_rng_for_attempt()`](../../scripts/m5_building_curve_protocol.py#L172-L178)。

這裡的「真正 seeded random」是指由正式 pseudorandom generator 產生 permutation，
不是先計算 deterministic optimization score 再只用 seed 解同分。它仍是可重現的
pseudo-random stream，不使用 OS entropy 或不可重現的 uncontrolled RNG。

### Within-site permutation

每個 site 先按 `building_id` 排序，消除輸入 DataFrame row order 的影響，再由該
attempt 的 PCG64 stream 執行：

```python
rng.permutation(building_ids_in_site)
```

Sampling without replacement，因此同一 site 與整條 ladder 都不會重複 building。

### Proportional site interleaving

令：

- (N_s)：candidate pool 中 site (s) 的 building 數量；
- (N)：candidate pool building 總數；
- (n_s(t-1))：前 (t-1) 個 positions 已從 site (s) 取出的數量。

position (t) 的 proportional deficit 為：

```text
d_s(t) = t * N_s / N - n_s(t-1)
```

每一步從仍有 building 的 sites 中選擇最大 (d_s(t)) 的 site，然後取該 site random
permutation 的下一棟。只有 site deficit 完全同分時，才用同一 RNG stream 產生的
site priority，再以 `site_id` 決勝。

這讓每個 prefix 的 site counts 緊貼 candidate-pool 比例，同時 building identity
確實由 random permutation 決定。它不是把所有 buildings 做 uniform shuffle，也不把
sites 人為平衡成相同比例。

完整 draw：
[`_site_stratified_random_draw()`](../../scripts/m5_building_curve_protocol.py#L181-L246)。

## Meter feasibility constraints

預設 evaluation meters 為 0、1、2、3，且：

```text
count_m(10) >= 2
count_m(20) >= count_m(10) + 1
count_m(50) >= count_m(20) + 1
count_m(100) >= count_m(50) + 1
```

`count_m(K)` 是 K-prefix 中包含 meter (m) 的不同 source-building 數量，不是
rows。Multi-meter building 可同時增加多個 meter counts。

Constraint audit：
[`_meter_constraint_audit()`](../../scripts/m5_building_curve_protocol.py#L249-L306)。

Sampler 先做一個必要但不放寬條件的 global capacity preflight。四個 budgets、起始
minimum 2、每次至少增加 1，代表每個 meter 在整個 candidate pool 至少必須存在於
5 棟 buildings；不足時立即回報 meter、K、available 與 required：
[`_preflight_meter_capacity()`](../../scripts/m5_building_curve_protocol.py#L309-L330)。

## Whole-ladder rejection sampling

每個 attempt 的流程是：

1. 對每個 site 產生新的 seeded random permutation。
2. 按 proportional site schedule 交錯成完整 100-building ladder。
3. 一次檢查 K=10/20/50/100 的所有 meter constraints。
4. 全部通過才接受。
5. 任一 constraint 失敗，就丟棄整條 ladder，以同一 `building_seed` 的下一個
   deterministic `attempt` stream 重抽。

禁止：

- 交換單一 building；
- greedy 補 meter；
- 按 meter rarity 排名；
- 放寬 q 或 growth constraint；
- 超過 attempts 後靜默接受。

預設最多 10,000 attempts。找不到 feasible ladder 時拋出
`LadderInfeasibilityError`，訊息包含 meter、K、best observed 與 required。
核心 orchestration：
[`build_nested_building_ladder()`](../../scripts/m5_building_curve_protocol.py#L341-L535)。

這是一個條件式 random sample：accepted ladders 是 site-stratified random draws
中滿足事先宣告 meter feasibility 的子集合。Meter 因此影響 accept/reject，但不影響
單一 building 的 score，因為系統根本沒有 selection score。

## Fit / early-stop roles

Roles 在 ladder 通過 meter gate 後才依 position 指派：

```text
position % 5 == 0 -> early_stop
其他 positions       -> fit
```

因此每 5 棟固定 4 fit / 1 early-stop，role 不會影響 sampling。較大 K 是相同 ladder
prefix，所以較小 K 的 role 永遠不變。

Validator 同時檢查 position rule、strict nesting、fit/ES partition、even-only、
uniqueness、digests、site census 與 meter growth：
[`validate_ladder()`](../../scripts/m5_building_curve_protocol.py#L537-L634)。

## Row 與 model randomness 分離

```text
building_seed = swept
row_seed      = 42
role_seed     = None
model_seed    = 42
```

Building RNG 不參與 row priority。Row selection 仍以固定 `row_seed` 對 raw row
identity 計算 stable priority：
[`average_building_capped_indices()`](../../scripts/m5_building_curve_protocol.py#L706-L758)。

因此跨 building seeds 的主要實驗差異是 source-building identity；row policy、role
policy 與 model seed 不跟著 sweep。

## Audit artifacts

CPU-only audit entry point：
[`build_sensitivity_audit()`](../../scripts/audit_m5_building_candidate_sensitivity.py#L188-L471)。

預設產生 seeds 42、43、44、45、46，輸出：

- `building_ladder_seed<seed>.csv`：逐 position building、site、role、accepted
  attempt、within-site draw rank 與 diagnostics。
- `building_ladder_seed<seed>.json`：sampling/RNG/constraints/cells/digests。
- `sampling_prefix_audit.csv`：每個 seed、每個 K 的 building IDs、site counts、
  每 meter source-building count、required minimum、pass/fail 與 digest。
- `building_overlap.csv`：每個 K 的 cross-seed intersection/Jaccard。
- `composition_audit.csv`：抽樣後才計算的 rows、anomalies、primary use、meter
  row-share、size/anomaly bins 等 diagnostics。
- `summary.json`：all-constraint gate、distinct-draw gate、accepted attempt 與所有
  reproducibility digests。

2026-08-07 的 5-seed preflight 結果全部通過。accepted zero-based attempts 為：

```text
seed 42 -> 8
seed 43 -> 7
seed 44 -> 7
seed 45 -> 7
seed 46 -> 22
```

K=10/20/50/100 各自都有 5 個不同 prefixes；所有 meter constraints 均通過。

## Validation tests

主要 tests：

- byte-identical rerun：
  [same-seed artifacts](../../tests/test_m5_building_candidate_sensitivity.py#L57-L68)
- different seeds 與 strict nesting：
  [distinct nested ladders](../../tests/test_m5_building_candidate_sensitivity.py#L70-L115)
- even-only 與 odd holdout isolation：
  [holdout isolation](../../tests/test_m5_building_candidate_sensitivity.py#L117-L137)
- anomaly diagnostics 不影響 identity：
  [label independence](../../tests/test_m5_building_candidate_sensitivity.py#L139-L170)
- site stratification 與 RNG provenance：
  [stratification audit](../../tests/test_m5_building_candidate_sensitivity.py#L172-L209)
- meter minimum/growth：
  [meter constraints](../../tests/test_m5_building_candidate_sensitivity.py#L211-L232)
- deterministic whole-ladder rejection/redraw：
  [deterministic redraw](../../tests/test_m5_building_candidate_sensitivity.py#L234-L262)
- explicit infeasibility、no silent relaxation：
  [no silent relaxation](../../tests/test_m5_building_candidate_sensitivity.py#L264-L277)
- fixed row seed：
  [row-seed isolation](../../tests/test_m5_building_candidate_sensitivity.py#L279-L291)

## 直接執行 audit

```bash
.venv/bin/python scripts/audit_m5_building_candidate_sensitivity.py \
  --building-seeds 42 43 44 45 46 \
  --budgets 10 20 50 100 \
  --row-seed 42 \
  --model-seed 42 \
  --meter-min-source-buildings 2 \
  --meter-growth-per-transition 1 \
  --max-sampling-attempts 10000
```

這個 command 只建立 sampling/audit artifacts，不會 fit models。
