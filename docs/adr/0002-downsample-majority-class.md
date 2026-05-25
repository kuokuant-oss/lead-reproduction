# Downsample normal class to 50:50 ratio (not oversampling, not class weights)

正常資料佔約 95%,異常資料佔約 5%,class imbalance 嚴重。解法是對**正常資料做 downsampling**,使 normal:abnormal = 50:50,而非使用 SMOTE、oversampling 或 class_weight 參數。

論文(§2.3.2)直接採用 downsampling 且未說明其他方法的比較結果,但競賽對照(Table 3)提供了間接驗證:所有未處理 class imbalance 的解法,AUC 均低於 0.90,與第一名的 0.9866 差距超過 10%。

若重現時略過 imbalance 處理,模型會偏向預測 0(正常),AUC 會顯著下滑。若改用 SMOTE 或 class_weight,結果是否等效論文未說明,需實驗驗證。

---

## Empirical mismatch

實測 LEAD dataset(Issue #7,2026-05-25):

- **論文宣稱**: 異常率約 5%(majority class 約 95%)
- **實測值**: 異常率 **2.13%**(majority class 97.87%)

差距約 2.5 倍,比論文描述更不平衡。影響:

- 若照論文做 50:50 downsampling,正常資料需被壓縮到異常資料的數量,訓練集規模將大幅縮小
- M2 重現時需驗證 50:50 是否仍為最佳比例,或需根據實際 2.13% 重新評估

本 ADR 記錄的決策(使用 downsampling 而非 oversampling/class_weight)仍維持,此段為補充實測資訊。

---

Last reviewed: 2026-05-25
