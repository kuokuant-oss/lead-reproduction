# Downsample normal class to 50:50 ratio (not oversampling, not class weights)

正常資料佔約 95%,異常資料佔約 5%,class imbalance 嚴重。解法是對**正常資料做 downsampling**,使 normal:abnormal = 50:50,而非使用 SMOTE、oversampling 或 class_weight 參數。

論文(§2.3.2)直接採用 downsampling 且未說明其他方法的比較結果,但競賽對照(Table 3)提供了間接驗證:所有未處理 class imbalance 的解法,AUC 均低於 0.90,與第一名的 0.9866 差距超過 10%。

若重現時略過 imbalance 處理,模型會偏向預測 0(正常),AUC 會顯著下滑。若改用 SMOTE 或 class_weight,結果是否等效論文未說明,需實驗驗證。

---

Last reviewed: pending
