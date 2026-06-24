# Post-processing with two hard override rules

Ensemble 輸出後套用兩條 hard rule 覆蓋模型預測(§2.4):

1. **`meter_reading == 1` → 強制預測異常(1)**: 根據競賽討論區的觀察,幾乎 100% 的 `meter_reading = 1` 都是異常。這是 LEAD dataset 特有的 artifact,不是通用規律。
2. **時間序列 start/end points → 強制預測正常(0)**: 視覺化觀察顯示各電表的時間序列起始和結束點幾乎都不是異常。

這兩條規則來自資料特性的人工觀察,不是模型學到的,且效果不可逆(覆蓋掉模型原本的機率輸出)。若重現時省略:Rule 1 會讓 `meter_reading = 1` 的那些行被模型以較低信心評分,直接損失 precision;Rule 2 的影響較小但仍可能引入邊界噪音。

注意:Rule 2 的「start/end points」定義論文未說明,需看 GitHub 確認邊界定義(見 docs/reference/unknowns.md #3)。

---

Last reviewed: pending
