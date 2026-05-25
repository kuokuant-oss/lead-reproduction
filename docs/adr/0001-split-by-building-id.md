# Split train/validation by building_id, not by time or random shuffle

Train 和 validation 資料集依 `building_id` 為單位切分:同一棟建築的所有讀數只進 train 或 validation,不跨集混用。

這個設計是為了讓本地 validation 分數能可靠反映 Kaggle 測試集的性能:測試集是 206 棟**未見過的建築**,若用 random shuffle split,同棟建築的相鄰時間點會同時出現在 train 和 validation,造成資訊洩漏,validation AUC 虛高且無法指引模型迭代。論文(§2.3.1)報告此 split 方式的 validation 與 leaderboard 差距 < 1%。

若重現時改用 random split 或 time-based split,本地評估結果將失去參考意義,feature engineering 和超參數調整的方向會出錯。

---

Last reviewed: pending
