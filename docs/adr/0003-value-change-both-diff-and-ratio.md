# Include both difference and ratio value-change features

Value-change features 同時包含兩種形式:差值 `X(t) - X(t-s)` 和比值 `(X(t)+1)/(X(t-s)+1)`。雖然兩者衡量的現象相似,但論文(§2.2.1)明確說明:「the result shows that the inclusion of both features could achieve the best prediction performance, so they are both retained.」

兩者互補的直覺:差值對絕對變化敏感,比值對相對變化敏感且在不同量級的時間序列間具有一致性(因此 ratio features 佔據 Top 10 中的四個位置,而 difference features 未入列,但論文仍建議保留兩者)。

若重現時只保留比值或只保留差值以節省特徵數或計算量,整體 AUC 會下降。應兩者都建。

---

Last reviewed: pending
