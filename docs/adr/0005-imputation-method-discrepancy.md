# Use mean imputation; README description appears incorrect

Missing value imputation 採用**每棟建築的 mean** 填補 NaN,以程式碼為準。

## Context

三個來源對 imputation 方法的描述不一致:

| 來源 | 說法 |
|------|------|
| 論文 §2.1 | "missing meter reading values were replaced with the **mean** value" |
| README.md | "Missing values (NaN) were replaced with the **median** value" |
| Feature generator notebook Cell 10 | `mean_reading = data.groupby('building_id').mean()['meter_reading']` → **mean** |

程式碼與論文一致,README 有誤。

## Decision

重現時使用 **mean per building** 填補 NaN,不使用 median。

## Consequence

- 若有人按 README 字面實作(median),結果會跟原始解法不同
- 任何引用 README 的決策都必須回到程式碼驗證
- 此 repo 已發現至少一處 README/程式碼不一致;不排除還有其他
- General principle:論文、README、程式碼三方不一致時,**以程式碼為準**(見 CONTEXT.md)

---

Last reviewed: 2026-05-25
