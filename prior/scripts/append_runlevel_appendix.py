"""Put the run-level material back into FINDINGS.md, demoted to an appendix.

It was wrongly promoted to a section of the main argument (§4.5) on 2026-09-09.
The main line is the PR-AUC question, and PR-AUC is hour-weighted, so a
run-level statistic is a separate question rather than evidence about it.
It stays in the document because the question is worth asking on its own.
"""
from __future__ import annotations
import sys
from pathlib import Path

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/FINDINGS.md")

APPENDIX = """
---

## 附錄 A：長零讀數區塊的 run-level 判讀（次要問題）

**與主線的關係先講清楚。** 本文 §4 解釋的是 PR-AUC 的差距，而 PR-AUC 以正樣本加權、
逐小時計數：一段 500 小時的異常停機就是 500 個正樣本。本附錄改以**區塊**為單位
（每段一個分數、一個標籤），因此**不是關於 PR-AUC 的證據** —— 它回答的是另一個問題：
「一段長零讀數，模型能否判斷它是合法停機還是故障」。這個問題本身值得問，但它的答案
不能用來推論 PR-AUC，反之亦然。

改用 run-level 的理由：同一段停機內的小時近乎重複，hour-level 統計量的有效樣本數
遠低於名目 n。這對「判斷一段區塊」的問法是必要的；對 PR-AUC 則不適用，因為逐小時
加權正是 PR-AUC 的定義。

操作化：把每棟建築的序列切成連續 `reading=0` 的最大區塊；只留整段落在視窗內、長度
≥ MIN_LEN、且標籤純粹（全異常或全正常）的區塊；每個 run 的分數 = 該 run 內逐時分數
的平均；再算 run-level ROC-AUC。附 paired bootstrap 95% CI（對 run 重抽）、逐 cell
gap、drop-one-building jackknife。

夏季（6/7/8 月），K=400：

| MIN_LEN | runs（異常/正常） | Tree | TabPFN | gap | bootstrap 95% CI | 逐 cell |
|---|---|---|---|---|---|---|
| ≥ 24 h | 124（13 / 111） | 0.6909 | 0.9210 | **−0.2301** | [−0.383, −0.096] | Tree 領先 0/10 |
| ≥ 72 h | 54（8 / 46） | **0.4864** | 0.9212 | **−0.4348** | [−0.707, −0.170] | Tree 領先 0/10 |
| ≥ 168 h | 23（3 / 20） | **0.1667** | 0.8333 | **−0.6667** | [−0.905, −0.364] | Tree 領先 0/10 |

→ 在夏季的長零讀數區塊上 TabPFN 明顯較強。Tree 在 ≥72 h 降至隨機水準（0.4864），
　在 ≥168 h 為 0.1667 —— 低於隨機，意即它系統性地把合法的長停機排在真實故障之前。
　gap 隨區塊長度單調擴大。
→ 穩健性同向：median 聚合 −0.247 / −0.459 / −0.733；K=725 −0.233 / −0.410 / −0.483
　（逐 cell 0/5）；jackknife 在 ≥24 h 的 gap 範圍 [−0.296, −0.158]，移除最有影響的
　b1275 後仍為 −0.158。

對照視窗：

| 視窗 | ≥24 h gap | CI | 備註 |
|---|---|---|---|
| 冬季 12/1/2 月 | −0.0273 | [−0.122, +0.059] | 與 0 無法區分 |
| 冬季 ≥72 h、≥168 h | — | — | 不可評分：11 段長區塊全部異常，0 段合法 |
| 全年 ≥72 h | +0.0003 | [−0.048, +0.049] | 打平 |
| 全年 ≥168 h | +0.0061 | [−0.069, +0.075] | 打平 |

→ 「這段長零讀數是合法還是故障」這個判讀問題只存在於夏季：冬季的長零讀數區塊 11 段
　全部是異常，沒有需要分辨的對象。
→ 全年合併時兩者打平。

**樣本數限制**：≥168 h 只有 3 段異常對 20 段正常。bootstrap CI 在三個長度層級都排除 0、
10/10 個 cell 同向、jackknife 亦成立，故**方向**可信；**幅度**在 n = 23 下不確定。
要收緊區間需要跨年度資料以增加長區塊樣本。

**與主線可能的連結（假設，未驗證）**：兩件事可以同時成立 —— Tree 把「長區塊」整叢的
小時都推得比整片正常列更高（PR-AUC 需要的正是這個），代價是叢內排序差（把合法長停機
也一併推高）；TabPFN 叢內排序好但整叢位置偏低。PR-AUC 只在乎叢的位置，run-level AUC
只在乎叢內排序。若 §4 的分層歸因顯示 TabPFN 的 `C_s` 損失集中在 >7 天那一格，這個
假設就得到支持。**這是待驗證的假設，不是本附錄的結論。**

腳本：`zero_run_summer.py`（在可攜分支上，非本目錄）。
"""


def main() -> int:
    src = P.read_text(encoding="utf-8")
    if "附錄 A" in src:
        print("appendix already present")
        return 1
    # append after the very end of the document
    src = src.rstrip("\n") + "\n" + APPENDIX
    # restore the section-7 row, marked as secondary
    old = "| `show_per_seed.py` / `show_context.py` | 逐 seed 表格 / 實驗設定 |"
    new = ("| `show_per_seed.py` / `show_context.py` | 逐 seed 表格 / 實驗設定 |\n"
           "| `zero_run_summer.py` | **附錄 A** 的 run-level 分析（次要問題；在可攜分支上，非本目錄） |")
    if src.count(old) != 1:
        print("section-7 anchor not unique")
        return 1
    src = src.replace(old, new, 1)
    P.write_text(src, encoding="utf-8")
    print(f"appended appendix A ({len(APPENDIX)} chars); "
          f"file now {len(src.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
