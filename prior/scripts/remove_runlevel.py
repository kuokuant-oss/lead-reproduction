"""Remove the run-level research direction from FINDINGS.md.

It was added on 2026-09-09 without the author's agreement. The author's actual
research question is why Tree leads TabPFN on PR-AUC, and PR-AUC is defined
hour-weighted, so changing the unit of analysis to runs answered a different
question. Everything about it comes out; the 2026-09-09 prose restyle stays,
because that was requested.

Removed:
  - the front-matter line announcing 4.5
  - section 4.5 in full
  - the cross-reference to 4.5 in the 4.4 tail
  - limitation 5 in section 6 (about 4.5's sample size), with renumbering
  - the zero_run_summer.py row in section 7
  - the run-level mentions in the portable-bundle note

Every replacement must apply exactly once or the script aborts.
"""
from __future__ import annotations
import sys
from pathlib import Path

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/FINDINGS.md")

src = P.read_text(encoding="utf-8")


def once(old: str, new: str) -> None:
    global src
    n = src.count(old)
    if n != 1:
        print(f"MATCH {n} (need 1): {old.strip()[:90]}")
        sys.exit(1)
    src = src.replace(old, new, 1)


# ---- front matter
once("本文數字皆由 2026-09-08 的 session 從 predictions.npz 重新計算。\n"
     "2026-09-09 新增 §4.5（長零讀數區塊的 run-level 判讀）。",
     "本文數字皆由 2026-09-08 的 session 從 predictions.npz 重新計算。")

# ---- section 4.5, sliced out between two sentinels
start = src.index("### 4.5 長零讀數區塊的判讀")
end = src.index("---\n\n## 5. 結論")
removed = end - start
src = src[:start] + src[end:]
print(f"removed section 4.5: {removed} chars")

# ---- 4.4 cross-reference
once("　→ **本節是全文各處「合併值與逐建築值量級不同」的共同成因**（§3.4 的季節性、§4.3 的 1241、\n"
     "　　§4.5 的 run-level 反轉皆屬此類），後續不再重述。",
     "　→ **本節是全文各處「合併值與逐建築值量級不同」的共同成因**（§3.4 的季節性、\n"
     "　　§4.3 的 building 1241 皆屬此類），後續不再重述。")

# ---- section 6: drop limitation 5, renumber 6 -> 5
once("本次分析有六項限制，依影響範圍排序。", "本次分析有五項限制，依影響範圍排序。")
once("""5. **§4.5 的 run-level 結論受樣本數限制。** 夏季 ≥168 h 的純標籤區塊只有 3 段異常對 20 段正常。
   bootstrap CI 在三個長度層級都排除 0、10/10 個 cell 同向、jackknife 亦成立，故**方向**可信；
   **幅度**在 n = 23 下不確定。要收緊區間需要跨年度資料以增加長區塊樣本。
6. **季節性目前只在族群層面成立。**""",
     """5. **季節性目前只在族群層面成立。**""")

# ---- section 7 script table
once("| `zero_run_summer.py` | **§4.5 的 run-level 分析**（在下述 GitHub 分支上，非本目錄） |\n", "")

# ---- portable bundle note
once("""### 可攜版本（2026-09-09）

§4.5 之後的工作已封裝成不依賴本機的自足包，推到 `kuokuant-oss/lead-reproduction`
的 orphan 分支 `analysis/hotwater-zero-run`（commit `22cd827`）。取得方式：""",
     """### 可攜版本（2026-09-09）

本文件的資料已封裝成不依賴本機的自足包，推到 `kuokuant-oss/lead-reproduction`
的 orphan 分支 `analysis/hotwater-prauc`。取得方式：""")

once("""git clone --depth 1 --single-branch -b analysis/hotwater-zero-run   https://github.com/kuokuant-oss/lead-reproduction.git hw""",
     """git clone --depth 1 --single-branch -b analysis/hotwater-prauc https://github.com/kuokuant-oss/lead-reproduction.git hw""")

once("""`scripts/verify_bundle.py`（逐 seed 對帳已定稿圖表）、`scripts/zero_run_summer.py`、
`out/` 的五份結果，以及 `prior/` 的本文件與報告頁原始碼。""",
     """`scripts/verify_bundle.py`（逐 seed 對帳已定稿圖表），以及 `prior/` 的本文件與
報告頁原始碼。""")

P.write_text(src, encoding="utf-8")

# ---- final audit
for pat in ("4.5", "run-level", "zero_run", "run level"):
    hits = src.count(pat)
    print(f"  remaining '{pat}': {hits}")
print(f"wrote {P}  ({len(src.splitlines())} lines)")
