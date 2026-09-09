"""Pass 2 on FINDINGS.md: the last 「不是」 tail, and §7 gains the 2026-09-09
scripts plus the GitHub branch that carries the portable bundle."""
from pathlib import Path
import sys

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/FINDINGS.md")
E = []
E.append((
    "  而新加入者的 0 大多合法,**合併比率被組成拉低,不是每棟建築內部變容易**。",
    "  而新加入者的 0 大多合法,**合併比率因此被組成拉低**；每棟建築內部的難度並未下降。"))
E.append((
    "| `show_per_seed.py` / `show_context.py` | 逐 seed 表格 / 實驗設定 |",
    "| `show_per_seed.py` / `show_context.py` | 逐 seed 表格 / 實驗設定 |\n"
    "| `season_situation.py` | 夏季情境的逐建築普遍度（x/73），寫 `data/season_situation.json` |\n"
    "| `extract_rn_season_cases.py` | 報告頁圖 3a/3b 的兩個季節個案（各兩個月） |\n"
    "| `restyle_prose*.py` · `restyle_findings*.py` | 依 research-paper-writer 風格改寫報告頁與本文件的行文 |\n"
    "| `zero_run_summer.py` | **§4.5 的 run-level 分析**（在下述 GitHub 分支上，非本目錄） |"))
E.append((
    "操作提醒：\n- 呼叫 `wsl.exe` 用 PowerShell 工具、完整字面路徑（Bash 工具會改寫 `/mnt/c` 並吃掉 `$VAR`）。",
    """### 可攜版本（2026-09-09）

§4.5 之後的工作已封裝成不依賴本機的自足包，推到 `kuokuant-oss/lead-reproduction`
的 orphan 分支 `analysis/hotwater-zero-run`（commit `22cd827`）。取得方式：

```bash
git clone --depth 1 --single-branch -b analysis/hotwater-zero-run \
  https://github.com/kuokuant-oss/lead-reproduction.git hw
cd hw && python scripts/verify_bundle.py    # 須出現 BUNDLE VERIFY PASS
```

包含 `data/hotwater_bundle.npz`（33 MB：標籤、讀數、建築、hour、epoch 時間，
以及兩模型在 K=400 十格與 K=725 五格的逐 cell 分數，uint16 量化）、
`scripts/verify_bundle.py`（逐 seed 對帳已定稿圖表）、`scripts/zero_run_summer.py`、
`out/` 的五份結果，以及 `prior/` 的本文件與報告頁原始碼。

⚠ 量化的已知後果：ROC 這類排序指標無損（最大差 5.4e-7），**average precision 對 tie
敏感，只到 1.8e-4**。需要精確 AP 的工作須回到原始 `predictions.npz`。

### 操作提醒

- 呼叫 `wsl.exe` 用 PowerShell 工具、完整字面路徑（Bash 工具會改寫 `/mnt/c` 並吃掉 `$VAR`）。"""))
E.append((
    "  改成前景執行、由 harness 的 background task 托住，log 寫到 `/mnt/c/...` 下。",
    """  改成前景執行、由 harness 的 background task 托住，log 寫到 `/mnt/c/...` 下。
- **`np.savez_compressed` 寫到 `/mnt/c` 可能印出成功卻沒有檔案。** 中間檔存在 WSL 檔系統
  （如 `~/.cache/`），只把人類要讀的輸出放 `/mnt/c`。
- pandas 為 3.0.x，`series.astype("int64") // 10**9` 不再是 epoch 秒（Series 是
  `datetime64[us]`，會塌成 0）。改用 `series.astype("datetime64[s]").astype("int64")`。"""))

src = P.read_text(encoding="utf-8")
for i, (old, new) in enumerate(E, 1):
    n = src.count(old)
    if n != 1:
        print(f"EDIT {i} matched {n} (need 1): {old.strip()[:80]}")
        sys.exit(1)
    src = src.replace(old, new, 1)
P.write_text(src, encoding="utf-8")
print(f"applied {len(E)} edits (pass 2)")
