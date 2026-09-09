"""Second restyle pass: the JS-generated captions/notes and the remaining
「不是X」 tails, plus the CSS for the ordinal 限制 list added in pass one.

Same rule as pass one: every replacement must apply exactly once.
"""
from __future__ import annotations
import sys
from pathlib import Path

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/report.template.html")
EDITS: list[tuple[str, str]] = []


def sub(old: str, new: str) -> None:
    EDITS.append((old, new))


# ---------------------------------------------------------------- CSS for §6
sub(
    """.warn{list-style:none;padding:0;margin:26px 0 0}""",
    """.lims{margin:22px 0 0;padding:0 0 0 1.6em;max-width:38em}
.lims li{margin-bottom:16px;font-size:15.5px;color:var(--ink-2);line-height:1.78}
.lims li::marker{font-family:var(--ff-mono);font-size:13px;color:var(--tree);font-weight:600}
.lims li b{color:var(--ink);font-weight:600}
.warn{list-style:none;padding:0;margin:26px 0 0}""")

# ---------------------------------------------------------------- fig 1 caption
sub(
    """    同一個物理故障，但 0.08 不是 0 —— 於是這 240 列落進 <em>reading ≠ 0</em> 那一半，
    而它們的鄰居落進 <em>reading = 0</em> 那一半。本頁後面量測的，就是這條切分線造成的差異，
    而不是物理上的差異。</figcaption>""",
    """    同一個物理故障，卻因為 0.08 大於 0 而落進 <em>reading ≠ 0</em> 那一半，
    鄰近的整數 0 則落進 <em>reading = 0</em> 那一半。本頁量測的是這條切分線造成的差異。</figcaption>""")

# ---------------------------------------------------------------- §1.2 heading + para
sub("""    <h3>非零讀數：卡住的錶與掉落，不是尖峰</h3>""",
    """    <h3>非零讀數的主要樣態是卡住的錶與讀數掉落</h3>""")

sub(
    """    計算 robust z 分數，異常並不是尖峰 —— 3.1% 超過 z&nbsp;&gt;&nbsp;3，正常是 2.6%，基本上沒有差別 ——
    但它們偏低的機率是六倍：<strong>11.3%</strong> 低於 z&nbsp;=&nbsp;−1，正常只有 <strong>1.9%</strong>。
    兩者的中位數重合。與一般讀數高度重疊，正是這一半 PR-AUC 偏低的原因。</p>""",
    """    計算 robust z 分數，尖峰方向兩類幾乎相同（z&nbsp;&gt;&nbsp;3 為 3.1% 對 2.6%），
    偏低方向則相差六倍（z&nbsp;&lt;&nbsp;−1 為 <strong>11.3%</strong> 對 <strong>1.9%</strong>），
    兩者中位數重合。這一半 PR-AUC 偏低可歸因於此分布重疊。</p>""")

# ---------------------------------------------------------------- §2 stratified tnote
sub(
    """      <div class="tnote">在零讀數上，同棟建築內的數值與合併值一致，所以那份鑑別力是真的存在於建築內部，
      不是建築身分的洩漏。在非零讀數上，同棟建築內的數值在每個 K 都<em>遠高於</em>合併值（+0.04 到 +0.10）——
      合併後的非零數字是被跨建築對不齊的分數尺度拉低的，低估了兩個模型在單一棟建築內的實際表現。</div>""",
    """      <div class="tnote">零讀數上同棟建築內的數值與合併值一致，該鑑別力因此存在於建築內部。
      非零讀數上同棟建築內的數值在每個 K 都<em>遠高於</em>合併值（+0.04 到 +0.10）：
      合併值低估了兩個模型在單一棟建築內的排序能力，落差可歸因於跨建築的分數尺度。</div>""")

# ---------------------------------------------------------------- §3 dek
sub(
    """      <p class="dek">固定閾值 0.50 在合併集上是可比的：兩個模型都標記 9–18% 的列，真實異常率是 14.26%。
      但在兩個子集上，它都不是各自的最佳操作點 —— 而且 F1 的絕對值在兩半之間完全不可比。</p>""",
    """      <p class="dek">固定閾值 0.50 在合併集上可比：兩個模型都標記 9–18% 的列，真實異常率 14.26%。
      在兩個子集上，最佳閾值分別落在別處，且 F1 的絕對值不可跨半比較。</p>""")

# ---------------------------------------------------------------- §3 calibration tnote tail
sub(
    """      <b>所以 0.50 不是「稍微偏低」，而是由平衡 context 訓練、不平衡資料評估這個設計決定的錯閾值。</b>
      任何要部署的操作點都應該在驗證集上重調，並按目標盛行率做先驗校正。</div>""",
    """      <b>0.50 的偏差量因此由「平衡 context 訓練、不平衡資料評估」這個設計決定，屬於系統性偏差。</b>
      部署用的操作點需在驗證集重調，並按目標盛行率做先驗校正。</div>""")

# ---------------------------------------------------------------- §5 claim 3
sub("""      <h4>非零那一欄不是一個模型層面的發現。</h4>""",
    """      <h4>非零那一欄由單一棟建築決定。</h4>""")

# ---------------------------------------------------------------- JS: tblcasesnote
sub(
    """    `<b>合併與逐建築在 r0 上方向一致（都偏 Tree），但量級差很多</b>：冬季合併
     ${sg(w.pooled["400|r0"].gap)} vs 逐建築中位數 ${sg(w.median_gap)}；夏季合併
     ${sg(s.pooled["400|r0"].gap)} vs 逐建築中位數 ${sg(s.median_gap)}。
     多出來的部分來自跨建築的分數尺度（見上文 between/within）。<br>
     <b>reading ≠ 0 才是季節方向會翻轉的那一半</b>：冬季 ${sg(w.pooled["400|rN"].gap)}（Tree），
     夏季 ${sg(s.pooled["400|rN"].gap)}（TabPFN）。但 §4.3 提醒過，非零那半的 gap 幾乎全來自
     building 1241，所以這個翻轉不宜當成模型性質。<br>
     <b>難度的資料來源在最後兩欄</b>：冬季零讀數率只有 ${(w.pooled.zero_rate * 100).toFixed(1)}%
     但其中 ${(w.pooled.p_anom_zero * 100).toFixed(1)}% 是異常；夏季零讀數率
     ${(s.pooled.zero_rate * 100).toFixed(1)}%、只有 ${(s.pooled.p_anom_zero * 100).toFixed(1)}% 是異常。
     0 變多而且多半合法，就是夏天 r0 絕對水準掉下來的原因。`;""",
    """    `表中三個層級指向同一方向而量級不同。r0 上合併與逐建築中位數皆偏 Tree：冬季
     ${sg(w.pooled["400|r0"].gap)} 對 ${sg(w.median_gap)}，夏季 ${sg(s.pooled["400|r0"].gap)}
     對 ${sg(s.median_gap)}，差額來自跨建築的分數尺度（§2）。<br>
     季節方向換向出現在 reading ≠ 0：冬季 ${sg(w.pooled["400|rN"].gap)}、
     夏季 ${sg(s.pooled["400|rN"].gap)}。該半的 gap 幾乎全數來自 building 1241（§4.3），
     此換向的解釋需以排除該建築後的結果為準。<br>
     最後兩欄給出難度的資料來源：冬季零讀數率 ${(w.pooled.zero_rate * 100).toFixed(1)}%、
     其中 ${(w.pooled.p_anom_zero * 100).toFixed(1)}% 為異常；夏季零讀數率
     ${(s.pooled.zero_rate * 100).toFixed(1)}%、其中僅 ${(s.pooled.p_anom_zero * 100).toFixed(1)}% 為異常。
     整體而言，夏季 r0 絕對水準的下降可歸因於零讀數變多且多數合法。`;""")

# ---------------------------------------------------------------- JS: op-table notes
sub(
    """    ? `<b>0.50 在這個子集上明顯太低</b>，最佳閾值落在 0.64–0.76。這一點會製造假象：Tree 的 F1@0.50 從
       K = 200 的 0.388 掉到 K = 725 的 0.324，看起來像是變差了，但它的 max-F1 其實<em>單調上升</em>
       —— 0.288 → 0.412 → 0.424 → 0.451 → 0.458。分數分布隨 K 整體上移，同一條 0.50 就越來越寬鬆，
       於是在非零列多標了卻沒多抓到。同理，K = 725 時 TabPFN 在 F1@0.50 上的「反超」（0.354 vs 0.324）
       在 max-F1 下反轉回來（0.426 vs 0.458）。<b>不要用固定 0.50 的 F1 比較這一半的兩個模型。</b>`""",
    """    ? `<b>此子集的最佳閾值落在 0.64–0.76，0.50 明顯偏低。</b>在固定 0.50 下 Tree 的 F1 由
       K = 200 的 0.388 降至 K = 725 的 0.324；其 max-F1 則<em>單調上升</em>
       （0.288 → 0.412 → 0.424 → 0.451 → 0.458）。相對地，K = 725 時 TabPFN 在 F1@0.50 上的
       0.354 對 0.324 領先，在 max-F1 下換向為 0.426 對 0.458。
       兩處變化可歸因於分數分布隨 K 整體上移，使同一條 0.50 逐漸寬鬆。
       整體而言，此子集的模型比較宜採 max-F1 或重調後的操作點（§6 限制三）。`""")

sub(
    """    : `在這個閾值下，兩個模型都標記 9% 到 18% 的列，真實率為 14.26%，而且彼此相差 1–3 個百分點 ——
       所以 0.50 在合併集上是一個可比的操作點，而不是隨手挑的（K = 725 時兩者分別標記 18.01% 與 17.90%），
       headroom 也只有 0.01–0.16。但這份可比性不會傳遞到子集：見另外兩個分頁。`;""",
    """    : `兩個模型在此閾值下標記 9% 到 18% 的列，彼此相差 1–3 個百分點，
       真實率為 14.26%（K = 725 時分別標記 18.01% 與 17.90%），headroom 為 0.01–0.16。
       0.50 因此在合併集上構成可比的操作點。此可比性不傳遞到子集，見另外兩個分頁。`;""")

# ---------------------------------------------------------------- §1 case intro tail
sub("""三個個案的讀數量級差很多，各自用自己的 Y 軸，不要跨圖比高度。""",
    """兩個個案的讀數量級不同，各自使用獨立的 Y 軸，圖間高度不可比。""")


def main() -> int:
    src = P.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            print(f"EDIT {i} matched {n} times (need 1):\n  "
                  + old.strip().splitlines()[0][:100])
            return 1
        src = src.replace(old, new, 1)
    P.write_text(src, encoding="utf-8")
    print(f"applied {len(EDITS)} prose edits (pass 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
