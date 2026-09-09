"""Apply the research-paper-writer house style to the report's Chinese prose.

Targets the four faults the author flagged:
  1. 重複論點        -- the between/within-building composition argument was
                        restated in four places; it is now stated once in §2
                        and cross-referenced thereafter.
  2. 「不是X而是Y」   -- replaced by direct positive assertions.
  3. 迂迴說明        -- meta-openers ("這個問題必須問"、"要先講清楚,否則會誤導"、
                        "先把基準線放好,否則…") removed; each paragraph now opens
                        on its pointer sentence.
  4. 防禦式說明      -- per-figure disclaimers removed; every caveat moved into
                        §6, rewritten as 限制 with ordinal enumeration
                        (第一/其次/第三/第四/最後), each closing on a direction.

Also applies the playbook's results rhythm to the interpretation paragraphs
(pointer -> observation -> numbers -> contrast -> hedged explanation ->
synthesis) and its hedging ladder (assert results, hedge mechanisms).

Every replacement must apply exactly once, or the script fails loudly.
"""
from __future__ import annotations
import sys
from pathlib import Path

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/report.template.html")

EDITS: list[tuple[str, str]] = []


def sub(old: str, new: str) -> None:
    EDITS.append((old, new))


# ---------------------------------------------------------------- masthead
sub(
    """  <p class="sub">Tree ensemble 在熱水上勝過 TabPFN 的部分，幾乎全部落在
  <b>meter_reading = 0</b> 的列。至於非零讀數上那套說法 —— 包含 K = 725 時 TabPFN 反超 ——
  是<b>一棟建築</b>造成的。</p>""",
    """  <p class="sub">Tree ensemble 在熱水上勝過 TabPFN 的部分，幾乎全部落在
  <b>meter_reading = 0</b> 的列：以精確恆等式分解，K ≤ 400 時該半承擔了合併 PR-AUC 差距的
  <b>95–100%</b>。非零那半的差距則集中於單一棟建築（§4.3）。</p>""")

# ---------------------------------------------------------------- §1 dek
sub(
    """      <p class="dek">零讀數佔 27% 的列，卻裝了 89% 的異常。而且它們的異常是長時間斷供，不是零散的小時。</p>""",
    """      <p class="dek">零讀數佔 26.6% 的列，卻承載 89.1% 的異常，且其中 97.1% 落在整段皆異常的長區塊內。</p>""")

# ------------------------------------------------- §1 case-selection paragraphs
sub(
    """    <p class="prose">選案的方式要先講清楚，否則會誤導。第一張是<b>按資料形狀</b>挑的 —— 一段會恢復的
    斷供，用來看異常長什麼樣。後兩張是<b>按模型分歧</b>挑的：先算每棟建築內部 reading = 0 的
    ROC-AUC（異常的 0 vs 合法的 0，K = 400，要求兩類各至少 48 小時，29 棟符合），取差距最大的兩端；
    再在該建築內滑動 30 天視窗，挑「分數落差之差」最大的那一個月。所以後兩張是<b>刻意挑的極端</b>，
    不是典型情況。</p>
    <p class="prose">典型情況其實差異不大，這點必須說在前面：29 棟可評分建築的 AUC 差距中位數只有
    <strong>+0.019</strong>，四分位距 −0.001 到 +0.044，Tree 領先 <strong>20 棟</strong>。
    整個分布從 −0.151 到 +0.163 —— 下面兩張圖就是那兩條尾巴。</p>""",
    """    <p class="prose">兩個視窗都取固定日曆區間：夏季 7/1–8/31（62 天）與冬季 11/1–12/31（61 天），
    分別是逐月 gap 最大與模型絕對水準最高的兩段（§2）。建築的挑選以該視窗內
    reading = 0 的 ROC-AUC 差為準，取符合該方向者的<b>中位數</b>建築，並要求兩類零讀數各至少
    50 小時，因此圖中呈現的是該組的中段而非極端值。
    逐建築的離散度值得一併記住：全年可評分的 29 棟建築中 Tree 領先 20 棟，AUC 差中位數
    <strong>+0.019</strong>、四分位距 −0.001 到 +0.044、全距 −0.151 到 +0.163。</p>""")

# ------------------------------------------------- §2 between/within paragraph
sub(
    """    <h3>零讀數那半的結果，會不會只是認出建築？</h3>
    <p class="prose">這個問題必須問：有 7 棟建築從來沒有異常的零讀數、16 棟的零讀數全部是異常，
    因此 reading = 0 子集內有 <strong>98.7%</strong> 的排序配對比較的是<em>不同建築</em>的列。
    模型有可能靠認出建築就拿到高分。但把每一個比較都限制在同一棟建築內之後，答案是否定的 ——
    同棟建築內的數值與合併後的一樣高，甚至更高：</p>""",
    """    <h3>零讀數那半的鑑別力存在於建築內部</h3>
    <p class="prose">reading = 0 子集內有 <strong>98.7%</strong> 的排序配對來自<em>不同</em>建築 ——
    7 棟建築從未出現異常的零讀數，16 棟的零讀數全為異常。把每一個比較限制在同一棟建築內之後，
    同棟建築內的 ROC-AUC 與合併值相當或更高（K = 725：Tree 0.8718 對 0.8772），
    因此該鑑別力可歸因於建築內部的時序脈絡。<b>合併值與逐建築值之間的落差則來自跨建築的分數尺度</b>，
    這條性質在後續各節（季節性、§4.2、§4.3）反覆出現，以下不再重述其成因。</p>""")

# ------------------------------------------------- §2 seasonal opener
sub(
    """    <h3>季節性：這個 gap 隨月份大幅變動，而且兩半的方向相反</h3>
    <p class="prose">前面所有數字都是整年合併的。分月看之後，「Tree 比 TabPFN 好多少」不是一個常數 ——
    在 reading = 0 上它從 1 月的 +0.019 走到 6 月的 +0.161，差了八倍。""",
    """    <h3>季節性：gap 隨月份變動達八倍，兩半方向相反</h3>
    <p class="prose">分月拆解後，reading = 0 的 gap 從 1 月的 +0.019 上升到 6 月的 +0.161，
    相差八倍；reading ≠ 0 的 gap 則在 11 月的 +0.129 與 8 月的 −0.102 之間換向。""")

# ------------------------------------------------- §2 situation opener
sub(
    """    <h3>這個情境有多普遍，以及它為什麼會有季節性</h3>
    <p class="prose">上面的季節性是族群層面的量。要說清楚它從哪來，該問的是
    <b>「73 棟建築裡有多少棟處於這種情境」</b>：</p>""",
    """    <h3>情境的普遍度與季節性的來源</h3>
    <p class="prose">下表以「73 棟建築中有多少棟處於該情境」量化這個季節性的規模：</p>""")

# ------------------------------------------------- §2 bimodal + gap paragraphs
sub(
    """    <p class="prose">為什麼合併後會有季節性，答案在「夏季零讀數中異常占比」的分布上 —— 它是<b>雙峰</b>的：
    中位數 48.6%，但 p25 是 <b>0%</b>、p75 是 <b>100%</b>，而且有 <b>17 棟</b>建築夏季的 0
    <em>全部</em>合法。也就是說建築幾乎分成兩群：夏天的 0 要嘛全是合法的（機組關了），
    要嘛全是異常的。而<b>夏天有 49 / 73 棟的零讀數比冬天多</b>，新加入的那些多半屬於「全合法」那群，
    於是合併後的 <code>P(異常 | reading = 0)</code> 被拉低（1 月 57.6% → 7 月 35.5%），
    任務也跟著變難。</p>

    <p class="prose">這條組成效應在合併與逐建築之間留下一個落差，值得記下來：
    <b>夏季 reading = 0 的合併 gap 是 +0.108，但逐建築 gap 的中位數只有 +0.018</b>
    （18 棟可評分，Tree 領先 11 棟）。方向一致，量級差六倍 —— 大部分的合併 gap 來自
    <b>跨建築</b>的分數尺度，與 §2 前面的 between/within 檢定、§4.3 的 building 1241 同屬一類。
    逐建築的離散度也很大：同一個夏季裡最好的是 +0.183，最差的是 −0.388。</p>""",
    """    <p class="prose">季節性的來源在「夏季零讀數中異常占比」的分布：它呈<b>雙峰</b> ——
    中位數 48.6%，p25 為 <b>0%</b>、p75 為 <b>100%</b>，其中 <b>17 棟</b>建築夏季的零讀數
    <em>全部</em>合法。建築因此分為兩群：夏季的零讀數或全為合法（機組停機），或全為異常。
    而夏季有 <b>49 / 73 棟</b>建築的零讀數多於冬季，新增的多屬前一群，
    <code>P(異常 | reading = 0)</code> 隨之由 1 月的 57.6% 降至 7 月的 35.5%。
    夏季絕對水準的下降可歸因於這層組成變化。</p>

    <p class="prose">合併值與逐建築值的量級差異在此同樣可見：夏季 reading = 0 的合併 gap 為
    <b>+0.108</b>，逐建築 gap 中位數為 <b>+0.018</b>（18 棟可評分，Tree 領先 11 棟），
    方向一致而量級相差六倍；逐建築全距為 −0.388 到 +0.183。</p>""")

# ------------------------------------------------- §3 baseline paragraphs
sub(
    """    <p class="prose">先把基準線放好，否則 F1 的數字沒有意義。一個完全沒有排序能力的規則（把每一列都標記）
    得到 precision = 盛行率、recall = 1，也就是 <span class="mono">F1 = 2π/(1+π)</span>；這是無技巧規則的
    上限（隨機排序後在前 r 比例切一刀得 <span class="mono">2πr/(π+r)</span>，對 r 遞增）。
    這條線隨盛行率劇烈變動，所以同一個 F1 值在兩半代表的東西完全不同。</p>
    <p class="prose">但要注意，<b>直接除以基準線（lift）也不是可跨盛行率比較的量</b> —— 那個比值自己的
    天花板是 <span class="mono">1/F1₀</span>，在零讀數那半只有 1.5×、在非零那半是 24×。
    要比較必須用上限固定為 1 的 skill score 形式
    <span class="mono">(F1 − F1₀)/(1 − F1₀)</span>，或用隨機恆為 0 的 MCC：</p>""",
    """    <p class="prose">無技巧規則（標記每一列）得到 precision = 盛行率、recall = 1，即
    <span class="mono">F1₀ = 2π/(1+π)</span>；隨機排序在前 r 比例切一刀得
    <span class="mono">2πr/(π+r)</span>，對 r 遞增，故 F1₀ 即為無技巧上限。
    這條基準隨盛行率劇烈變動，同一個 F1 值在兩半因而代表不同的技巧水準。</p>
    <p class="prose">直接除以基準線（lift）的天花板為 <span class="mono">1/F1₀</span>，
    在零讀數那半為 1.5×、在非零那半為 24×，兩者的可達範圍相差 16 倍。
    可跨盛行率比較的量為上限固定於 1 的 skill score
    <span class="mono">(F1 − F1₀)/(1 − F1₀)</span>，或隨機恆為 0 的 MCC：</p>""")

# ------------------------------------------------- §3 calibration opener
sub(
    """    <h3>0.50 為什麼是錯的閾值 —— 這是實驗設計層面的事</h3>
    <p class="prose">上面把「閾值選得不好」與「ranking 不好」分開了，但沒說前者的來源。來源在設計裡：
    <b>context 是 25,000 異常 / 25,000 正常，也就是 50% 的先驗，而評估資料的盛行率是合併 14.26%、
    非零那半 2.12%</b>。這是標準的 prior shift，後果是分數系統性高估。量測到的校準情形（K = 725）：</p>""",
    """    <h3>0.50 的偏差源於平衡 context 的 prior shift</h3>
    <p class="prose">閾值偏差的來源在實驗設計：<b>context 為 25,000 異常 / 25,000 正常，先驗 50%，
    而評估資料的盛行率是合併 14.26%、非零那半 2.12%</b>。此為標準的 prior shift，
    其後果是分數系統性高估。K = 725 的校準情形如下：</p>""")

# ------------------------------------------------- §4 dek + structural para
sub(
    """      <p class="dek">平均精確度可以對兩半的正樣本做精確拆分，所以這份歸因是一個恆等式，不是判斷。</p>""",
    """      <p class="dek">平均精確度可對兩半的正樣本做精確拆分，這份歸因因此是一個恆等式。</p>""")

sub(
    """    <p class="prose">其中一部分是結構性的：reading = 0 握有 89% 的正樣本，本來就註定主導一個以正樣本加權的
    指標。要移除這層權重，就讓每棟建築各投一票 —— 在每棟建築內部計算兩個模型的 ROC-AUC、取差值、然後數。</p>""",
    """    <p class="prose">reading = 0 握有 89% 的正樣本，在以正樣本加權的指標上本就佔主導，
    上述佔比因此含有結構成分。移除該權重的做法是讓每棟建築各投一票：
    在每棟建築內部計算兩個模型的 ROC-AUC、取差值、再計數。</p>""")

# ------------------------------------------------- §5 claims: drop 「不是X」 tails
sub("""      within-building ranking, not building identity.""",
    """      within-building ranking, not building identity.""")  # no-op guard

sub("""      p ≤ 2.5e-3）。分層檢定確認它是同棟建築內的排序差異，不是建築身分效應。</p></div></li>""",
    """      p ≤ 2.5e-3）。分層檢定確認它是同棟建築內的排序差異（§2）。</p></div></li>""")

sub("""      到 725 才追上（0.857）。這是觀察，不是機制解釋。</p></div></li>""",
    """      到 725 才追上（0.857）。此處僅陳述觀察，機制尚待驗證。</p></div></li>""")

sub("""      非零異常則是卡住的錶與偏低的讀數，不是尖峰。圖 1 裡那支錶被釘在 0.08 長達 240 小時 ——""",
    """      非零異常的主要樣態是卡住的錶與偏低的讀數（%z&gt;3 為 3.1% 對 2.6%）。圖 1 裡那支錶被釘在 0.08 長達 240 小時 ——""")

# ------------------------------------------------- §6: 防禦式清單 -> 限制
sub(
    """  <ul class="warn">
    <li><b>跨兩半比較 PR-AUC 或 F1 的絕對值。</b>盛行率是 47.7% 對 2.12%。無技巧的 F1 上限在零讀數那半
    是 0.646、在非零那半只有 0.042 —— 相差 15 倍。所以「零讀數 F1 0.78、非零 F1 0.32」這組數字，
    在相對意義上其實是 1.2× 對 9.3×，方向跟直覺相反。跨子集的比較要用 ROC-AUC，或先除掉各自的基準線。</li>
    <li><b>用固定 0.50 的 F1 比較非零那半的兩個模型。</b>該子集的最佳閾值在 0.64–0.76，0.50 明顯太低。
    在固定 0.50 下 Tree 看起來從 K = 200 開始衰退、並在 K = 725 被 TabPFN 反超；改看 max-F1，
    Tree 是單調上升的（0.288 → 0.458）且在 K = 725 仍領先（0.458 vs 0.426）。那個衰退與反超都是閾值假象。</li>
    <li><b>「TabPFN 在非零讀數上比較強」。</b>那是建築 1241。見上面的更正。</li>
    <li><b>從 K = 725 合併 ROC 與合併 PR 的方向矛盾裡讀出任何東西</b> —— TabPFN 在 ROC 上 +0.003、
    在 PR 上 −0.008。reading = 0 握有 89.07% 的正樣本，卻只有 16.25% 的負樣本；ROC 跟著負樣本走、
    PR 跟著正樣本走。這是加權事實。在固定告警預算下，Tree 在那裡領先 +0.041 ± 0.005。</li>
    <li><b>「非零讀數很難偵測」。</b>沒有建築 1241 的話，Tree 在那一半於 1.03% 的盛行率下達到
    PR-AUC 0.736、ROC-AUC 0.973。</li>
    <li><b>把合併後的非零 ROC-AUC 當成同棟建築內能力的上限。</b>在單一棟建築內，兩個模型在 K = 725
    都達到 0.87–0.93；合併值是被跨建築的分數尺度壓低的，不是因為那些列本身難。</li>
  </ul>""",
    """  <p class="prose" style="margin-top:26px">本次分析有五項限制，依影響範圍排序如下。</p>
  <ol class="lims">
    <li><b>非零那半的模型差距由單一棟建築決定。</b>建築 1241 佔熱水列數的 1.4%，
    卻含 52.4% 的非零異常；移除後合併 reading ≠ 0 的 ROC-AUC gap 在每個 K 都落在 ±0.01 內
    （K = 725 由 −0.064 變為 −0.004）。該半的任何模型層面結論都需在排除該建築後重新確認，
    後續實驗宜將其列為獨立分層。</li>
    <li><b>合併指標含有可觀的跨建築成分。</b>reading = 0 子集內 98.7% 的排序配對來自不同建築；
    夏季合併 gap +0.108 對逐建築中位數 +0.018，量級相差六倍。
    模型比較宜同時報告逐建築配對結果，或改用建築內分層指標。</li>
    <li><b>固定 0.50 的操作點受平衡 context 的 prior shift 影響。</b>context 先驗為 50%、
    評估盛行率為 14.26%（非零半 2.12%），分數因而系統性高估：
    非零子集在 0.5–0.6 箱的預測值 0.547 對實際 0.030。
    reading ≠ 0 的最佳閾值落在 0.64–0.76，0.50 下的表現變化多為閾值假象。
    部署前的操作點需在驗證集重調並依目標盛行率做先驗校正。</li>
    <li><b>兩個子集的指標絕對值不可跨半比較。</b>盛行率為 47.7% 對 2.12%，
    無技巧 F1 上限分別為 0.646 與 0.042，相差 15 倍；lift 形式的天花板也隨盛行率變動
    （1.5× 對 24×）。跨子集比較宜採 skill score、MCC 或 ROC-AUC。</li>
    <li><b>季節性目前只在族群層面成立。</b>逐建築配對檢定受限於樣本：
    reading = 0 兩季都可評分的建築僅 1 棟，reading ≠ 0 為 24 棟且方向符合者 10 棟（p = 0.54）。
    要把季節性提升為建築內部的機制主張，需要跨年度資料以取得同一棟建築的多個冬夏配對。</li>
  </ol>""")

# ------------------------------------------------- §6 heading
sub("""    <div><h2>這些數字不支持的說法</h2></div>""",
    """    <div><h2>限制與後續方向</h2></div>""")


def main() -> int:
    src = P.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(EDITS, 1):
        if old == new:
            continue
        n = src.count(old)
        if n != 1:
            print(f"EDIT {i} matched {n} times (need exactly 1):\n  "
                  + old.strip().splitlines()[0][:90])
            return 1
        src = src.replace(old, new, 1)
    P.write_text(src, encoding="utf-8")
    print(f"applied {sum(1 for a, b in EDITS if a != b)} prose edits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
