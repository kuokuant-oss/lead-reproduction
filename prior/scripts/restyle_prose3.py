"""Third restyle pass: the last four 「不是X」 tails and the remaining 「必須」.

「模型必須從…挑出」is left in place: it states what the task requires, not a
defensive aside, so it is retained.
"""
from __future__ import annotations
import sys
from pathlib import Path

P = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
         "/analysis_2026-09-08/report.template.html")
EDITS: list[tuple[str, str]] = []


def sub(old: str, new: str) -> None:
    EDITS.append((old, new))


sub("""    <h3>零讀數：帶訊號的是區塊長度，不是時段</h3>""",
    """    <h3>零讀數的訊號來自區塊長度</h3>""")

sub("""    要分辨故障的 0 與合法的 0，問的是時序脈絡與區塊長度，不是一天中的哪個時段。</p>""",
    """    要分辨故障的 0 與合法的 0，可用的訊號是時序脈絡與區塊長度。</p>""")

sub("""     落在該組中段，不是挑出來的極端值。""",
    """     落在該組中段，屬於代表性的中位案例。""")

sub("""<b>S1–S3 說明這不是罕見情境</b>""",
    """<b>S1–S3 給出情境的普遍度</b>""")


def main() -> int:
    src = P.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            print(f"EDIT {i} matched {n} times (need 1):\n  " + old.strip()[:100])
            return 1
        src = src.replace(old, new, 1)
    P.write_text(src, encoding="utf-8")
    print(f"applied {len(EDITS)} prose edits (pass 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
