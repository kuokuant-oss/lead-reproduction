"""Render the meter-level PR-AUC detail table from the published figure data.

Run on Windows:
    uv --directory C:\\Users\\User\\projects\\lead-analysis run python scripts\\make_meter_table.py

Input  : m5_exp_b_building_count_meter_pr_auc_v5_fixed50k_provisional_2026-09-01_data.json
Output : m5_exp_b_building_count_meter_pr_auc_detail.tex
Then   : pdflatex -interaction=nonstopmode -output-directory=<outdir> <tex>
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
DATA = OUT / "m5_exp_b_building_count_meter_pr_auc_v5_fixed50k_provisional_2026-09-01_data.json"
TEX = OUT / "m5_exp_b_building_count_meter_pr_auc_detail.tex"

BUDGETS = (50, 100, 200, 400, 725)
MODELS = (("ensemble", "Tree"), ("tabpfn", "TabPFN"))
METERS = (
    ("chilled_water", "Chilled water"),
    ("steam", "Steam"),
    ("hot_water", "Hot water"),
)

pts = {
    (p["budget"], p["meter"], p["model"]): p
    for p in json.loads(DATA.read_text(encoding="utf-8"))["plot_points"]
}


def cell(budget: int, meter: str, model: str) -> str:
    p = pts.get((budget, meter, model))
    if p is None:
        return "--"
    se = p["standard_error"]
    if se is None:
        return f"{p['mean_pr_auc']:.3f}"
    return rf"{p['mean_pr_auc']:.3f}\,$\pm$\,{se:.3f}"


group = "Meter"
sub = ""
cmid = ""
for i, b in enumerate(BUDGETS):
    first = 2 + 2 * i
    group += rf" & \multicolumn{{2}}{{c}}{{K={b}}}"
    sub += r" & {Tree} & {TabPFN}"
    cmid += rf"\cmidrule(lr){{{first}-{first + 1}}}"

rows = []
for meter, label in METERS:
    cells = [cell(b, meter, m) for b in BUDGETS for m, _ in MODELS]
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")

body = "\n".join(rows)
ncol = 2 * len(BUDGETS)

TEX.write_text(
    rf"""\documentclass[10pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{newtxtext,newtxmath}}
\usepackage[paperwidth=16in,paperheight=8.5in,margin=0.55in]{{geometry}}
\usepackage{{booktabs,tabularx,siunitx}}
\usepackage[font=small,labelfont=bf]{{caption}}
\usepackage[active,tightpage]{{preview}}
\setlength\PreviewBorder{{10pt}}
\sisetup{{detect-all}}
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\pagestyle{{empty}}
\newsavebox{{\tblbox}}
\begin{{document}}
\sbox{{\tblbox}}{{%
\scriptsize
\setlength{{\tabcolsep}}{{2.0pt}}
\begin{{tabular}}{{l*{{{ncol}}}{{c}}}}
\toprule
{group} \\
{cmid}
{sub} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\begin{{preview}}\begin{{minipage}}{{\wd\tblbox}}
\centering
\captionof{{table}}{{Experiment B meter-level PR-AUC; mean $\pm$ standard error across building seeds; K=725 across row seeds.}}
\label{{tab:m5-exp-b-meter-detail}}
\usebox{{\tblbox}}
\end{{minipage}}\end{{preview}}
\end{{document}}
""",
    encoding="utf-8",
    newline="\n",
)
print("wrote", TEX)
