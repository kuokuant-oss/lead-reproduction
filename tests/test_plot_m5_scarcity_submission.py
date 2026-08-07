from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lead import PROC, ROOT


def _load_plot_module():
    path = ROOT / "scripts" / "plot_m5_scarcity_submission.py"
    spec = importlib.util.spec_from_file_location("plot_m5_scarcity_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_parser_and_macro() -> None:
    module = _load_plot_module()
    records = module.load_matched_meter_metrics(
        ROOT / "docs" / "reports" / "m5-matched-context-breakdown.md"
    )
    assert len(records) == 40
    macro = module.macro_records(records)
    value = next(
        row["pr_auc_macro"]
        for row in macro
        if row["budget"] == 50_000 and row["model"] == "tabpfn"
    )
    assert value == pytest.approx(0.86555)


def test_building_loader_keeps_only_complete_model_pairs() -> None:
    module = _load_plot_module()
    manifest_path = (
        PROC
        / "m5_building_curve"
        / "protocol"
        / "representative"
        / "seed42"
        / "building_ladder.json"
    )
    manifest = module.load_building_manifest(manifest_path)
    records, budgets = module.load_building_meter_metrics(
        PROC / "m5_building_curve" / "aggregate" / "metrics.csv",
        manifest,
    )
    assert budgets == [10, 20, 50, 100]
    assert len(records) == len(budgets) * 2 * 4
    for budget in budgets:
        for model in module.MODEL_ORDER:
            meters = {
                row["meter"]
                for row in records
                if row["budget"] == budget and row["model"] == model
            }
            assert meters == set(module.METER_ORDER)


def test_latex_tables_use_booktabs_without_vertical_rules(tmp_path: Path) -> None:
    module = _load_plot_module()
    matched = module.load_matched_meter_metrics(
        ROOT / "docs" / "reports" / "m5-matched-context-breakdown.md"
    )
    manifest_path = (
        PROC
        / "m5_building_curve"
        / "protocol"
        / "representative"
        / "seed42"
        / "building_ladder.json"
    )
    manifest = module.load_building_manifest(manifest_path)
    building, _ = module.load_building_meter_metrics(
        PROC / "m5_building_curve" / "aggregate" / "metrics.csv",
        manifest,
    )
    matched_macro = module.macro_records(matched)
    building_macro = module.macro_records(building)
    outputs = (
        (
            "Experiment A",
            tmp_path / "exp_a.tex",
            matched,
            matched_macro,
            "matched_context",
        ),
        (
            "Experiment B",
            tmp_path / "exp_b.tex",
            building,
            building_macro,
            "building_count",
        ),
    )
    for label, out, records, macro, experiment in outputs:
        module.render_meter_detail_table(records, macro, experiment, out)
        latex = out.read_text(encoding="utf-8")
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex
        tabular_spec = latex.split("\\begin{tabular}", 1)[1].split("\n", 1)[0]
        assert "|" not in tabular_spec
        assert latex.startswith("\\documentclass[10pt]{article}")
        assert "\\begin{preview}" in latex
        assert "\\usepackage{newtxtext,newtxmath}" in latex
        assert latex.rstrip().endswith("\\end{document}")
        assert label in latex
        other = "Experiment B" if label == "Experiment A" else "Experiment A"
        assert other not in latex
        assert "\\footnotesize" not in latex
        expected_budgets = (
            ("N=5k", "N=10k", "N=20k", "N=50k", "N=100k")
            if label == "Experiment A"
            else ("K=10", "K=20", "K=50", "K=100")
        )
        for budget in expected_budgets:
            assert budget in latex


def test_pr_auc_format_uses_three_decimal_truncation() -> None:
    module = _load_plot_module()
    assert module._format_truncated(0.9569) == "0.956"
    assert module._format_truncated(0.956) == "0.956"
    assert module._format_truncated(-0.0119, signed=True) == "-0.011"
