"""Split the 100k-context raw-reading scatter into one figure per meter."""

from __future__ import annotations

import argparse

import numpy as np

from plot_m5_score_covariate_scatter import (
    FEATURES,
    FIGURE_ROOT,
    METER_NAMES,
    MODELS,
    RAW_READING_DISPLAY_CAP,
    load_full_feature_columns,
    load_prediction,
    load_raw_meter_fields,
    plot_covariate,
)


CONTEXTS = (5_000, 10_000, 20_000, 50_000, 100_000)
CONTEXT_LABELS = {
    5_000: "5k",
    10_000: "10k",
    20_000: "20k",
    50_000: "50k",
    100_000: "100k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meters",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 3),
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        choices=CONTEXTS,
        default=list(CONTEXTS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = load_full_feature_columns()
    reference_index, reference_y, _, _ = load_prediction("tabpfn", "17", 5_000)
    if not np.array_equal(reference_index, columns["raw_index"]):
        raise AssertionError("feature export and prediction row order differ")
    if not np.array_equal(reference_y, columns["anomaly"]):
        raise AssertionError("feature export and prediction labels differ")

    meter, raw_reading = load_raw_meter_fields(reference_index)
    for context in args.contexts:
        all_scores: dict[tuple[str, int], np.ndarray] = {}
        for model in MODELS:
            for feature_text in FEATURES:
                _, _, _, score = load_prediction(model, feature_text, context)
                all_scores[(model, int(feature_text))] = score

        for meter_id in args.meters:
            selected = meter == meter_id
            label = METER_NAMES[meter_id]
            meter_scores = {key: value[selected] for key, value in all_scores.items()}
            meter_reading = raw_reading[selected]
            anomaly = reference_y[selected] == 1
            subtitle = (
                f"All {int(selected.sum()):,} {label.lower()} rows. "
                f"Dark = normal; light = {int(anomaly.sum()):,} true anomalies."
            )
            xlabel = "Raw meter_reading (symlog scale)"
            xticks = [0, 1, 10, 100, 1_000, 10_000, 100_000]
            xticklabels = ["0", "1", "10", "100", "1k", "10k", "100k"]
            if meter_id == 2:
                overflow = meter_reading > RAW_READING_DISPLAY_CAP
                if int(overflow.sum()) != 3_195 or not anomaly[overflow].all():
                    raise AssertionError("unexpected steam overflow composition")
                meter_reading = np.minimum(meter_reading, RAW_READING_DISPLAY_CAP)
                subtitle = (
                    f"All {int(selected.sum()):,} steam rows; "
                    f"{int(anomaly.sum()):,} anomalies. Dark = normal; light = anomaly. "
                    "Right edge: 3,195 building 1099 anomalies above 300k "
                    "(3,067 above 1M)."
                )
                xlabel = (
                    "Raw meter_reading (symlog scale; values above 300k at right edge)"
                )
                xticks.append(300_000)
                xticklabels.append("≥300k")

            plot_covariate(
                meter_scores,
                context=context,
                x=meter_reading,
                anomaly_mask=anomaly,
                meter=meter[selected],
                title=f"{label}: model decision by raw meter reading",
                subtitle=subtitle,
                xlabel=xlabel,
                output=FIGURE_ROOT
                / (
                    f"m5_context_{CONTEXT_LABELS[context]}_"
                    f"score_covariate_raw_reading_{label.lower()}.png"
                ),
                xscale="symlog",
                xticks=xticks,
                xticklabels=xticklabels,
                legend_meter_ids=(meter_id,),
            )
            print(f"plotted {label} at {context:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
