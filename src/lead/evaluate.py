"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    pred_label = (pred >= 0.5).astype("int8")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred_label, average="binary", pos_label=1, zero_division=0
    )
    return {
        "val_auc": float(roc_auc_score(y_true, pred)),
        "precision_05": float(precision),
        "recall_05": float(recall),
        "f1_05": float(f1),
    }
