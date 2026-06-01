from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probas >= threshold).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, probas)) if len(np.unique(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, probas))
        if len(np.unique(y_true)) > 1
        else float(y_true.mean()),
        "brier_score": float(brier_score_loss(y_true, probas)),
        "accuracy": float((preds == y_true).mean()),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    return metrics


def find_best_threshold(y_true: np.ndarray, probas: np.ndarray) -> tuple[float, float]:
    """Pick threshold that maximizes F1 on validation data."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        preds = (probas >= threshold).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_model(
    features_path: Path,
    model_path: Path,
    output_path: Path,
    threshold: float,
) -> dict:
    import joblib
    import pandas as pd

    from caresignal.schemas import FEATURE_COLUMNS

    df = pd.read_parquet(features_path)
    pipeline = joblib.load(model_path)
    x = df[FEATURE_COLUMNS]
    y = df["readmitted_30d"].to_numpy()
    probas = pipeline.predict_proba(x)[:, 1]
    report = compute_metrics(y, probas, threshold)
    report["rows"] = int(len(df))
    report["threshold"] = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
