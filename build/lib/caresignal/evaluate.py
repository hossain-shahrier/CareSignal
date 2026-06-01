from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probas >= threshold).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, probas)) if len(np.unique(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, probas))
        if len(np.unique(y_true)) > 1
        else float(y_true.mean()),
        "brier_score": float(brier_score_loss(y_true, probas)),
        "accuracy": float((preds == y_true).mean()),
    }
    return metrics


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
