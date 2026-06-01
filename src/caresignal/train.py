from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

LOG1P_FEATURES = ("length_of_stay_days", "days_since_last_discharge")

from caresignal import __version__
from caresignal.evaluate import compute_metrics, find_best_threshold
from caresignal.schemas import FEATURE_COLUMNS


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_pipeline(model_cfg: dict) -> Pipeline:
    numeric_features = [
        "age",
        "length_of_stay_days",
        "prior_admissions",
        "condition_count",
        "procedure_count",
        "days_since_last_discharge",
    ]
    log1p_numeric = [f for f in numeric_features if f in LOG1P_FEATURES]
    plain_numeric = [f for f in numeric_features if f not in LOG1P_FEATURES]
    categorical_features = ["sex"]

    log1p_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), plain_numeric),
            ("num_log", log1p_pipe, log1p_numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    classifier = LGBMClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        max_depth=model_cfg.get("max_depth", 4),
        num_leaves=model_cfg.get("num_leaves", 31),
        min_child_samples=model_cfg.get("min_child_samples", 20),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=42,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)]).set_output(
        transform="pandas"
    )


def _split_indices(
    groups: pd.Series,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(indices, y, groups=groups))

    if val_size <= 0 or len(train_val_idx) < 2:
        return train_val_idx, np.array([], dtype=int), test_idx

    relative_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_idx_local, val_idx_local = next(
        gss_val.split(train_val_idx, y.iloc[train_val_idx], groups=groups.iloc[train_val_idx])
    )
    return train_val_idx[train_idx_local], train_val_idx[val_idx_local], test_idx


def _select_model_config(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    base_cfg: dict,
    tuning_cfg: dict,
) -> dict:
    if not tuning_cfg.get("enabled", False):
        return base_cfg

    candidates = tuning_cfg.get("candidates") or [
        {**base_cfg},
        {**base_cfg, "n_estimators": 300, "max_depth": 6, "num_leaves": 31},
        {**base_cfg, "n_estimators": 400, "max_depth": 8, "learning_rate": 0.03, "num_leaves": 48},
    ]

    best_cfg = base_cfg
    best_auc = -1.0
    for candidate in candidates:
        pipeline = build_model_pipeline(candidate)
        pipeline.fit(x_train, y_train)
        if len(x_val) == 0 or y_val.nunique() < 2:
            return candidate
        probas = pipeline.predict_proba(x_val)[:, 1]
        from caresignal.evaluate import compute_metrics

        auc = compute_metrics(y_val.to_numpy(), probas, 0.5)["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_cfg = candidate
    return best_cfg


def train_model(features_path: Path, config_path: Path, artifacts_dir: Path) -> dict:
    config = load_config(config_path)
    df = pd.read_parquet(features_path)
    if df.empty:
        raise ValueError("Feature matrix is empty; add FHIR bundles before training.")

    seed = config.get("seed", 42)
    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.15)
    tuning_cfg = config.get("tuning", {})

    x = df[FEATURE_COLUMNS]
    y = df["readmitted_30d"]
    groups = df["patient_id"]

    train_idx, val_idx, test_idx = _split_indices(
        groups, y, test_size=test_size, val_size=val_size, seed=seed
    )
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
    x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

    model_cfg = _select_model_config(
        x_train, y_train, x_val, y_val, config.get("model", {}), tuning_cfg
    )
    pipeline = build_model_pipeline(model_cfg)
    pipeline.fit(x_train, y_train)

    min_cal_rows = int(tuning_cfg.get("min_calibration_rows", 50))
    cal_method = str(tuning_cfg.get("calibration_method", "sigmoid"))
    if cal_method not in {"sigmoid", "isotonic"}:
        cal_method = "sigmoid"
    if tuning_cfg.get("calibrate", True) and len(x_val) >= min_cal_rows:
        model = CalibratedClassifierCV(FrozenEstimator(pipeline), method=cal_method)
        model.fit(x_val, y_val)
    else:
        model = pipeline

    default_threshold = float(config.get("threshold", 0.5))
    if len(x_val) > 0 and y_val.nunique() > 1:
        val_probas = model.predict_proba(x_val)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val.to_numpy(), val_probas)
    else:
        threshold = default_threshold
        val_f1 = 0.0

    test_probas = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([])
    metrics = {
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "train_patients": int(groups.iloc[train_idx].nunique()),
        "val_patients": int(groups.iloc[val_idx].nunique()) if len(val_idx) else 0,
        "test_patients": int(groups.iloc[test_idx].nunique()),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
        "val_f1_at_threshold": val_f1,
        "model_params": model_cfg,
        "calibrated": bool(tuning_cfg.get("calibrate", True) and len(x_val) >= min_cal_rows),
    }
    if len(x_test):
        metrics.update(compute_metrics(y_test.to_numpy(), test_probas, threshold))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = artifacts_dir / "holdout.parquet"
    df.iloc[test_idx].to_parquet(holdout_path, index=False)

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(model, model_path)

    manifest = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
