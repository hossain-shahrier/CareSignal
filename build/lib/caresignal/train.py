from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from caresignal import __version__
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
    ]
    categorical_features = ["sex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
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
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=42,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)]).set_output(
        transform="pandas"
    )


def train_model(features_path: Path, config_path: Path, artifacts_dir: Path) -> dict:
    config = load_config(config_path)
    df = pd.read_parquet(features_path)
    if df.empty:
        raise ValueError("Feature matrix is empty; add FHIR bundles before training.")

    seed = config.get("seed", 42)
    x = df[FEATURE_COLUMNS]
    y = df["readmitted_30d"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["split"]["test_size"],
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_model_pipeline(config.get("model", {}))
    pipeline.fit(x_train, y_train)

    probas = pipeline.predict_proba(x_test)[:, 1] if len(x_test) else []
    threshold = config.get("threshold", 0.5)
    metrics = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
    }
    if len(x_test):
        from caresignal.evaluate import compute_metrics

        metrics.update(compute_metrics(y_test.to_numpy(), probas, threshold))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    manifest = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
