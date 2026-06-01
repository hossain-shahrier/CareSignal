from __future__ import annotations

import json
from pathlib import Path

import pytest

from caresignal.evaluate import evaluate_model
from caresignal.pipeline import build_feature_matrix, load_config, save_feature_matrix
from caresignal.schemas import FEATURE_COLUMNS
from caresignal.train import train_model

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference"
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config" / "train.yaml"


def _artifacts_stale() -> bool:
    model_path = ARTIFACTS_DIR / "model.joblib"
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    if not model_path.exists() or not manifest_path.exists():
        return True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("feature_names") != FEATURE_COLUMNS


@pytest.fixture(scope="session")
def trained_artifacts() -> Path:
    if _artifacts_stale():
        config = load_config(CONFIG_PATH)
        processed_dir = Path(config["paths"]["processed_dir"])
        features_path = processed_dir / "features.parquet"
        features = build_feature_matrix(REFERENCE_DIR)
        save_feature_matrix(features, features_path)
        manifest = train_model(features_path, CONFIG_PATH, ARTIFACTS_DIR)
        threshold = float(manifest["metrics"].get("threshold", config.get("threshold", 0.5)))
        evaluate_model(
            ARTIFACTS_DIR / "holdout.parquet",
            ARTIFACTS_DIR / "model.joblib",
            ARTIFACTS_DIR / "evaluation.json",
            threshold,
        )
    return ARTIFACTS_DIR


@pytest.fixture
def app(trained_artifacts: Path):
    from caresignal.api.app import create_app

    return create_app(trained_artifacts)
