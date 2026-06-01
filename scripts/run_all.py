from __future__ import annotations

import argparse
import logging
from pathlib import Path

from caresignal.evaluate import evaluate_model
from caresignal.pipeline import build_feature_matrix, load_config, save_feature_matrix
from caresignal.train import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CareSignal end-to-end pipeline.")
    parser.add_argument("--fhir-dir", type=Path, default=Path("data/reference"))
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_dir = Path(config["paths"]["processed_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    features_path = processed_dir / "features.parquet"

    logger.info("Building features from %s", args.fhir_dir)
    features = build_feature_matrix(args.fhir_dir)
    save_feature_matrix(features, features_path)
    logger.info("Wrote %s rows to %s", len(features), features_path)

    logger.info("Training model")
    manifest = train_model(features_path, args.config, artifacts_dir)
    logger.info("Training metrics: %s", manifest["metrics"])

    threshold = float(manifest["metrics"].get("threshold", config.get("threshold", 0.5)))
    holdout_path = artifacts_dir / "holdout.parquet"
    eval_features = holdout_path if holdout_path.exists() else features_path
    report = evaluate_model(
        eval_features,
        artifacts_dir / "model.joblib",
        artifacts_dir / "evaluation.json",
        threshold,
    )
    logger.info("Evaluation report: %s", report)


if __name__ == "__main__":
    main()
