from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    pipeline: object
    manifest: dict


def load_model_bundle(artifacts_dir: Path) -> ModelBundle:
    model_path = artifacts_dir / "model.joblib"
    manifest_path = artifacts_dir / "manifest.json"
    if not model_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Model bundle missing in {artifacts_dir}. Run scripts/run_all.py first."
        )
    pipeline = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("Loaded model version %s", manifest.get("version"))
    return ModelBundle(pipeline=pipeline, manifest=manifest)
