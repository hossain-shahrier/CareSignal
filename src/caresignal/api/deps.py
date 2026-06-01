from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def resolve_artifacts_dir(explicit: Path | None = None) -> Path:
    """Find artifacts/ whether running from repo root or pip-installed in Docker."""
    if explicit is not None:
        return explicit

    env_dir = os.environ.get("ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir)

    candidates: list[Path] = [
        Path.cwd() / "artifacts",
        Path("/app/artifacts"),
    ]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "artifacts")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "model.joblib").is_file() and (resolved / "manifest.json").is_file():
            logger.info("Using artifacts directory %s", resolved)
            return resolved

    return Path.cwd() / "artifacts"


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
