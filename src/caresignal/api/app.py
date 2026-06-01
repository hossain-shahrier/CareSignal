from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from caresignal import __version__
from caresignal.api.deps import load_model_bundle
from caresignal.pipeline import clip_patient_features
from caresignal.schemas import FEATURE_COLUMNS, HealthResponse, PatientFeatures, PredictionResponse

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PACKAGE_STATIC = Path(__file__).resolve().parent.parent / "static"
ROOT_STATIC = PROJECT_ROOT / "static"


def _resolve_static_dir() -> Path:
    for candidate in (PACKAGE_STATIC, ROOT_STATIC):
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return candidate
    return PACKAGE_STATIC


STATIC_DIR = _resolve_static_dir()


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    bundle_dir = artifacts_dir or ARTIFACTS_DIR
    bundle = load_model_bundle(bundle_dir)
    threshold = float(bundle.manifest.get("metrics", {}).get("threshold", 0.5))

    app = FastAPI(
        title="CareSignal",
        description="30-day hospital readmission risk API",
        version=__version__,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=bundle.manifest.get("version", __version__),
        )

    @app.get("/info", include_in_schema=False)
    def model_info() -> dict:
        metrics = bundle.manifest.get("metrics", {})
        return {
            "version": bundle.manifest.get("version", __version__),
            "trained_at": bundle.manifest.get("trained_at"),
            "threshold": threshold,
            "features": FEATURE_COLUMNS,
            "metrics": {
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "positive_rate": metrics.get("positive_rate"),
                "test_patients": metrics.get("test_patients"),
                "calibrated": metrics.get("calibrated"),
            },
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: PatientFeatures) -> PredictionResponse:
        start = time.perf_counter()
        features = clip_patient_features(features)
        frame = pd.DataFrame([features.model_dump()])
        risk_score = float(bundle.pipeline.predict_proba(frame[FEATURE_COLUMNS])[0][1])
        predicted = risk_score >= threshold
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict completed latency_ms=%.2f model_version=%s risk_score=%.4f",
            latency_ms,
            bundle.manifest.get("version"),
            risk_score,
        )
        return PredictionResponse(risk_score=risk_score, predicted_readmission=predicted)

    def _demo_page() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise FileNotFoundError(f"Demo UI missing at {index}. Pull latest code and restart the server.")
        return FileResponse(index)

    @app.get("/", include_in_schema=False)
    @app.get("/demo", include_in_schema=False)
    def demo_ui() -> FileResponse:
        return _demo_page()

    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    else:
        logger.warning("Static directory not found at %s — demo UI disabled", STATIC_DIR)

    app.state.model_bundle = bundle
    return app
