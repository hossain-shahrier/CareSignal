from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from caresignal import __version__
from caresignal.api.deps import load_model_bundle
from caresignal.schemas import FEATURE_COLUMNS, HealthResponse, PatientFeatures, PredictionResponse

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"


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

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: PatientFeatures) -> PredictionResponse:
        start = time.perf_counter()
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

    app.state.model_bundle = bundle
    return app
