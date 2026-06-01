from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from caresignal.pipeline import clip_patient_features
from caresignal.schemas import FEATURE_COLUMNS, PatientFeatures

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

HIGH = PatientFeatures(
    age=78,
    sex="male",
    length_of_stay_days=1,
    prior_admissions=3,
    condition_count=25,
    procedure_count=3,
    days_since_last_discharge=14,
)
LOW = PatientFeatures(
    age=32,
    sex="female",
    length_of_stay_days=1,
    prior_admissions=0,
    condition_count=2,
    procedure_count=0,
    days_since_last_discharge=0,
)


def _score(model, features: PatientFeatures) -> float:
    row = clip_patient_features(features)
    frame = pd.DataFrame([row.model_dump()])
    return float(model.predict_proba(frame[FEATURE_COLUMNS])[0, 1])


@pytest.fixture(scope="module")
def trained_model():
    path = ARTIFACTS / "model.joblib"
    if not path.is_file():
        pytest.skip("artifacts/model.joblib missing — run training first")
    return joblib.load(path)


@pytest.fixture(scope="module")
def threshold() -> float:
    manifest = ARTIFACTS / "manifest.json"
    if not manifest.is_file():
        return 0.29
    return float(json.loads(manifest.read_text(encoding="utf-8"))["metrics"]["threshold"])


def test_preset_low_below_high(trained_model, threshold: float) -> None:
    low = _score(trained_model, LOW)
    high = _score(trained_model, HIGH)
    assert low < high, f"expected low ({low:.3f}) < high ({high:.3f})"
    assert high >= threshold, f"high-risk preset should be elevated at threshold {threshold}"


def test_recent_utilization_above_distant_gap(trained_model) -> None:
    base = dict(
        age=56,
        sex="male",
        length_of_stay_days=1,
        prior_admissions=2,
        condition_count=15,
        procedure_count=5,
    )
    recent = PatientFeatures(**base, days_since_last_discharge=10)
    distant = PatientFeatures(**base, days_since_last_discharge=400)
    assert _score(trained_model, recent) > _score(trained_model, distant)


def test_calibrated_scores_are_not_all_plateaus(trained_model) -> None:
    cases = [LOW, HIGH]
    for los in (0.5, 1, 3, 7):
        cases.append(
            PatientFeatures(
                age=60,
                sex="male",
                length_of_stay_days=los,
                prior_admissions=1,
                condition_count=8,
                procedure_count=2,
                days_since_last_discharge=30,
            )
        )
    scores = {_score(trained_model, c) for c in cases}
    assert len(scores) >= 4
    assert max(scores) - min(scores) > 0.05, f"scores too flat: {sorted(scores)}"
