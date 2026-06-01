from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Shared contract for training features and API /predict input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=0, le=120)
    sex: Literal["male", "female", "unknown"]
    length_of_stay_days: float = Field(ge=0)
    prior_admissions: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    procedure_count: int = Field(ge=0)
    days_since_last_discharge: float = Field(
        ge=0,
        description="Days from previous inpatient discharge to this admission; 0 if first stay.",
    )


FEATURE_COLUMNS: list[str] = [
    "age",
    "sex",
    "length_of_stay_days",
    "prior_admissions",
    "condition_count",
    "procedure_count",
    "days_since_last_discharge",
]


class PredictionResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    predicted_readmission: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
