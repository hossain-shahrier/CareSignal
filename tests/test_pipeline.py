from __future__ import annotations

from pathlib import Path

from caresignal.pipeline import build_feature_matrix

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_build_feature_matrix_row_count():
    df = build_feature_matrix(REFERENCE_DIR)
    assert len(df) >= 6
    assert "readmitted_30d" in df.columns
    assert set(df["readmitted_30d"].unique()).issubset({0, 1})


def test_multi_stay_prior_admissions():
    df = build_feature_matrix(REFERENCE_DIR)
    multi = df[df["patient_id"] == "patient-multi"].sort_values("prior_admissions")
    assert len(multi) == 3
    assert list(multi["prior_admissions"]) == [0, 1, 2]
    index_stay = multi[multi["encounter_id"] == "enc-index"].iloc[0]
    assert index_stay["readmitted_30d"] == 1
    assert index_stay["days_since_last_discharge"] > 300


def test_days_since_last_discharge_first_stay_zero():
    df = build_feature_matrix(REFERENCE_DIR)
    first_stays = df[df["prior_admissions"] == 0]
    assert (first_stays["days_since_last_discharge"] == 0).all()
