from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from caresignal.fhir.labels import inpatient_stays, label_index_discharges
from caresignal.fhir.parser import (
    conditions_for_patient,
    discover_fhir_files,
    get_patient,
    load_bundle,
    normalize_sex,
    patient_age_at,
    procedures_for_encounter,
    resources_by_type,
)
from caresignal.schemas import FEATURE_COLUMNS, PatientFeatures


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_features_from_bundle(bundle_path: Path) -> list[dict]:
    bundle = load_bundle(bundle_path)
    patient = get_patient(bundle)
    if not patient:
        return []

    patient_id = patient.get("id")
    if not patient_id:
        return []

    grouped = resources_by_type(bundle)
    conditions = grouped.get("Condition", [])
    procedures = grouped.get("Procedure", [])
    stays = inpatient_stays(bundle, patient_id)
    labeled = label_index_discharges(bundle)

    rows: list[dict] = []
    for index, (stay, label) in enumerate(labeled):
        prior = sum(1 for earlier in stays[:index] if earlier.discharge_end < stay.discharge_end)
        age = patient_age_at(patient.get("birthDate"), stay.discharge_end)
        if age is None:
            continue

        features = PatientFeatures(
            age=age,
            sex=normalize_sex(patient.get("gender")),
            length_of_stay_days=(stay.discharge_end - stay.admission_start).total_seconds()
            / 86400.0,
            prior_admissions=prior,
            condition_count=conditions_for_patient(conditions, patient_id, stay.discharge_end),
            procedure_count=procedures_for_encounter(
                procedures,
                patient_id,
                stay.encounter_id,
                stay.admission_start,
                stay.discharge_end,
            ),
        )
        row = features.model_dump()
        row["patient_id"] = patient_id
        row["encounter_id"] = stay.encounter_id
        row["readmitted_30d"] = label
        row["source_file"] = bundle_path.name
        rows.append(row)
    return rows


def build_feature_matrix(fhir_dir: Path) -> pd.DataFrame:
    all_rows: list[dict] = []
    for bundle_path in discover_fhir_files(fhir_dir):
        all_rows.extend(build_features_from_bundle(bundle_path))
    if not all_rows:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "readmitted_30d"])
    return pd.DataFrame(all_rows)


def save_feature_matrix(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
