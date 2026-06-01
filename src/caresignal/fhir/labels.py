from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from caresignal.fhir.parser import (
    encounter_period,
    get_patient,
    is_inpatient_encounter,
    load_bundle,
    reference_to_patient_id,
)

READMISSION_WINDOW_DAYS = 30


@dataclass(frozen=True)
class InpatientStay:
    encounter_id: str
    admission_start: datetime
    discharge_end: datetime


def inpatient_stays(bundle: dict[str, Any], patient_id: str) -> list[InpatientStay]:
    stays: list[InpatientStay] = []
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource") or {}
        if resource.get("resourceType") != "Encounter":
            continue
        subject = reference_to_patient_id((resource.get("subject") or {}).get("reference"))
        if subject != patient_id:
            continue
        if not is_inpatient_encounter(resource):
            continue
        start, end = encounter_period(resource)
        if start is None or end is None or end < start:
            continue
        stays.append(
            InpatientStay(
                encounter_id=resource.get("id") or "",
                admission_start=start,
                discharge_end=end,
            )
        )
    stays.sort(key=lambda stay: stay.admission_start)
    return stays


def readmitted_within_30_days(
    stays: list[InpatientStay], index_discharge: datetime
) -> bool:
    window_end = index_discharge + timedelta(days=READMISSION_WINDOW_DAYS)
    for stay in stays:
        if stay.admission_start <= index_discharge:
            continue
        if stay.admission_start <= window_end:
            return True
        break
    return False


def label_index_discharges(bundle: dict[str, Any]) -> list[tuple[InpatientStay, int]]:
    patient = get_patient(bundle)
    if not patient:
        return []
    patient_id = patient.get("id")
    if not patient_id:
        return []

    stays = inpatient_stays(bundle, patient_id)
    labeled: list[tuple[InpatientStay, int]] = []
    for index, stay in enumerate(stays):
        future_stays = stays[index + 1 :]
        label = int(readmitted_within_30_days(future_stays, stay.discharge_end))
        labeled.append((stay, label))
    return labeled


def load_labeled_stays(path: Path) -> list[tuple[InpatientStay, int]]:
    bundle = load_bundle(path)
    return label_index_discharges(bundle)
