from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# HL7 v3 ActCode inpatient-style codes (Synthea + reference fixtures).
INPATIENT_CLASS_CODES = {"IMP", "ACUTE", "NONAC", "EMER", "SS"}


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def reference_to_patient_id(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref.replace("urn:uuid:", "", 1)
    if "Patient/" in ref:
        part = ref.split("Patient/", 1)[-1]
        return part.split("/")[0] if part else None
    return None


def load_bundle(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_fhir_files(fhir_dir: Path) -> list[Path]:
    if not fhir_dir.is_dir():
        return []
    patterns = ("*.json", "fhir/*.json", "fhir/**/*.json", "**/*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(fhir_dir.glob(pattern))
    return sorted({path.resolve() for path in files if path.is_file()})


def resources_by_type(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource")
        if not resource:
            continue
        resource_type = resource.get("resourceType")
        if not resource_type:
            continue
        grouped.setdefault(resource_type, []).append(resource)
    return grouped


def get_patient(bundle: dict[str, Any]) -> dict[str, Any] | None:
    for resource in resources_by_type(bundle).get("Patient", []):
        return resource
    return None


def is_inpatient_encounter(encounter: dict[str, Any]) -> bool:
    if encounter.get("status") != "finished":
        return False
    encounter_class = encounter.get("class") or {}
    code = encounter_class.get("code")
    if isinstance(code, str) and code.upper() in INPATIENT_CLASS_CODES:
        return True
    coding = encounter_class.get("coding") or []
    for item in coding:
        if str(item.get("code", "")).upper() in INPATIENT_CLASS_CODES:
            return True
    return False


def encounter_period(encounter: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    period = encounter.get("period") or {}
    return parse_iso_datetime(period.get("start")), parse_iso_datetime(period.get("end"))


def patient_age_at(birth_date: str | None, at: datetime) -> int | None:
    if not birth_date:
        return None
    try:
        born = datetime.fromisoformat(birth_date)
    except ValueError:
        return None
    years = at.year - born.year
    if (at.month, at.day) < (born.month, born.day):
        years -= 1
    return max(years, 0)


def normalize_sex(gender: str | None) -> str:
    if not gender:
        return "unknown"
    value = gender.strip().lower()
    if value in {"male", "female"}:
        return value
    return "unknown"


def conditions_for_patient(
    conditions: list[dict[str, Any]], patient_id: str, before: datetime
) -> int:
    count = 0
    for condition in conditions:
        subject = reference_to_patient_id((condition.get("subject") or {}).get("reference"))
        if subject != patient_id:
            continue
        onset = parse_iso_datetime(
            condition.get("onsetDateTime") or condition.get("recordedDate")
        )
        if onset is not None and onset <= before:
            count += 1
    return count


def procedures_for_encounter(
    procedures: list[dict[str, Any]],
    patient_id: str,
    encounter_id: str,
    start: datetime,
    end: datetime,
) -> int:
    count = 0
    for procedure in procedures:
        if procedure.get("status") not in {None, "completed", "in-progress"}:
            continue
        subject = reference_to_patient_id((procedure.get("subject") or {}).get("reference"))
        if subject != patient_id:
            continue
        encounter_ref = (procedure.get("encounter") or {}).get("reference", "")
        if encounter_id:
            if encounter_id not in encounter_ref:
                continue
        else:
            period = procedure.get("performedPeriod") or {}
            performed = parse_iso_datetime(
                procedure.get("performedDateTime") or period.get("start")
            )
            if performed is None or not (start <= performed <= end):
                continue
        count += 1
    return count
