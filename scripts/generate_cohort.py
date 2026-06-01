"""Generate synthetic FHIR R4 bundles compatible with the CareSignal feature pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TZ = "-05:00"


def _iso(dt: datetime) -> str:
    return dt.strftime(f"%Y-%m-%dT%H:%M:%S{TZ}")


def _patient_bundle(
    rng: random.Random,
    *,
    patient_id: str,
    sex: str,
    birth_year: int,
    stays: list[tuple[datetime, datetime, list[datetime], int]],
) -> dict:
    """Build one transaction bundle. stays: (admit, discharge, procedure_times, condition_count_at_discharge)."""
    entries: list[dict] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": patient_id,
                "gender": sex,
                "birthDate": f"{birth_year:04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            }
        }
    ]

    condition_onsets: list[datetime] = []
    for admit, discharge, proc_times, cond_target in stays:
        enc_id = f"enc-{uuid4().hex[:8]}"
        entries.append(
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": enc_id,
                    "status": "finished",
                    "class": {"code": "IMP", "display": "inpatient encounter"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "period": {"start": _iso(admit), "end": _iso(discharge)},
                }
            }
        )
        while len(condition_onsets) < cond_target:
            onset = admit - timedelta(days=rng.randint(30, 900))
            condition_onsets.append(onset)
            entries.append(
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"cond-{uuid4().hex[:8]}",
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "onsetDateTime": _iso(onset),
                    }
                }
            )
        for performed in proc_times:
            entries.append(
                {
                    "resource": {
                        "resourceType": "Procedure",
                        "id": f"proc-{uuid4().hex[:8]}",
                        "status": "completed",
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "encounter": {"reference": f"Encounter/{enc_id}"},
                        "performedDateTime": _iso(performed),
                    }
                }
            )

    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def _simulate_patient(rng: random.Random, index: int) -> dict:
    patient_id = f"synth-{index:05d}"
    sex = rng.choice(["male", "female"])
    birth_year = rng.randint(1940, 2000)
    anchor = datetime(2018, 1, 1, 8, 0, 0)

    n_stays = rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 28, 18, 9])[0]
    stays: list[tuple[datetime, datetime, list[datetime], int]] = []
    cursor = anchor + timedelta(days=rng.randint(0, 400))

    chronic_conditions = rng.randint(0, 6)

    for stay_idx in range(n_stays):
        los_days = rng.uniform(2.0, 12.0)
        admit = cursor
        discharge = admit + timedelta(days=los_days, hours=rng.randint(0, 8))
        n_proc = rng.randint(0, 3)
        proc_times = [
            admit + timedelta(days=rng.uniform(0.5, max(los_days - 0.5, 1.0)))
            for _ in range(n_proc)
        ]
        cond_count = min(chronic_conditions + rng.randint(0, 2), 12)
        stays.append((admit, discharge, proc_times, cond_count))

        if stay_idx == n_stays - 1:
            break

        age_at_discharge = discharge.year - birth_year
        readmit_logit = (
            -2.2
            + 0.035 * max(age_at_discharge - 50, 0)
            + 0.25 * min(stay_idx, 3)
            + 0.12 * cond_count
            + 0.08 * n_proc
            + (-0.06 * los_days)
        )
        p_readmit = 1.0 / (1.0 + pow(2.718281828, -readmit_logit))

        if rng.random() < p_readmit:
            gap_days = rng.randint(3, 28)
        else:
            gap_days = rng.randint(35, 220)
        cursor = discharge + timedelta(days=gap_days)

    return _patient_bundle(rng, patient_id=patient_id, sex=sex, birth_year=birth_year, stays=stays)


FULL_COHORT_SIZE = 5000


def clear_output_dir(output_dir: Path) -> int:
    removed = 0
    if not output_dir.is_dir():
        return removed
    for path in output_dir.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


def generate_cohort(output_dir: Path, count: int, seed: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    for index in range(count):
        bundle = _simulate_patient(rng, index)
        path = output_dir / f"{bundle['entry'][0]['resource']['id']}.json"
        path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        written += 1
        if written % 500 == 0:
            logger.info("Wrote %s / %s bundles", written, count)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic FHIR cohort for CareSignal.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--count",
        type=int,
        default=FULL_COHORT_SIZE,
        help=f"Number of patient bundles (default {FULL_COHORT_SIZE}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing *.json in output-dir before generating.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Shorthand for --count {FULL_COHORT_SIZE} --replace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full:
        args.count = FULL_COHORT_SIZE
        args.replace = True
    if args.replace:
        removed = clear_output_dir(args.output_dir)
        if removed:
            logger.info("Removed %s existing bundles from %s", removed, args.output_dir)
    logger.info("Generating %s patients into %s", args.count, args.output_dir)
    n = generate_cohort(args.output_dir, args.count, args.seed)
    logger.info("Done. Wrote %s FHIR bundles.", n)


if __name__ == "__main__":
    main()
