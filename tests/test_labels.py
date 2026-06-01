from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from caresignal.fhir.labels import InpatientStay, label_index_discharges, readmitted_within_30_days
from caresignal.fhir.parser import load_bundle

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_readmitted_within_30_days_positive():
    discharge = datetime(2024, 1, 5, 16, 0)
    stays = [
        InpatientStay("enc-2", datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 18, 12, 0))
    ]
    assert readmitted_within_30_days(stays, discharge) is True


def test_readmitted_within_30_days_negative():
    discharge = datetime(2024, 3, 3, 16, 0)
    stays = [
        InpatientStay("enc-2", datetime(2024, 4, 5, 8, 0), datetime(2024, 4, 7, 12, 0))
    ]
    assert readmitted_within_30_days(stays, discharge) is False


def test_readmitted_within_30_days_boundary():
    discharge = datetime(2024, 1, 1, 16, 0)
    window_end = discharge + timedelta(days=30)
    stays = [InpatientStay("enc-2", window_end, window_end + timedelta(days=1))]
    assert readmitted_within_30_days(stays, discharge) is True


def test_label_readmit_yes_bundle():
    bundle = load_bundle(REFERENCE_DIR / "patient_readmit_yes.json")
    labeled = label_index_discharges(bundle)
    assert len(labeled) == 2
    assert labeled[0][1] == 1
    assert labeled[1][1] == 0


def test_label_readmit_no_bundle():
    bundle = load_bundle(REFERENCE_DIR / "patient_readmit_no.json")
    labeled = label_index_discharges(bundle)
    assert len(labeled) == 1
    assert labeled[0][1] == 0
