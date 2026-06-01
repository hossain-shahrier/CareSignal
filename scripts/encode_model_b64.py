"""Encode model.joblib as base64 text for Hugging Face git (no binary LFS)."""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "model.joblib"
OUT = ROOT / "artifacts" / "model.joblib.b64"


def main() -> None:
    if not MODEL.is_file():
        raise SystemExit(f"Missing {MODEL} — train first: python scripts/run_all.py --fhir-dir data/reference")
    OUT.write_text(base64.b64encode(MODEL.read_bytes()).decode("ascii"), encoding="ascii")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
