"""Restore model.joblib from base64 text (Docker / HF build)."""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "model.joblib"
B64 = ROOT / "artifacts" / "model.joblib.b64"


def main() -> None:
    if MODEL.is_file():
        print(f"Model already present: {MODEL}")
        return
    if not B64.is_file():
        raise SystemExit(f"Missing {B64}")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_bytes(base64.b64decode(B64.read_text(encoding="ascii")))
    print(f"Restored {MODEL} ({MODEL.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
