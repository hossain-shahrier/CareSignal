"""Copy external FHIR patient bundles (e.g. Synthea output) into data/raw for training."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def import_bundles(source_dir: Path, dest_dir: Path, *, replace: bool) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        for path in dest_dir.glob("*.json"):
            path.unlink()

    copied = 0
    for path in sorted(source_dir.rglob("*.json")):
        if not path.is_file():
            continue
        target = dest_dir / path.name
        if target.exists() and not replace:
            stem = path.stem
            suffix = 1
            while target.exists():
                target = dest_dir / f"{stem}-{suffix}{path.suffix}"
                suffix += 1
        shutil.copy2(path, target)
        copied += 1
        if copied % 500 == 0:
            logger.info("Copied %s files...", copied)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import FHIR JSON bundles into data/raw.")
    parser.add_argument("source_dir", type=Path, help="Folder with patient *.json bundles")
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument("--replace", action="store_true", help="Clear data/raw before import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = import_bundles(args.source_dir, args.dest, replace=args.replace)
    logger.info("Imported %s bundles into %s", n, args.dest)
    logger.info("Next: python scripts/run_all.py --fhir-dir %s", args.dest)


if __name__ == "__main__":
    main()
