"""Generate the full CareSignal cohort (5000 patients) and train the model."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from generate_cohort import FULL_COHORT_SIZE, clear_output_dir, generate_cohort

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full data/raw cohort and train CareSignal.")
    parser.add_argument("--count", type=int, default=FULL_COHORT_SIZE)
    parser.add_argument("--fhir-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removed = clear_output_dir(args.fhir_dir)
    if removed:
        logger.info("Cleared %s old bundles from %s", removed, args.fhir_dir)

    logger.info("Step 1/2: generating %s FHIR bundles", args.count)
    written = generate_cohort(args.fhir_dir, args.count, args.seed)
    logger.info("Wrote %s bundles to %s", written, args.fhir_dir)

    if args.skip_train:
        return

    logger.info("Step 2/2: feature pipeline + training")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_all.py"),
        "--fhir-dir",
        str(args.fhir_dir),
        "--config",
        str(args.config),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
