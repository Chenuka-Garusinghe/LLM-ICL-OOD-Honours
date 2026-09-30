#!/usr/bin/env python3
"""Write the per-task synthetic datasets (generator_spec.pdf, dataset construction).

Each suite in configs/default.yaml (`suites`) becomes
data/synthetic/v3/<suite>/: one parquet per task under tasks/ plus
task_manifest.json. Tasks and data are fully determined by the config, so
rerunning reproduces the same files.

Usage: python scripts/make_synth_data.py [--suites eval,pilot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.suites import make_suite, suite_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", default="eval,pilot", help="comma-separated suite names")
    args = parser.parse_args()
    config = load_config()
    for name in args.suites.split(","):
        manifest = make_suite(name, config)
        families = [t["family"] for t in manifest["tasks"]]
        print(f"{name}: {len(families)} tasks ({families.count('linear')} linear, "
              f"{families.count('tree')} tree) -> {suite_dir(name, config)}")


if __name__ == "__main__":
    main()
