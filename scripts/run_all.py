#!/usr/bin/env python3
"""Run the complete revised pipeline from historical CSVs to final outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20251010)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    subprocess.run(
        [sys.executable, str(scripts / "prepare_analysis_data.py"), "--root", str(root)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "run_revised_analysis.py"),
            "--root",
            str(root),
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
