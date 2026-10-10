#!/usr/bin/env python3
"""Build a compact, checksum-documented Zenodo reproducibility archive."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = DIST / "llm_big5_ad_sem_r3_reproducibility"
ARCHIVE = DIST / "llm_big5_ad_sem_r3_reproducibility.zip"

FILES = [
    "README.md",
    "requirements.txt",
    "0_LLM_questionare_generation.ipynb",
    "1_response_to_json.py",
    "2_json_to_csv.py",
    "3_merge_csv_by_model.py",
    "4_multi_group_sem.ipynb",
    "5_sem.ipynb",
    "docs/ANALYSIS_PROTOCOL.md",
    "data/DATA_DICTIONARY.md",
    "data/README.md",
    "data/raw/README.md",
    "data/processed/README.md",
    "data/processed/primary_chinese_promotion.csv",
    "data/processed/exploratory_configurations_promotion.csv",
    "scripts/analysis_common.py",
    "scripts/prepare_analysis_data.py",
    "scripts/run_analysis.py",
    "scripts/run_all.py",
    "tests/test_generated_outputs.py",
]

DIRECTORIES = [
    "results",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required_inputs() -> None:
    for run in range(1, 11):
        FILES.append(f"csv_output/deepseek-v2_236b_{run}.csv")
        DIRECTORIES.append(f"json_output/experiment_outputs_deepseek-v2_236b_{run}")
    for name in [
        "deepseek-v2_236b_promotion.csv",
        "gemma2_9b_promotion.csv",
        "jwnder_jais-adaptive_7b_promotion.csv",
        "sailor2_20b_promotion.csv",
    ]:
        FILES.append(f"csv_output_merged/{name}")


def main() -> None:
    copy_required_inputs()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for relative in FILES:
        source = ROOT / relative
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in DIRECTORIES:
        source = ROOT / relative
        target = STAGE / relative
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc"),
        )

    entries = []
    for path in sorted(p for p in STAGE.rglob("*") if p.is_file()):
        entries.append(f"{sha256(path)}  {path.relative_to(STAGE).as_posix()}")
    (STAGE / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in STAGE.rglob("*") if p.is_file()):
            arcname = Path(STAGE.name) / path.relative_to(STAGE)
            archive.write(path, arcname.as_posix())
    print(ARCHIVE)
    print(f"sha256={sha256(ARCHIVE)}")
    print(f"bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
