"""Shared data validation without importing either native training runtime."""

import json
from pathlib import Path

import pandas as pd

from derm_context.download import sha256_file
from derm_context.prepare import assert_split_integrity


def load_prepared(processed_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    if sha256_file(processed_dir / "metadata.parquet") != manifest["processed_sha256"]:
        raise ValueError("Prepared metadata changed since preparation")
    if sha256_file(processed_dir / "splits.csv") != manifest["split_sha256"]:
        raise ValueError("Split assignments changed since preparation")
    frame = pd.read_parquet(processed_dir / "metadata.parquet")
    assert_split_integrity(frame)
    return frame, manifest
