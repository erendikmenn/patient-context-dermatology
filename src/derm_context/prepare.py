"""Validate the official archive and freeze patient-disjoint splits."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from derm_context.download import sha256_file
from derm_context.features import feature_columns, select_features


def merge_labels(metadata: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.rename(columns={"malignant": "target"})
    for name, table in [("metadata", metadata), ("labels", labels)]:
        if "isic_id" not in table or table.isic_id.isna().any():
            raise ValueError(f"Missing image IDs in {name}")
        if table.isic_id.duplicated().any():
            raise ValueError(f"Duplicate image IDs in {name}")
    if set(metadata.isic_id) != set(labels.isic_id):
        raise ValueError("Metadata and label image IDs do not match")
    if "target" not in labels or not labels.target.isin([0, 1]).all():
        raise ValueError("Labels must be complete and binary")
    if "target" in metadata or "malignant" in metadata:
        raise ValueError("Input metadata unexpectedly contains an outcome")
    result = metadata.merge(labels[["isic_id", "target"]], on="isic_id", validate="one_to_one")
    if "patient_id" not in result or result.patient_id.isna().any():
        raise ValueError("Missing patient IDs")
    result["patient_id"] = result.patient_id.astype(str)
    if result.patient_id.str.strip().eq("").any():
        raise ValueError("Empty patient IDs")
    result["target"] = result.target.astype("int8")
    return result


def split_patients(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    patients = frame.groupby("patient_id", sort=True).target.max()
    train_ids, heldout_ids = train_test_split(
        patients.index,
        test_size=0.30,
        stratify=patients.values,
        random_state=seed,
    )
    valid_ids, test_ids = train_test_split(
        heldout_ids,
        test_size=0.5,
        stratify=patients.loc[heldout_ids].values,
        random_state=seed,
    )
    assignments = {str(i): "train" for i in train_ids}
    assignments.update({str(i): "validation" for i in valid_ids})
    assignments.update({str(i): "test" for i in test_ids})
    result = frame.patient_id.astype(str).map(assignments)
    check = frame.assign(split=result)
    assert_split_integrity(check)
    return result


def assert_split_integrity(frame: pd.DataFrame) -> None:
    if frame["split"].isna().any() or set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Invalid or incomplete split assignments")
    if frame.groupby("patient_id")["split"].nunique().max() != 1:
        raise ValueError("Patient overlap between splits")
    for name, part in frame.groupby("split"):
        if set(part.target.unique()) != {0, 1}:
            raise ValueError(f"Both classes required in {name}")


def prepare(raw_dir: Path, processed_dir: Path, reports_dir: Path, seed: int = 42) -> dict:
    if (processed_dir / "metadata.parquet").exists():
        raise FileExistsError(
            "Prepared dataset exists; use a new output directory to change splits"
        )
    manifest = json.loads((raw_dir / "download_manifest.json").read_text())
    for name in ["input.zip", "ground_truth.csv", "supplement.csv"]:
        if sha256_file(raw_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Download checksum mismatch: {name}")
    labels = pd.read_csv(raw_dir / "ground_truth.csv")
    with zipfile.ZipFile(raw_dir / "input.zip") as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one metadata CSV, got {csv_names}")
        print(f"Reading metadata: {csv_names[0]}", flush=True)
        with archive.open(csv_names[0]) as handle:
            metadata = pd.read_csv(handle, low_memory=False)
        frame = merge_labels(metadata, labels)
        image_names = [n for n in archive.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
        image_ids = [Path(n).stem for n in image_names]
        if len(set(image_ids)) != len(image_ids) or set(image_ids) != set(frame.isic_id):
            raise ValueError("ZIP image IDs are duplicated or do not match metadata")
        print(f"Checking CRC for {len(image_names):,} images and metadata", flush=True)
        bad_file = archive.testzip()
        if bad_file:
            raise ValueError(f"Corrupt ZIP member: {bad_file}")
        sample = np.random.default_rng(seed).choice(
            len(image_names),
            size=min(512, len(image_names)),
            replace=False,
        )
        sizes = []
        for index in sample:
            with Image.open(io.BytesIO(archive.read(image_names[int(index)]))) as img:
                img.load()
                sizes.append(img.size)
        members = dict(zip(image_ids, image_names, strict=True))
        frame["image_member"] = frame.isic_id.map(members)
    supplement = pd.read_csv(raw_dir / "supplement.csv", low_memory=False)
    if supplement.isic_id.duplicated().any() or set(supplement.isic_id) != set(frame.isic_id):
        raise ValueError("Supplement IDs do not match the main dataset")
    # Only provenance columns enter the prepared table; diagnosis fields stay in raw data.
    provenance = [c for c in ["attribution", "copyright_license"] if c in supplement]
    for column in provenance:
        if column in frame:
            frame = frame.drop(columns=column)
    frame = frame.merge(supplement[["isic_id"] + provenance], on="isic_id", validate="one_to_one")
    selected = select_features(frame, "morphometry")
    for column in selected:
        frame[column] = selected[column]
    frame["split"] = split_patients(frame, seed)
    numeric, categorical = feature_columns("morphometry")
    retained = ["isic_id", "patient_id", "target", "split", "image_member"]
    frame = frame[retained + numeric + categorical + provenance].sort_values("isic_id")
    summary = {
        "dataset": manifest["dataset"],
        "license": manifest["license"],
        "seed": seed,
        "rows": len(frame),
        "patients": int(frame.patient_id.nunique()),
        "positive_lesions": int(frame.target.sum()),
        "prevalence": float(frame.target.mean()),
        "positive_patients": int(frame.groupby("patient_id").target.max().sum()),
        "patient_overlap": 0,
        "image_count": len(image_names),
        "zip_crc_checked": True,
        "decoded_image_sample": len(sizes),
        "all_images_decoded": False,
        "sample_width_range": [min(x[0] for x in sizes), max(x[0] for x in sizes)],
        "sample_height_range": [min(x[1] for x in sizes), max(x[1] for x in sizes)],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "missing_fraction": {c: float(frame[c].isna().mean()) for c in numeric},
        "splits": {},
        "files": manifest["files"],
    }
    for name, part in frame.groupby("split"):
        summary["splits"][name] = {
            "rows": len(part),
            "patients": int(part.patient_id.nunique()),
            "positive_lesions": int(part.target.sum()),
            "prevalence": float(part.target.mean()),
            "positive_patients": int(part.groupby("patient_id").target.max().sum()),
        }
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / "metadata.parquet", index=False)
    frame[["isic_id", "patient_id", "split"]].to_csv(processed_dir / "splits.csv", index=False)
    summary["processed_sha256"] = sha256_file(processed_dir / "metadata.parquet")
    summary["split_sha256"] = sha256_file(processed_dir / "splits.csv")
    (processed_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (reports_dir / "data_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ["rows", "patients", "positive_lesions", "splits"]}, indent=2
        ),
        flush=True,
    )
    return summary
