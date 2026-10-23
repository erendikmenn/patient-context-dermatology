"""Freeze existing fitted models and validation thresholds before held-out evaluation."""

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from derm_context.dataset import load_prepared
from derm_context.download import sha256_file
from derm_context.embeddings import write_json

RUNS = ["metadata-full-control-v1", "image-eg2-70-full-v1", "fusion-eg2-70-full-v1"]
PROTOCOL = Path("reports/heldout_test_protocol.json")


def cohort_hash(frame):
    return hashlib.sha256(
        frame[["isic_id", "patient_id", "target"]].to_csv(index=False).encode()
    ).hexdigest()


def verify_files(files, root=Path(".")):
    for filename, expected in files.items():
        if sha256_file(root / filename) != expected:
            raise ValueError(f"Frozen file changed: {filename}")


def verify_protocol(path=PROTOCOL):
    protocol = json.loads(path.read_text())
    verify_files(protocol["files"])
    for name, version in protocol["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Frozen runtime changed: {name}")
    frame, source = load_prepared(Path("data/processed"))
    for field in ["processed_sha256", "split_sha256"]:
        if source[field] != protocol[field]:
            raise ValueError(f"Frozen data changed: {field}")
    selected = frame[frame.split == "test"].sort_values("isic_id").reset_index(drop=True)
    if cohort_hash(selected) != protocol["test_cohort_sha256"]:
        raise ValueError("Held-out cohort changed")
    return protocol, selected, source


def freeze(path=PROTOCOL):
    if path.exists():
        raise FileExistsError("Protocol already frozen; verify it instead of overwriting")
    frame, source = load_prepared(Path("data/processed"))
    selected = frame[frame.split == "test"].sort_values("isic_id").reset_index(drop=True)
    extraction = json.loads(Path("reports/embeddinggemma2-70-full-extraction.json").read_text())
    reports = [json.loads(Path(f"reports/{name}.json").read_text()) for name in RUNS]
    files = {}
    models = []
    for name, report in zip(RUNS, reports, strict=True):
        if report["test_evaluated"] or report["threshold_selected_on"] != "validation":
            raise ValueError("Models must be selected without test evaluation")
        for field in [
            "config",
            "xgboost_parameters",
            "versions",
            "embedding_config",
            "training_cohort_sha256",
            "validation_cohort_sha256",
        ]:
            if report[field] != reports[0][field]:
                raise ValueError(f"Models differ in {field}")
        for field in ["processed_sha256", "split_sha256"]:
            if report[field] != source[field]:
                raise ValueError(f"Model and dataset differ in {field}")
        for split, field in [
            ("train", "training_cohort_sha256"),
            ("validation", "validation_cohort_sha256"),
        ]:
            subset = frame[frame.split == split].sort_values("isic_id")
            if cohort_hash(subset) != report[field]:
                raise ValueError("Model cohort differs from prepared data")
        if not report["all_original_training_rows"] or not report["encoder_frozen"]:
            raise ValueError("Expected frozen encoder and full-data heads")
        model_files = [
            f"artifacts/{name}/model.json",
            f"artifacts/{name}/run.json",
            f"reports/{name}.json",
        ]
        if report["mode"] != "image":
            model_files.append(f"artifacts/{name}/preprocessor.joblib")
        if json.loads(Path(f"artifacts/{name}/run.json").read_text()) != report:
            raise ValueError("Saved model run and public report differ")
        files.update({f: sha256_file(Path(f)) for f in model_files})
        models.append(
            {
                "run_name": name,
                "mode": report["mode"],
                "threshold": report["validation"]["operating_point"]["threshold"],
                "best_iteration": report["best_iteration"],
                "feature_count": report["feature_count"],
            }
        )
    for filename in [
        "configs/embeddinggemma2-70.json",
        "configs/baseline.json",
        "reports/embeddinggemma2-70-full-extraction.json",
    ]:
        files[filename] = sha256_file(Path(filename))
    if extraction["identity"]["config"] != reports[0]["embedding_config"]:
        raise ValueError("Encoder configuration differs from training")
    result = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "purpose": "One held-out patient test of existing models; no refitting or threshold tuning",
        "primary_comparison": "fusion minus metadata average_precision",
        "secondary_metrics": [
            "roc_auc",
            "pauc_tpr_ge_0_8_unstandardized",
            "brier_score",
            "log_loss",
            "fixed_validation_threshold_counts",
        ],
        "bootstrap": {
            "iterations": 1000,
            "seed": 42,
            "unit": "patient",
            "interval": "paired percentile 95%; fitted models held fixed",
        },
        "models": models,
        "files": files,
        "versions": reports[0]["versions"],
        "embedding_config": reports[0]["embedding_config"],
        "embedding_hardware": extraction["hardware"],
        "batch_size": extraction["batch_size"],
        "processed_sha256": source["processed_sha256"],
        "split_sha256": source["split_sha256"],
        "test_cohort_sha256": cohort_hash(selected),
        "test_rows": len(selected),
        "test_patients": int(selected.patient_id.nunique()),
        "test_positives": int(selected.target.sum()),
        "test_positive_patients": int(selected.groupby("patient_id").target.max().sum()),
        "calibrated": False,
        "refit": False,
        "threshold_selected_on": "validation",
        "external_validation": False,
    }
    write_json(path, result)
    verify_protocol(path)
    print(f"Frozen {len(models)} models and thresholds for {len(selected):,} held-out lesions")
    return result


if __name__ == "__main__":
    freeze()
