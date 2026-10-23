"""Evaluate frozen XGBoost heads and validation thresholds on held-out patients."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from derm_context.download import sha256_file
from derm_context.embeddings import validate_embeddings, write_json
from derm_context.evaluate_full import paired_bootstrap
from derm_context.features import select_features
from derm_context.metrics import classification_metrics
from derm_context.test_embeddings import CACHE
from derm_context.test_protocol import PROTOCOL, verify_protocol

REPORT = Path("reports/heldout_test_results.json")


def load_test_cache(cache_dir, selected, protocol, protocol_path):
    state = json.loads((cache_dir / "manifest.json").read_text())
    if not state["complete"] or state["rows_done"] != len(selected):
        raise ValueError("Test extraction is incomplete")
    expected = {
        "protocol_sha256": sha256_file(protocol_path),
        "rows": len(selected),
        "test_cohort_sha256": protocol["test_cohort_sha256"],
        "config": protocol["embedding_config"],
    }
    if state["identity"] != expected:
        raise ValueError("Test embeddings belong to another protocol")
    for filename, key in [("rows.csv", "rows_sha256"), ("vectors.npy", "vectors_sha256")]:
        if sha256_file(cache_dir / filename) != state[key]:
            raise ValueError(f"Test cache checksum mismatch: {filename}")
    rows = pd.read_csv(cache_dir / "rows.csv")
    if rows.isic_id.duplicated().any() or not rows.equals(selected[["isic_id", "split"]]):
        raise ValueError("Test cache identities/order/splits differ")
    if set(rows.split) != {"test"}:
        raise ValueError("Only held-out test rows are permitted")
    for field in ["device", "dtype", "versions", "max_soft_tokens"]:
        if state["hardware"][field] != protocol["embedding_hardware"][field]:
            raise ValueError(f"Test hardware differs: {field}")
    if state["batch_size"] != protocol["batch_size"]:
        raise ValueError("Test batch size differs")
    vectors = np.load(cache_dir / "vectors.npy", mmap_mode="r")
    dimension = protocol["embedding_config"]["dimension"]
    if vectors.shape != (len(selected), dimension):
        raise ValueError("Test embedding shape mismatch")
    for start in range(0, len(vectors), 4096):
        validate_embeddings(vectors[start : start + 4096], dimension)
    return vectors, state


def predict_frozen(entry, selected, vectors, artifact_dir=Path("artifacts")):
    directory = artifact_dir / entry["run_name"]
    run = json.loads((directory / "run.json").read_text())
    if entry["threshold"] != run["validation"]["operating_point"]["threshold"]:
        raise ValueError("Frozen threshold differs from validation")
    if entry["mode"] != run["mode"]:
        raise ValueError("Frozen mode differs")
    model = XGBClassifier()
    model.load_model(directory / "model.json")
    if model.best_iteration != entry["best_iteration"]:
        raise ValueError("Frozen model iteration differs")
    if entry["mode"] in {"metadata", "fusion"}:
        preprocessor = joblib.load(directory / "preprocessor.joblib")
        metadata = preprocessor.transform(
            select_features(selected, run["config"]["feature_set"])
        ).astype(np.float32)
    if entry["mode"] == "image":
        features = vectors
    elif entry["mode"] == "fusion":
        features = np.concatenate([vectors, metadata], axis=1)
    elif entry["mode"] == "metadata":
        features = metadata
    else:
        raise ValueError("Unknown frozen mode")
    if features.shape[1] != entry["feature_count"] or model.n_features_in_ != features.shape[1]:
        raise ValueError("Frozen feature count differs")
    # Explicit range preserves the validation-selected early stopping round.
    return model.predict_proba(features, iteration_range=(0, entry["best_iteration"] + 1))[:, 1]


def evaluate_test(
    protocol_path=PROTOCOL,
    cache_dir=CACHE,
    report_path=REPORT,
    destination=Path("artifacts/heldout-test-v1"),
):
    if report_path.exists() or destination.exists():
        raise FileExistsError("Held-out evaluation already exists; do not overwrite it")
    protocol, selected, _ = verify_protocol(protocol_path)
    vectors, extraction = load_test_cache(cache_dir, selected, protocol, protocol_path)
    started = time.monotonic()
    scores, models = [], []
    predictions = selected[["isic_id", "patient_id", "target"]].copy()
    for entry in protocol["models"]:
        prediction = predict_frozen(entry, selected, vectors)
        scores.append(prediction)
        predictions[entry["mode"]] = prediction
        models.append(
            {
                **entry,
                "test": classification_metrics(
                    selected.target.to_numpy(), prediction, entry["threshold"]
                ),
                "validation": json.loads(Path(f"reports/{entry['run_name']}.json").read_text())[
                    "validation"
                ],
            }
        )
        print(f"Predicted {len(selected):,} held-out lesions: {entry['run_name']}", flush=True)
    bootstrap = paired_bootstrap(
        selected.target.to_numpy(),
        selected.patient_id.to_numpy(),
        scores,
        iterations=protocol["bootstrap"]["iterations"],
        seed=protocol["bootstrap"]["seed"],
    )
    bootstrap["conditions"] = (
        "Held-out patients, fixed previously fitted models and validation thresholds; "
        "conditional on this training fit; same-source internal test, not external validation"
    )
    for entry, difference in zip(protocol["models"][1:], bootstrap["differences"], strict=True):
        difference["run_name"] = entry["run_name"]
    verify_protocol(protocol_path)  # No model, threshold, preprocessing or data mutation.
    destination.mkdir(parents=True)
    predictions.to_parquet(destination / "predictions.parquet", index=False)
    training_report = json.loads(
        Path(f"reports/{protocol['models'][0]['run_name']}.json").read_text()
    )
    result = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
        "test_cohort_sha256": protocol["test_cohort_sha256"],
        "test_rows": len(selected),
        "test_patients": protocol["test_patients"],
        "test_positives": int(selected.target.sum()),
        "test_positive_patients": protocol["test_positive_patients"],
        "train_rows": training_report["train_rows"],
        "validation_rows": training_report["validation"]["n"],
        "test_evaluated": True,
        "external_validation": False,
        "refit": False,
        "threshold_selected_on": "validation",
        "calibrated": False,
        "frozen_files_verified_before_and_after": True,
        "primary_comparison": protocol["primary_comparison"],
        "models": models,
        "bootstrap": bootstrap,
        "embedding_vectors_sha256": extraction["vectors_sha256"],
        "predictions_sha256": sha256_file(destination / "predictions.parquet"),
        "evaluation_seconds": time.monotonic() - started,
    }
    write_json(report_path, result)
    write_json(destination / "result.json", result)
    print("Frozen held-out evaluation complete", flush=True)
    return result


if __name__ == "__main__":
    evaluate_test()
