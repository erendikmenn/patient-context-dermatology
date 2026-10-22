import json

import numpy as np
import pandas as pd

from derm_context.download import sha256_file
from derm_context.train import predict, train


def test_training_and_inference_without_labels(tmp_path):
    """Synthetic data checks wiring only, never used as a real performance result."""
    rows = []
    for split, count in [("train", 80), ("validation", 40), ("test", 40)]:
        for i in range(count):
            rows.append(
                {
                    "isic_id": f"{split}-{i}",
                    "patient_id": f"{split}-p{i // 2}",
                    "split": split,
                    "target": int(i % 4 == 0),
                    "age_approx": 30 + i % 50,
                    "clin_size_long_diam_mm": 3 + i % 5,
                    "sex": "male" if i % 2 else "female",
                    "anatom_site_general": "torso",
                }
            )
    frame = pd.DataFrame(rows)
    processed = tmp_path / "processed"
    processed.mkdir()
    frame.to_parquet(processed / "metadata.parquet", index=False)
    frame[["isic_id", "patient_id", "split"]].to_csv(processed / "splits.csv", index=False)
    manifest = {
        "processed_sha256": sha256_file(processed / "metadata.parquet"),
        "split_sha256": sha256_file(processed / "splits.csv"),
    }
    (processed / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "seed": 42,
        "feature_set": "clinical",
        "n_estimators": 20,
        "max_depth": 2,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 3,
        "n_jobs": 1,
        "class_weight": "none",
        "target_sensitivity": 0.95,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = train(processed, tmp_path / "artifacts", tmp_path / "reports", config_path, "smoke")
    assert result["train_rows"] == 80
    assert result["validation"]["n"] == 40
    assert not result["test_evaluated"]
    assert result["reload_predictions_verified"]
    inference = frame[frame.split == "validation"].drop(columns=["target", "split", "patient_id"])
    inference_path = tmp_path / "unlabelled.csv"
    inference.to_csv(inference_path, index=False)
    predictions_path = tmp_path / "predictions.csv"
    predict(inference_path, tmp_path / "artifacts" / "smoke", predictions_path)
    actual = pd.read_csv(predictions_path)
    saved = pd.read_parquet(tmp_path / "artifacts" / "smoke" / "validation_predictions.parquet")
    np.testing.assert_allclose(actual.score, saved.score, atol=1e-7)
    assert actual.isic_id.tolist() == inference.isic_id.tolist()
