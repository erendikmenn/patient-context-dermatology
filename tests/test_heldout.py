import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from derm_context import evaluate_test as evaluation
from derm_context import test_embeddings as extraction
from derm_context.download import sha256_file
from derm_context.embeddings import write_json
from derm_context.test_protocol import verify_files


def test_frozen_artifact_mutation_is_rejected(tmp_path):
    model = tmp_path / "model.json"
    model.write_text("original model")
    files = {"model.json": sha256_file(model)}
    verify_files(files, tmp_path)
    model.write_text("retrained model")
    with pytest.raises(ValueError, match="Frozen file changed"):
        verify_files(files, tmp_path)


def test_saved_preprocessing_and_early_stopping_used_without_fit_or_rethreshold(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = (x.a + x.b > 0).astype(int)
    scaler = StandardScaler().fit(x.iloc[:60])
    features = scaler.transform(x).astype(np.float32)
    model = XGBClassifier(n_estimators=12, max_depth=2, early_stopping_rounds=3, n_jobs=1)
    model.fit(features[:60], y[:60], eval_set=[(features[60:80], y[60:80])], verbose=False)
    expected = model.predict_proba(features[80:])[:, 1]
    destination = tmp_path / "frozen"
    destination.mkdir()
    model.save_model(destination / "model.json")
    joblib.dump(scaler, destination / "preprocessor.joblib")
    threshold = 0.83
    write_json(
        destination / "run.json",
        {
            "mode": "metadata",
            "config": {"feature_set": "toy"},
            "validation": {"operating_point": {"threshold": threshold}},
        },
    )
    entry = {
        "run_name": "frozen",
        "mode": "metadata",
        "threshold": threshold,
        "best_iteration": model.best_iteration,
        "feature_count": 3,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Test evaluation must never fit")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(evaluation, "select_features", lambda frame, _: frame)
    actual = evaluation.predict_frozen(entry, x.iloc[80:], None, tmp_path)
    np.testing.assert_array_equal(actual, expected)
    assert entry["threshold"] == threshold
    with pytest.raises(ValueError, match="threshold"):
        evaluation.predict_frozen({**entry, "threshold": 0.5}, x.iloc[80:], None, tmp_path)
    with pytest.raises(ValueError, match="iteration"):
        evaluation.predict_frozen(
            {**entry, "best_iteration": 0 if model.best_iteration else 1},
            x.iloc[80:],
            None,
            tmp_path,
        )


def test_heldout_extraction_resume_and_cache_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = Path("data/raw")
    raw.mkdir(parents=True)
    archive = raw / "input.zip"
    archive.write_bytes(b"fake archive")
    frame = pd.DataFrame(
        {
            "isic_id": [f"test{i:04}" for i in range(600)],
            "split": ["test"] * 600,
            "image_member": list(range(600)),
        }
    )
    hardware = {"device": "cpu", "dtype": "float32", "versions": {}, "max_soft_tokens": 70}
    protocol = {
        "embedding_config": {"dimension": 4},
        "test_cohort_sha256": "test",
        "embedding_hardware": hardware,
        "batch_size": 16,
    }
    source = {"files": {"input.zip": {"sha256": sha256_file(archive)}}}
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    monkeypatch.setattr(extraction, "verify_protocol", lambda _: (protocol, frame, source))
    monkeypatch.setattr(extraction, "load_encoder", lambda *args: (None, hardware))
    monkeypatch.setattr(extraction, "synchronize", lambda _: None)

    class Archive:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, member):
            return member

    monkeypatch.setattr(extraction, "ImageArchive", Archive)
    encoded = []
    calls = 0

    def encode(_model, images, _config, _batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        encoded.extend(images)
        return np.eye(4, dtype=np.float32)[np.asarray(images) % 4]

    monkeypatch.setattr(extraction, "encode", encode)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        extraction.extract_test(cache, protocol_path)
    assert json.loads((cache / "manifest.json").read_text())["rows_done"] == 512
    extraction.extract_test(cache, protocol_path)
    assert encoded == list(range(600))
    vectors, state = evaluation.load_test_cache(cache, frame, protocol, protocol_path)
    np.testing.assert_array_equal(vectors, np.eye(4)[np.arange(600) % 4])
    assert state["complete"]
    with pytest.raises(ValueError, match="identities/order/splits"):
        evaluation.load_test_cache(
            cache, frame.iloc[::-1].reset_index(drop=True), protocol, protocol_path
        )
    rows = pd.read_csv(cache / "rows.csv")
    rows.loc[0, "split"] = "train"
    rows.to_csv(cache / "rows.csv", index=False)
    with pytest.raises(ValueError, match="checksum"):
        evaluation.load_test_cache(cache, frame, protocol, protocol_path)
