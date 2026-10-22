import numpy as np
import pandas as pd
import pytest

from derm_context.download import sha256_file
from derm_context.embeddings import select_cohort, validate_embeddings, write_json
from derm_context.train_embeddings import load_cache


def make_cache(tmp_path):
    frame = pd.DataFrame(
        {
            "isic_id": ["a", "b", "c", "d", "e", "f"],
            "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    source = {"processed_sha256": "data-hash", "split_sha256": "split-hash"}
    chosen = frame.iloc[[1, 0, 3, 2]]
    chosen[["isic_id", "split"]].to_csv(tmp_path / "rows.csv", index=False)
    np.save(tmp_path / "vectors.npy", np.eye(4, dtype=np.float32))
    manifest = {
        "complete": True,
        "rows_done": 4,
        "identity": {**source, "rows": 4, "config": {"dimension": 4}},
        "rows_sha256": sha256_file(tmp_path / "rows.csv"),
        "vectors_sha256": sha256_file(tmp_path / "vectors.npy"),
    }
    write_json(tmp_path / "manifest.json", manifest)
    return frame, source, manifest


def test_cache_alignment_uses_image_ids_not_original_row_position(tmp_path):
    frame, source, _ = make_cache(tmp_path)
    chosen, values, _ = load_cache(tmp_path, frame, source)
    assert chosen.isic_id.tolist() == ["b", "a", "d", "c"]
    assert chosen.target.tolist() == [1, 0, 1, 0]
    assert values.shape == (4, 4)


@pytest.mark.parametrize("problem", ["incomplete", "changed_split", "changed_vectors"])
def test_cache_corruption_or_wrong_split_fails_closed(tmp_path, problem):
    frame, source, manifest = make_cache(tmp_path)
    if problem == "incomplete":
        manifest["complete"] = False
    elif problem == "changed_split":
        manifest["identity"]["split_sha256"] = "different"
    else:
        np.save(tmp_path / "vectors.npy", np.zeros((4, 4), dtype=np.float32))
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_cache(tmp_path, frame, source)


def test_test_images_cannot_enter_training_cache(tmp_path):
    frame, source, manifest = make_cache(tmp_path)
    frame.loc[frame.isic_id == "b", "split"] = "test"
    rows = pd.read_csv(tmp_path / "rows.csv")
    rows.loc[rows.isic_id == "b", "split"] = "test"
    rows.to_csv(tmp_path / "rows.csv", index=False)
    manifest["rows_sha256"] = sha256_file(tmp_path / "rows.csv")
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="test leakage"):
        load_cache(tmp_path, frame, source)


def test_optional_training_subsample_retains_positives_and_entire_validation(tmp_path):
    frame, _, _ = make_cache(tmp_path)
    chosen = select_cohort(frame, max_train_negatives=1)
    assert not chosen.split.eq("test").any()
    assert set(chosen.loc[chosen.split == "validation", "isic_id"]) == {"c", "d"}
    assert "b" in chosen.isic_id.tolist()


def test_nonfinite_and_nonunit_vectors_rejected():
    for values in [np.zeros((2, 4)), np.full((2, 4), np.nan)]:
        with pytest.raises(ValueError):
            validate_embeddings(values, dimension=4)
