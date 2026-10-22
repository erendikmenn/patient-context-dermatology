import json

import numpy as np
import pandas as pd
import pytest

from derm_context import expand_embeddings as module
from derm_context.download import sha256_file
from derm_context.embeddings import write_json


def setup_cache(tmp_path):
    config = {"dimension": 4, "seed": 42}
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    frame = pd.DataFrame(
        {
            "isic_id": [f"i{i:04}" for i in range(1200)],
            "split": ["train"] * 900 + ["validation"] * 200 + ["test"] * 100,
            "target": [i % 2 for i in range(1200)],
            "image_member": list(range(1200)),
        }
    )
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"fake test archive")
    source = {
        "processed_sha256": "data",
        "split_sha256": "split",
        "files": {"input.zip": {"sha256": sha256_file(archive)}},
    }
    hardware = {"device": "cpu", "dtype": "float32", "versions": {}, "max_soft_tokens": 70}
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    frame.iloc[[1000, 3]][["isic_id", "split"]].to_csv(seed_dir / "rows.csv", index=False)
    np.save(seed_dir / "vectors.npy", np.eye(4, dtype=np.float32)[[0, 3]])
    state = {
        "identity": {
            "config": config,
            "processed_sha256": "data",
            "split_sha256": "split",
            "rows": 2,
        },
        "rows_done": 2,
        "complete": True,
        "batch_size": 16,
        "hardware": hardware,
        "extraction_seconds": 1.0,
        "rows_sha256": sha256_file(seed_dir / "rows.csv"),
        "vectors_sha256": sha256_file(seed_dir / "vectors.npy"),
    }
    write_json(seed_dir / "manifest.json", state)
    return config_path, frame, source, archive, seed_dir, hardware


def test_expansion_resumes_only_missing_rows_and_preserves_id_order(tmp_path, monkeypatch):
    config, frame, source, archive, seed_dir, hardware = setup_cache(tmp_path)
    monkeypatch.setattr(module, "load_prepared", lambda _: (frame, source))
    monkeypatch.setattr(module, "load_encoder", lambda *args: (None, hardware))
    monkeypatch.setattr(module, "synchronize", lambda _: None)

    class Archive:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, member):
            return member

    monkeypatch.setattr(module, "ImageArchive", Archive)
    encoded = []
    calls = 0

    def encode(_model, images, _config, _batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        encoded.extend(images)
        return np.eye(4, dtype=np.float32)[np.array(images) % 4]

    monkeypatch.setattr(module, "encode", encode)
    out = tmp_path / "full"
    args = (config, tmp_path, archive, tmp_path, seed_dir, out)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.expand(*args)
    partial = json.loads((out / "manifest.json").read_text())
    assert partial["new_rows_done"] == 512
    assert partial["rows_done"] == 514
    result = module.expand(*args)
    assert result["complete"] and result["rows_done"] == 1100
    assert result["new_rows_done"] == 1098
    assert len(encoded) == len(set(encoded)) == 1098
    assert 3 not in encoded and 1000 not in encoded and max(encoded) < 1100
    rows = pd.read_csv(out / "rows.csv")
    assert rows.isic_id.tolist() == frame.iloc[:1100].isic_id.tolist()
    np.testing.assert_array_equal(
        np.load(out / "vectors.npy"), np.eye(4, dtype=np.float32)[np.arange(1100) % 4]
    )
    assert module.expand(*args) == result  # completed runs verify, never re-encode


@pytest.mark.parametrize("problem", ["config", "checksum", "test_image"])
def test_reuse_rejects_incompatible_or_corrupt_cache(tmp_path, problem):
    _, frame, source, _, seed_dir, _ = setup_cache(tmp_path)
    selected = frame[frame.split != "test"]
    identity = {"config": {"dimension": 4, "seed": 42}, **source}
    if problem == "config":
        identity["config"]["seed"] = 7
    elif problem == "checksum":
        (seed_dir / "vectors.npy").write_bytes(b"corrupt")
    else:
        rows = pd.read_csv(seed_dir / "rows.csv")
        rows.loc[0, "isic_id"] = "i1199"
        rows.to_csv(seed_dir / "rows.csv", index=False)
        state = json.loads((seed_dir / "manifest.json").read_text())
        state["rows_sha256"] = sha256_file(seed_dir / "rows.csv")
        write_json(seed_dir / "manifest.json", state)
    with pytest.raises(ValueError):
        module.reusable_cache(seed_dir, selected, identity)
