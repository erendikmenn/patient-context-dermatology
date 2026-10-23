"""Encode only the held-out cohort under a previously frozen test protocol."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from derm_context.download import sha256_file
from derm_context.embeddings import (
    encode,
    load_encoder,
    synchronize,
    validate_embeddings,
    write_json,
)
from derm_context.images import ImageArchive
from derm_context.test_protocol import PROTOCOL, verify_protocol

CACHE = Path("data/embeddings/embeddinggemma2-70-test")


def extract_test(output_dir=CACHE, protocol_path=PROTOCOL):
    protocol, selected, source = verify_protocol(protocol_path)
    archive_path = Path("data/raw/input.zip")
    if sha256_file(archive_path) != source["files"]["input.zip"]["sha256"]:
        raise ValueError("Image archive changed")
    config = protocol["embedding_config"]
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "rows": len(selected),
        "test_cohort_sha256": protocol["test_cohort_sha256"],
        "config": config,
    }
    checkpoint = output_dir / "manifest.json"
    expected_rows = selected[["isic_id", "split"]]
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity:
            raise ValueError("Test cache belongs to a different protocol")
        if sha256_file(output_dir / "rows.csv") != state["rows_sha256"]:
            raise ValueError("Test row checksum changed")
        if not pd.read_csv(output_dir / "rows.csv").equals(expected_rows):
            raise ValueError("Test cache row order changed")
        if state["complete"]:
            if sha256_file(output_dir / "vectors.npy") != state["vectors_sha256"]:
                raise ValueError("Test vector checksum changed")
            return state
        vectors = np.load(output_dir / "vectors.npy", mmap_mode="r+")
        if vectors.shape != (len(selected), config["dimension"]):
            raise ValueError("Test vector shape changed")
        if not 0 <= state["rows_done"] <= len(selected):
            raise ValueError("Invalid test checkpoint cursor")
        for start in range(0, state["rows_done"], 4096):
            validate_embeddings(
                vectors[start : min(start + 4096, state["rows_done"])], config["dimension"]
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Unrecognized test cache contents")
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_rows.to_csv(output_dir / "rows.csv", index=False)
        vectors = np.lib.format.open_memmap(
            output_dir / "vectors.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(selected), config["dimension"]),
        )
        state = {
            "identity": identity,
            "rows_done": 0,
            "complete": False,
            "test_encoded": True,
            "extraction_seconds": 0.0,
            "rows_sha256": sha256_file(output_dir / "rows.csv"),
        }
        write_json(checkpoint, state)
    model, hardware = load_encoder(
        config, Path("data/model-cache"), protocol["embedding_hardware"]["device"]
    )
    for field in ["device", "dtype", "versions", "max_soft_tokens"]:
        if hardware[field] != protocol["embedding_hardware"][field]:
            raise ValueError(f"Test encoder differs from frozen runtime: {field}")
    state["hardware"] = hardware
    state["batch_size"] = protocol["batch_size"]
    write_json(checkpoint, state)
    print(f"Test extraction: {state['rows_done']:,}/{len(selected):,}", flush=True)
    with ImageArchive(archive_path) as archive:
        for start in range(state["rows_done"], len(selected), 512):
            tick = time.monotonic()
            end = min(start + 512, len(selected))
            images = [archive.read(member) for member in selected.iloc[start:end].image_member]
            vectors[start:end] = encode(model, images, config, protocol["batch_size"])
            synchronize(hardware["device"])
            vectors.flush()
            state["rows_done"] = end
            state["extraction_seconds"] += time.monotonic() - tick
            write_json(checkpoint, state)
            if end % 2048 == 0 or end == len(selected):
                rate = end / state["extraction_seconds"]
                print(
                    f"Test {end:,}/{len(selected):,}; {rate:.1f} images/s; "
                    f"ETA {(len(selected) - end) / rate / 60:.1f} min",
                    flush=True,
                )
    verify_protocol(protocol_path)
    state["vectors_sha256"] = sha256_file(output_dir / "vectors.npy")
    state["complete"] = True
    write_json(checkpoint, state)
    write_json(Path("reports/heldout_test_extraction.json"), state)
    print("Held-out test embeddings complete", flush=True)
    return state


if __name__ == "__main__":
    extract_test()
