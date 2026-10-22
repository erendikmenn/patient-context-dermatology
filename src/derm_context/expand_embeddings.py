"""Expand a verified pilot cache to the full train/validation cohort, without re-encoding it."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from derm_context.dataset import load_prepared
from derm_context.download import sha256_file
from derm_context.embeddings import (
    encode,
    load_encoder,
    select_cohort,
    synchronize,
    validate_embeddings,
    write_json,
)
from derm_context.images import ImageArchive


def reusable_cache(cache_dir: Path, selected: pd.DataFrame, identity: dict):
    state = json.loads((cache_dir / "manifest.json").read_text())
    if not state.get("complete") or state["rows_done"] != state["identity"]["rows"]:
        raise ValueError("Reuse requires a completed cache")
    for key in ["config", "processed_sha256", "split_sha256"]:
        if state["identity"][key] != identity[key]:
            raise ValueError(f"Reuse cache has different {key}")
    for filename, key in [("rows.csv", "rows_sha256"), ("vectors.npy", "vectors_sha256")]:
        if sha256_file(cache_dir / filename) != state[key]:
            raise ValueError(f"Reuse cache checksum mismatch: {filename}")
    rows = pd.read_csv(cache_dir / "rows.csv")
    if rows.isic_id.duplicated().any() or len(rows) != state["rows_done"]:
        raise ValueError("Invalid reuse cache IDs/count")
    positions = pd.Index(selected.isic_id).get_indexer(rows.isic_id)
    if (positions < 0).any():
        raise ValueError("Reuse cache includes images outside the train/validation cohort")
    if not np.array_equal(selected.iloc[positions].split, rows.split):
        raise ValueError("Reuse cache split labels differ")
    values = np.load(cache_dir / "vectors.npy", mmap_mode="r")
    if values.shape != (len(rows), identity["config"]["dimension"]):
        raise ValueError("Reuse cache shape mismatch")
    for start in range(0, len(values), 4096):
        validate_embeddings(values[start : start + 4096], values.shape[1])
    reused = np.zeros(len(selected), dtype=bool)
    reused[positions] = True
    return values, positions, np.flatnonzero(~reused), state


def expand(
    config_path: Path,
    processed_dir: Path,
    archive_path: Path,
    model_cache_dir: Path,
    reuse_dir: Path,
    output_dir: Path,
    batch_size: int = 16,
    device: str = "auto",
) -> dict:
    if batch_size < 1 or reuse_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a positive batch size and a distinct output directory")
    config = json.loads(config_path.read_text())
    frame, source = load_prepared(processed_dir)
    if sha256_file(archive_path) != source["files"]["input.zip"]["sha256"]:
        raise ValueError("Image archive changed since preparation")
    selected = select_cohort(frame)
    identity = {
        "config": config,
        "processed_sha256": source["processed_sha256"],
        "split_sha256": source["split_sha256"],
        "max_train_negatives": None,
        "rows": len(selected),
    }
    seed_vectors, reused_positions, pending, seed_state = reusable_cache(
        reuse_dir, selected, identity
    )
    provenance = {
        "vectors_sha256": seed_state["vectors_sha256"],
        "rows_sha256": seed_state["rows_sha256"],
        "rows": len(reused_positions),
        "original_extraction_seconds": seed_state["extraction_seconds"],
    }
    checkpoint = output_dir / "manifest.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity or state["reused_cache"] != provenance:
            raise ValueError("Existing expansion belongs to a different model, source, or cache")
        if sha256_file(output_dir / "rows.csv") != state["rows_sha256"]:
            raise ValueError("Expansion row file changed")
        if state.get("complete"):
            if sha256_file(output_dir / "vectors.npy") != state["vectors_sha256"]:
                raise ValueError("Completed vectors changed")
            print("Verified completed expanded cache", flush=True)
            return state
        vectors = np.load(output_dir / "vectors.npy", mmap_mode="r+")
        if vectors.shape != (len(selected), config["dimension"]):
            raise ValueError("Expansion shape mismatch")
        cursor = state["new_rows_done"]
        if not 0 <= cursor <= len(pending) or state["rows_done"] != len(reused_positions) + cursor:
            raise ValueError("Invalid expansion checkpoint cursor")
        if not np.array_equal(vectors[reused_positions], seed_vectors):
            raise ValueError("Copied seed vectors changed")
        for start in range(0, cursor, 4096):
            validate_embeddings(
                vectors[pending[start : min(start + 4096, cursor)]], config["dimension"]
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Output directory contains unrecognized files")
        output_dir.mkdir(parents=True, exist_ok=True)
        selected[["isic_id", "split"]].to_csv(output_dir / "rows.csv", index=False)
        vectors = np.lib.format.open_memmap(
            output_dir / "vectors.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(selected), config["dimension"]),
        )
        vectors[reused_positions] = seed_vectors
        vectors.flush()
        state = {
            "identity": identity,
            "rows_done": len(reused_positions),
            "new_rows_done": 0,
            "reused_cache": provenance,
            "new_rows_total": len(pending),
            "complete": False,
            "test_encoded": False,
            "extraction_seconds": 0.0,
            "rows_sha256": sha256_file(output_dir / "rows.csv"),
        }
        write_json(checkpoint, state)
    model, hardware = load_encoder(config, model_cache_dir, device)
    for previous in [seed_state["hardware"], state.get("hardware", hardware)]:
        for field in ["device", "dtype", "versions", "max_soft_tokens"]:
            if previous[field] != hardware[field]:
                raise ValueError(f"Cannot combine embeddings with different {field}")
    if batch_size != seed_state["batch_size"]:
        raise ValueError("Use the same batch size as the reused cache")
    state["hardware"] = hardware
    state["batch_size"] = batch_size
    write_json(checkpoint, state)
    print(f"Reusing {len(reused_positions):,}; extracting {len(pending):,} new images", flush=True)
    with ImageArchive(archive_path) as archive:
        for start in range(state["new_rows_done"], len(pending), 512):
            tick = time.monotonic()
            end = min(start + 512, len(pending))
            positions = pending[start:end]
            images = [archive.read(member) for member in selected.iloc[positions].image_member]
            vectors[positions] = encode(model, images, config, batch_size)
            synchronize(hardware["device"])
            vectors.flush()
            state["new_rows_done"] = end
            state["rows_done"] = len(reused_positions) + end
            state["extraction_seconds"] += time.monotonic() - tick
            write_json(checkpoint, state)
            if end % 4096 == 0 or end == len(pending):
                rate = end / state["extraction_seconds"]
                print(
                    f"New {end:,}/{len(pending):,}; {rate:.1f} images/s; "
                    f"ETA {(len(pending) - end) / rate / 60:.1f} min",
                    flush=True,
                )
    state["vectors_sha256"] = sha256_file(output_dir / "vectors.npy")
    state["complete"] = True
    write_json(checkpoint, state)
    print("Full embedding cache complete", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/embeddinggemma2-70.json"))
    parser.add_argument("--reuse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    expand(
        args.config,
        Path("data/processed"),
        Path("data/raw/input.zip"),
        Path("data/model-cache"),
        args.reuse_dir,
        args.output_dir,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
