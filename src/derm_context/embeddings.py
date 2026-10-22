"""Real image embeddings with a pinned, frozen EmbeddingGemma 2 encoder."""

import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from derm_context.dataset import load_prepared
from derm_context.download import sha256_file
from derm_context.images import ImageArchive


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def validate_embeddings(values: np.ndarray, dimension: int = 768) -> None:
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"Unexpected embedding shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Encoder returned non-finite embeddings")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=0.015):
        raise ValueError("Encoder embeddings are not unit-normalized")


def load_encoder(config: dict, cache_dir: Path, device: str = "auto"):
    if platform.system() == "Darwin" and "xgboost" in sys.modules:
        raise RuntimeError("Run image encoding in a fresh process without importing XGBoost first")
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    if config["dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("EmbeddingGemma 2 requires bfloat16 or float32; float16 is unsafe")
    if device == "cpu":
        dtype = torch.float32
    print(
        f"Loading {config['model_id']} revision={config['revision']} on {device}, {dtype}",
        flush=True,
    )
    started = time.monotonic()
    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        device=device,
        cache_folder=str(cache_dir),
        trust_remote_code=False,
        model_kwargs={"dtype": dtype},
        config_kwargs={"audio_config": None} if config["disable_audio"] else {},
        processor_kwargs={"max_soft_tokens": config.get("max_soft_tokens", 280)},
    )
    actual_tokens = model[0].processor.image_processor.max_soft_tokens
    if actual_tokens != config.get("max_soft_tokens", 280):
        raise ValueError(f"Image token budget was not applied: {actual_tokens}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "max_soft_tokens": actual_tokens,
        "device": device,
        "dtype": str(dtype),
        "load_seconds": time.monotonic() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "sentence-transformers", "pillow"]
        },
    }
    print(f"Encoder loaded: {metadata}", flush=True)
    return model, metadata


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def encode(model, images: list, config: dict, batch_size: int) -> np.ndarray:
    vectors = model.encode(
        [{"image": image} for image in images],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config["normalize_embeddings"],
        truncate_dim=config["dimension"],
        show_progress_bar=False,
    )
    values = np.asarray(vectors, dtype=np.float32)
    validate_embeddings(values, config["dimension"])
    return values


def benchmark(
    config_path: Path,
    processed_dir: Path,
    archive_path: Path,
    cache_dir: Path,
    output_path: Path,
    device: str = "auto",
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Benchmark exists: {output_path}")
    config = json.loads(config_path.read_text())
    frame, manifest = load_prepared(processed_dir)
    training = frame[frame.split == "train"]
    sample = training.sample(
        n=min(config["benchmark_images"], len(training)), random_state=config["seed"]
    )
    decode_start = time.monotonic()
    with ImageArchive(archive_path) as archive:
        images = [archive.read(member) for member in sample.image_member]
    decode_seconds = time.monotonic() - decode_start
    model, hardware = load_encoder(config, cache_dir, device)
    measurements = []
    for batch_size in config["batch_sizes"]:
        encode(model, images[:batch_size], config, batch_size)
        synchronize(hardware["device"])
        start = time.monotonic()
        vectors = encode(model, images, config, batch_size)
        synchronize(hardware["device"])
        seconds = time.monotonic() - start
        if float(vectors.std(axis=0).mean()) < 1e-6:
            raise ValueError("Encoder returned effectively identical vectors for all images")
        measurement = {
            "batch_size": batch_size,
            "images": len(images),
            "encode_seconds": seconds,
            "images_per_second": len(images) / seconds,
            "mean_coordinate_std": float(vectors.std(axis=0).mean()),
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    best = max(measurements, key=lambda m: m["images_per_second"])
    train_validation_rows = int(frame.split.isin(["train", "validation"]).sum())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "config": config,
        "hardware": hardware,
        "measurements": measurements,
        "decoded_images": len(images),
        "decode_seconds": decode_seconds,
        "sample_source": "training patients only; uniform lesion sample",
        "process_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024),
        "selected_batch_size": best["batch_size"],
        "train_validation_rows": train_validation_rows,
        "estimated_encode_seconds_full_train_validation": train_validation_rows
        / best["images_per_second"],
        "estimate_excludes_decode_checkpoint_and_variable_size_overhead": True,
        "processed_sha256": manifest["processed_sha256"],
        "split_sha256": manifest["split_sha256"],
        "test_encoded": False,
    }
    write_json(output_path, report)
    return report


def select_cohort(frame, max_train_negatives: int | None = None, seed: int = 42):
    selected = frame[frame.split.isin(["train", "validation"])].copy()
    if max_train_negatives is not None:
        if max_train_negatives < 1:
            raise ValueError("max_train_negatives must be positive")
        negatives = selected[(selected.split == "train") & (selected.target == 0)]
        keep = negatives.sample(n=min(len(negatives), max_train_negatives), random_state=seed).index
        selected = selected[
            (selected.split != "train") | (selected.target == 1) | selected.index.isin(keep)
        ]
    return selected.sort_values("isic_id").reset_index(drop=True)


def extract(
    config_path: Path,
    processed_dir: Path,
    archive_path: Path,
    model_cache_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str = "auto",
    max_train_negatives: int | None = None,
) -> dict:
    config = json.loads(config_path.read_text())
    frame, source_manifest = load_prepared(processed_dir)
    if sha256_file(archive_path) != source_manifest["files"]["input.zip"]["sha256"]:
        raise ValueError("Image archive changed since data preparation")
    selected = select_cohort(frame, max_train_negatives, config["seed"])
    identity = {
        "config": config,
        "processed_sha256": source_manifest["processed_sha256"],
        "split_sha256": source_manifest["split_sha256"],
        "max_train_negatives": max_train_negatives,
        "rows": len(selected),
    }
    checkpoint = output_dir / "manifest.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity:
            raise ValueError("Embedding cache belongs to a different model/cohort/config")
        if state.get("complete"):
            for filename, key in [("vectors.npy", "vectors_sha256"), ("rows.csv", "rows_sha256")]:
                if sha256_file(output_dir / filename) != state[key]:
                    raise ValueError(f"Completed embedding cache is corrupt: {filename}")
            print("Verified completed embedding cache", flush=True)
            return state
        import pandas as pd

        saved_ids = pd.read_csv(output_dir / "rows.csv")
        if saved_ids.isic_id.tolist() != selected.isic_id.tolist():
            raise ValueError("Embedding row order changed")
        vectors = np.load(output_dir / "vectors.npy", mmap_mode="r+")
        if vectors.shape != (len(selected), config["dimension"]):
            raise ValueError("Embedding cache has an invalid shape")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Unrecognized files in embedding output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        selected[["isic_id", "split"]].to_csv(output_dir / "rows.csv", index=False)
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
            "test_encoded": False,
            "extraction_seconds": 0.0,
        }
        write_json(checkpoint, state)
    model, hardware = load_encoder(config, model_cache_dir, device)
    if "hardware" in state:
        for field in ["device", "dtype", "versions"]:
            if state["hardware"][field] != hardware[field]:
                raise ValueError(f"Cannot resume with changed {field}")
    state["hardware"] = hardware
    state["batch_size"] = batch_size
    chunk_size = max(512, batch_size)
    print(f"Extracting {len(selected):,} images starting at {state['rows_done']:,}", flush=True)
    with ImageArchive(archive_path) as archive:
        for start in range(state["rows_done"], len(selected), chunk_size):
            tick = time.monotonic()
            end = min(start + chunk_size, len(selected))
            images = [archive.read(member) for member in selected.iloc[start:end].image_member]
            vectors[start:end] = encode(model, images, config, batch_size)
            synchronize(hardware["device"])
            vectors.flush()
            state["rows_done"] = end
            state["extraction_seconds"] += time.monotonic() - tick
            write_json(checkpoint, state)
            if end % 4096 == 0 or end == len(selected):
                rate = end / state["extraction_seconds"]
                print(
                    f"{end:,}/{len(selected):,}; {rate:.1f} images/s; "
                    f"ETA {(len(selected) - end) / rate / 60:.1f} min",
                    flush=True,
                )
    state["complete"] = True
    state["vectors_sha256"] = sha256_file(output_dir / "vectors.npy")
    state["rows_sha256"] = sha256_file(output_dir / "rows.csv")
    write_json(checkpoint, state)
    print("Embedding extraction complete", flush=True)
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["benchmark", "extract"], default="benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/embeddinggemma2.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/embeddinggemma2-benchmark.json")
    )
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-dir", type=Path, default=Path("data/embeddings/embeddinggemma2")
    )
    parser.add_argument("--max-train-negatives", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark(
            args.config,
            Path("data/processed"),
            Path("data/raw/input.zip"),
            Path("data/model-cache"),
            args.output,
            args.device,
        )
    else:
        if args.batch_size < 1:
            parser.error("batch-size must be positive")
        extract(
            args.config,
            Path("data/processed"),
            Path("data/raw/input.zip"),
            Path("data/model-cache"),
            args.embedding_dir,
            args.batch_size,
            args.device,
            args.max_train_negatives,
        )
