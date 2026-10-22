"""Compare frozen image features and image+metadata on the existing patient split."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from derm_context.dataset import load_prepared
from derm_context.download import sha256_file
from derm_context.embeddings import validate_embeddings, write_json
from derm_context.features import make_preprocessor, select_features
from derm_context.metrics import choose_threshold, classification_metrics


def load_cache(cache_dir: Path, frame: pd.DataFrame, source: dict):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["rows_done"] != manifest["identity"]["rows"]:
        raise ValueError("Embedding extraction is incomplete")
    if manifest["identity"]["processed_sha256"] != source["processed_sha256"]:
        raise ValueError("Embeddings and prepared dataset do not match")
    if manifest["identity"]["split_sha256"] != source["split_sha256"]:
        raise ValueError("Embedding split changed")
    for filename, key in [("rows.csv", "rows_sha256"), ("vectors.npy", "vectors_sha256")]:
        if sha256_file(cache_dir / filename) != manifest[key]:
            raise ValueError(f"Embedding cache checksum mismatch: {filename}")
    rows = pd.read_csv(cache_dir / "rows.csv")
    if rows.isic_id.duplicated().any() or not set(rows.isic_id).issubset(set(frame.isic_id)):
        raise ValueError("Invalid embedding image IDs")
    selected = frame.set_index("isic_id").loc[rows.isic_id].reset_index()
    if not np.array_equal(selected.split, rows.split) or set(selected.split) != {
        "train",
        "validation",
    }:
        raise ValueError("Unexpected cache split or test leakage")
    expected_validation = set(frame.loc[frame.split == "validation", "isic_id"])
    if set(selected.loc[selected.split == "validation", "isic_id"]) != expected_validation:
        raise ValueError("Comparison requires the entire original validation group")
    values = np.load(cache_dir / "vectors.npy", mmap_mode="r")
    if values.shape != (len(rows), manifest["identity"]["config"]["dimension"]):
        raise ValueError("Embedding row count/shape mismatch")
    for start in range(0, len(values), 4096):
        validate_embeddings(values[start : start + 4096], values.shape[1])
    return selected, values, manifest


def train_embeddings(
    cache_dir: Path,
    processed_dir: Path,
    config_path: Path,
    artifact_dir: Path,
    reports_dir: Path,
    mode: str,
    run_name: str,
) -> dict:
    if mode not in {"image", "fusion", "metadata"}:
        raise ValueError("Unknown model mode")
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run_name must be a single directory name")
    destination = artifact_dir / run_name
    if destination.exists():
        raise FileExistsError(f"Run exists: {run_name}")
    config = json.loads(config_path.read_text())
    frame, source = load_prepared(processed_dir)
    selected, vectors, embedding_manifest = load_cache(cache_dir, frame, source)
    train_mask = (selected.split == "train").to_numpy()
    validation_mask = (selected.split == "validation").to_numpy()
    train_frame = selected[train_mask]
    validation_frame = selected[validation_mask]
    y_train, y_valid = train_frame.target.to_numpy(), validation_frame.target.to_numpy()
    preprocessor = None
    if mode in {"fusion", "metadata"}:
        preprocessor = make_preprocessor(config["feature_set"])
        train_meta = preprocessor.fit_transform(
            select_features(train_frame, config["feature_set"])
        ).astype(np.float32)
        valid_meta = preprocessor.transform(
            select_features(validation_frame, config["feature_set"])
        ).astype(np.float32)
    if mode == "image":
        x_train, x_valid = vectors[train_mask], vectors[validation_mask]
    elif mode == "fusion":
        x_train = np.concatenate([vectors[train_mask], train_meta], axis=1)
        x_valid = np.concatenate([vectors[validation_mask], valid_meta], axis=1)
    else:
        x_train, x_valid = train_meta, valid_meta
    parameters = {
        k: config[k]
        for k in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "early_stopping_rounds",
            "n_jobs",
        ]
    }
    if config["class_weight"] not in {"sqrt_ratio", "none"}:
        raise ValueError("Unsupported class weight")
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    weight = float(np.sqrt(ratio)) if config["class_weight"] == "sqrt_ratio" else 1.0
    parameters.update(
        tree_method="hist",
        device="cpu",
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=config["seed"],
        scale_pos_weight=weight,
    )
    print(
        f"{mode}: train={len(y_train):,}, validation={len(y_valid):,}, features={x_train.shape[1]}",
        flush=True,
    )
    started = time.monotonic()
    model = XGBClassifier(**parameters)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=50)
    scores = model.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid, scores, config["target_sensitivity"])
    destination.mkdir(parents=True)
    model.save_model(destination / "model.json")
    if preprocessor is not None:
        joblib.dump(preprocessor, destination / "preprocessor.joblib")
    reloaded = XGBClassifier()
    reloaded.load_model(destination / "model.json")
    np.testing.assert_allclose(scores, reloaded.predict_proba(x_valid)[:, 1], atol=1e-7, rtol=0)
    report = {
        "run_name": run_name,
        "mode": mode,
        "config": config,
        "xgboost_parameters": parameters,
        "embedding_config": embedding_manifest["identity"]["config"],
        "embedding_hardware": embedding_manifest["hardware"],
        "embedding_vectors_sha256": embedding_manifest["vectors_sha256"],
        "train_rows": len(y_train),
        "train_positives": int(y_train.sum()),
        "train_patients": int(train_frame.patient_id.nunique()),
        "all_original_training_rows": set(train_frame.isic_id)
        == set(frame.loc[frame.split == "train", "isic_id"]),
        "all_original_validation_rows": True,
        "validation_patients": int(validation_frame.patient_id.nunique()),
        "validation_positive_patients": int(
            validation_frame.groupby("patient_id").target.max().sum()
        ),
        "feature_count": x_train.shape[1],
        "best_iteration": int(model.best_iteration),
        "validation": classification_metrics(y_valid, scores, threshold),
        "constant_baseline_validation": classification_metrics(
            y_valid, np.full(len(y_valid), y_train.mean())
        ),
        "test_evaluated": False,
        "calibrated": False,
        "encoder_frozen": True,
        "threshold_selected_on": "validation",
        "reload_predictions_verified": True,
        "elapsed_train_seconds": time.monotonic() - started,
        "processed_sha256": source["processed_sha256"],
        "split_sha256": source["split_sha256"],
        "learning_curve": model.evals_result(),
    }
    write_json(destination / "run.json", report)
    write_json(reports_dir / f"{run_name}.json", report)
    predictions = validation_frame[["isic_id", "patient_id", "target"]].copy()
    predictions["score"] = scores
    predictions.to_parquet(destination / "validation_predictions.parquet", index=False)
    print(json.dumps(report["validation"], indent=2), flush=True)
    return report


def comparison(reports_dir: Path, run_names: list[str]) -> None:
    runs = [json.loads((reports_dir / f"{name}.json").read_text()) for name in run_names]
    if (
        len({r["split_sha256"] for r in runs}) != 1
        or len({r["validation"]["n"] for r in runs}) != 1
    ):
        raise ValueError("Cannot compare different validation cohorts")
    lines = [
        "# EmbeddingGemma 2 model karşılaştırması",
        "",
        "Bütün sonuçlar aynı hasta bazlı doğrulama grubundadır; test değerlendirilmedi.",
        "",
        "| Model | Eğitim satırı | Özellik | AP | ROC-AUC | pAUC (0–0.20) | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        m = run["validation"]
        lines.append(
            f"| {run['run_name']} | {run['train_rows']:,} | "
            f"{run.get('feature_count', run.get('transformed_features'))} | "
            f"{m['average_precision']:.6f} | {m['roc_auc']:.6f} | "
            f"{m['pauc_tpr_ge_0_8_unstandardized']:.6f} | {m['brier_score']:.6f} |"
        )
    lines += [
        "",
        "## Doğrulamada seçilen yüksek duyarlılık eşikleri",
        "",
        "| Model | Duyarlılık | Özgüllük | Precision | Yanlış pozitif |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        op = run["validation"]["operating_point"]
        lines.append(
            f"| {run['run_name']} | {op['sensitivity']:.4f} | {op['specificity']:.4f} | "
            f"{op['precision']:.4f} | {op['false_positive']:,} |"
        )
    lines += [
        "",
        "## Yorumlama koşulları",
        "",
        (
            "Encoder donduruldu; XGBoost başlıkları eğitildi. Aynı temel XGBoost ayarları kullanıldı, "
            "erken durdurma her model için doğrulamada seçildi. Çok boyutlu embedding'in en iyi olası "
            "başlığı bulunduğu iddia edilmez. Eşikler aynı doğrulamada seçildiği için eşik metrikleri "
            "bağımsız test performansı değildir. Olasılıklar kalibre edilmedi."
        ),
        "",
    ]
    if len({r["train_rows"] for r in runs}) != 1:
        lines += [
            (
                "Eğitim satırı sayıları farklıdır. Daha küçük eğitim örneği kullanan modelin "
                "farkı yalnızca embedding'e bağlanamaz; tablo bu ayrımı açıkça gösterir."
            ),
            "",
        ]
    else:
        lines += [
            (
                "Karşılaştırılan modellerin eğitim satırı sayıları eşittir. Embedding modelleri "
                "tüm özgün eğitim ve doğrulama satırlarını kullanıyorsa rapor JSON'larında "
                "`all_original_training_rows=true` ve `all_original_validation_rows=true` bulunur."
            ),
            "",
        ]
    (reports_dir / "embedding_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "fusion", "metadata"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/embeddings/embeddinggemma2"))
    args = parser.parse_args()
    train_embeddings(
        args.cache_dir,
        Path("data/processed"),
        Path("configs/baseline.json"),
        Path("artifacts"),
        Path("reports"),
        args.mode,
        args.run_name,
    )
