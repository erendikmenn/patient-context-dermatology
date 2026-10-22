"""Train once on train patients, select iteration/threshold only on validation."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from derm_context.download import sha256_file
from derm_context.features import feature_columns, make_preprocessor, select_features
from derm_context.metrics import choose_threshold, classification_metrics
from derm_context.prepare import assert_split_integrity


def load_prepared(processed_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    if sha256_file(processed_dir / "metadata.parquet") != manifest["processed_sha256"]:
        raise ValueError("Prepared metadata changed since preparation")
    if sha256_file(processed_dir / "splits.csv") != manifest["split_sha256"]:
        raise ValueError("Split assignments changed since preparation")
    frame = pd.read_parquet(processed_dir / "metadata.parquet")
    assert_split_integrity(frame)
    return frame, manifest


def save_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    metrics, dummy = report["validation"], report["constant_baseline_validation"]
    op = metrics["operating_point"]
    lines = [
        f"# İlk model sonuçları — {report['run_name']}",
        "",
        "Bu sonuçlar hasta bazında ayrılmış **doğrulama** grubundadır. Test grubu açılmadı.",
        "",
        (
            f"Özellik grubu: `{report['feature_set']}`. "
            "Model çıktıları henüz kalibre edilmemiştir; klinik risk olarak yorumlanmamalıdır."
        ),
        "",
        "| Metrik | XGBoost | Sabit prevalans tahmini |",
        "| --- | ---: | ---: |",
    ]
    for label, key in [
        ("Average precision (AP)", "average_precision"),
        ("ROC-AUC", "roc_auc"),
        ("pAUC, TPR ≥ 0.80 (0–0.20)", "pauc_tpr_ge_0_8_unstandardized"),
        ("Brier (düşük daha iyi)", "brier_score"),
    ]:
        lines.append(f"| {label} | {metrics[key]:.6f} | {dummy[key]:.6f} |")
    lines += [
        "",
        (
            f"Doğrulama: {metrics['n']:,} lezyon; {metrics['positives']} malign. "
            f"Malignite prevalansı %{100 * metrics['prevalence']:.4f}."
        ),
        "",
        (
            f"En iyi ağaç turu (sıfırdan başlayan indeks): {report['best_iteration']}. "
            f"Eğitim ve kayıt süresi: {report['elapsed_seconds']:.1f} saniye."
        ),
        "",
        "## Araştırma eşiği",
        "",
        (
            "Eşik doğrulamada seçildi ve aşağıdaki ölçümler aynı grupta yapıldı; "
            "bağımsız test performansı değildir."
        ),
        "",
        (
            f"Eşik: {op['threshold']:.6f}; duyarlılık: %{100 * op['sensitivity']:.2f}; "
            f"özgüllük: %{100 * op['specificity']:.2f}; precision: %{100 * op['precision']:.2f}."
        ),
        (
            f"Doğru pozitif: {op['true_positive']}; yanlış negatif: {op['false_negative']}; "
            f"yanlış pozitif: {op['false_positive']}; doğru negatif: {op['true_negative']}."
        ),
        "",
        "## Sınırlar",
        "",
        (
            "Bu tek sabit bölme üzerinde ilk geliştirme deneyidir. Hasta bazında güven "
            "aralıkları, merkezler arası doğrulama, kalibrasyon ve final test henüz yapılmadı. "
            "Morfometri özellikleri veri setinde hazır bulunan görüntü ölçümleridir. "
            "EmbeddingGemma 2 bu eğitimde kullanılmadı."
        ),
        "",
    ]
    destination.with_suffix(".md").write_text("\n".join(lines))


def train(
    processed_dir: Path, artifact_dir: Path, reports_dir: Path, config_path: Path, run_name: str
) -> dict:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    output = artifact_dir / run_name
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}; choose a new run name")
    config = json.loads(config_path.read_text())
    frame, manifest = load_prepared(processed_dir)
    feature_set = config["feature_set"]
    training = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    # Test rows do not participate in fitting, early stopping, thresholds, or metrics.
    x_train = select_features(training, feature_set)
    x_valid = select_features(validation, feature_set)
    preprocessor = make_preprocessor(feature_set)
    train_matrix = preprocessor.fit_transform(x_train).astype(np.float32)
    valid_matrix = preprocessor.transform(x_valid).astype(np.float32)
    y_train, y_valid = training.target.to_numpy(), validation.target.to_numpy()
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    if config["class_weight"] not in {"sqrt_ratio", "none"}:
        raise ValueError("class_weight must be sqrt_ratio or none")
    weight = float(np.sqrt(ratio)) if config["class_weight"] == "sqrt_ratio" else 1.0
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
    parameters.update(
        tree_method="hist",
        device="cpu",
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=config["seed"],
        scale_pos_weight=weight,
    )
    print(
        f"Training {len(training):,} lesions, validating {len(validation):,}; "
        f"features={train_matrix.shape[1]}, positive weight={weight:.3f}",
        flush=True,
    )
    started = time.monotonic()
    model = XGBClassifier(**parameters)
    model.fit(train_matrix, y_train, eval_set=[(valid_matrix, y_valid)], verbose=50)
    probabilities = model.predict_proba(valid_matrix)[:, 1]
    threshold = choose_threshold(y_valid, probabilities, config["target_sensitivity"])
    constant = np.full(len(y_valid), y_train.mean())
    output.mkdir(parents=True)
    model.save_model(output / "model.json")
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    reloaded_model = XGBClassifier()
    reloaded_model.load_model(output / "model.json")
    reloaded_preprocessor = joblib.load(output / "preprocessor.joblib")
    reloaded_predictions = reloaded_model.predict_proba(
        reloaded_preprocessor.transform(x_valid).astype(np.float32)
    )[:, 1]
    np.testing.assert_allclose(probabilities, reloaded_predictions, rtol=0, atol=1e-7)
    numeric, categorical = feature_columns(feature_set)
    versions = {
        lib: importlib.metadata.version(lib)
        for lib in ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "joblib"]
    }
    feature_names = preprocessor.get_feature_names_out()
    importance = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "run_name": run_name,
        "feature_set": feature_set,
        "config": config,
        "xgboost_parameters": parameters,
        "feature_columns": numeric + categorical,
        "transformed_features": len(feature_names),
        "best_iteration": int(model.best_iteration),
        "train_rows": len(training),
        "train_positives": int(y_train.sum()),
        "train_patients": int(training.patient_id.nunique()),
        "validation_patients": int(validation.patient_id.nunique()),
        "validation_positive_patients": int(validation.groupby("patient_id").target.max().sum()),
        "validation": classification_metrics(y_valid, probabilities, threshold),
        "constant_baseline_validation": classification_metrics(y_valid, constant),
        "test_evaluated": False,
        "calibrated": False,
        "threshold_selected_on": "validation",
        "patient_overlap": 0,
        "reload_predictions_verified": True,
        "elapsed_seconds": time.monotonic() - started,
        "processed_sha256": manifest["processed_sha256"],
        "split_sha256": manifest["split_sha256"],
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "learning_curve": model.evals_result(),
        "top_features_gain": [
            {"feature": str(name), "importance": float(value)} for name, value in importance[:15]
        ],
    }
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    predictions = validation[["isic_id", "patient_id", "target"]].copy()
    predictions["score"] = probabilities
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    save_report(report, reports_dir / run_name)
    print(json.dumps(report["validation"], indent=2), flush=True)
    return report


def predict(input_path: Path, model_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output exists: {output_path}")
    frame = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    run = json.loads((model_dir / "run.json").read_text())
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = XGBClassifier()
    model.load_model(model_dir / "model.json")
    features = select_features(frame, run["feature_set"])
    matrix = preprocessor.transform(features).astype(np.float32)
    result = pd.DataFrame({"score": model.predict_proba(matrix)[:, 1]})
    if "isic_id" in frame:
        result.insert(0, "isic_id", frame.isic_id.to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
