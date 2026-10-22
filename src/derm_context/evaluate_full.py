"""Audit the matched full-data experiment and estimate paired patient-bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from derm_context.embeddings import write_json

METRICS = ["average_precision", "roc_auc", "pauc_tpr_ge_0_8_unstandardized"]
RUNS = ["metadata-full-control-v1", "image-eg2-70-full-v1", "fusion-eg2-70-full-v1"]


def weighted_metrics(y, scores, weights):
    standardized = roc_auc_score(1 - y, 1 - scores, sample_weight=weights, max_fpr=0.2)
    return np.array(
        [
            average_precision_score(y, scores, sample_weight=weights),
            roc_auc_score(y, scores, sample_weight=weights),
            0.02 + (2 * standardized - 1) * 0.18,
        ]
    )


def paired_bootstrap(y, patients, scores, iterations=1000, seed=42):
    if iterations < 2:
        raise ValueError("At least two bootstrap iterations are required")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes are required")
    patients, codes = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(iterations):
        counts = np.bincount(
            rng.integers(len(patients), size=len(patients)), minlength=len(patients)
        )
        weights = counts[codes]
        if not np.any(weights[y == 1]) or not np.any(weights[y == 0]):
            continue
        samples.append([weighted_metrics(y, score, weights) for score in scores])
        if (i + 1) % 100 == 0:
            print(f"Patient bootstrap {i + 1}/{iterations}", flush=True)
    if len(samples) < 2:
        raise ValueError("Too few bootstrap samples containing both classes")
    samples = np.array(samples)
    point = np.array([weighted_metrics(y, score, np.ones(len(y))) for score in scores])
    differences = []
    for j in range(1, len(scores)):
        metrics = {}
        for k, metric in enumerate(METRICS):
            delta = samples[:, j, k] - samples[:, 0, k]
            low, high = np.quantile(delta, [0.025, 0.975])
            metrics[metric] = {
                "delta": float(point[j, k] - point[0, k]),
                "ci95_lower": float(low),
                "ci95_upper": float(high),
            }
        differences.append(metrics)
    return {
        "requested_iterations": iterations,
        "valid_iterations": len(samples),
        "seed": seed,
        "patients": len(patients),
        "differences": differences,
        "method": "paired patient-cluster percentile bootstrap; all lesions share patient weight",
        "conditions": "fixed fitted models; validation selection uncertainty not included; not a final test",
    }


def evaluate(artifact_dir, reports_dir, iterations=1000):
    reports = [json.loads((reports_dir / f"{name}.json").read_text()) for name in RUNS]
    baseline = json.loads((reports_dir / "baseline-v1.json").read_text())
    for r in reports:
        if not r["all_original_training_rows"] or not r["all_original_validation_rows"]:
            raise ValueError("Full-data comparison requires all training and validation rows")
        for field in [
            "config",
            "xgboost_parameters",
            "processed_sha256",
            "split_sha256",
            "versions",
        ]:
            if r[field] != baseline[field]:
                raise ValueError(f"Baseline and full model differ in {field}")
        if r["train_rows"] != baseline["train_rows"] or r["test_evaluated"]:
            raise ValueError("Unexpected cohort or test evaluation")
    for field in ["training_cohort_sha256", "validation_cohort_sha256"]:
        if len({r[field] for r in reports}) != 1:
            raise ValueError(f"Ordered cohort identity mismatch: {field}")
    frames = []
    for name in ["baseline-v1", *RUNS]:
        frame = pd.read_parquet(artifact_dir / name / "validation_predictions.parquet")
        if frame.isic_id.duplicated().any():
            raise ValueError("Duplicate prediction IDs")
        frames.append(frame.sort_values("isic_id").reset_index(drop=True))
    base = frames[0]
    for frame in frames[1:]:
        if not frame[["isic_id", "patient_id", "target"]].equals(
            base[["isic_id", "patient_id", "target"]]
        ):
            raise ValueError("Prediction identities or labels do not match")
        if not np.isfinite(frame.score).all():
            raise ValueError("Invalid predictions")
    np.testing.assert_allclose(base.score, frames[1].score, atol=1e-7, rtol=0)
    print(
        "Baseline reproduced; ordered cohorts, settings, versions and predictions verified",
        flush=True,
    )
    bootstrap = paired_bootstrap(
        base.target.to_numpy(),
        base.patient_id.to_numpy(),
        [f.score.to_numpy() for f in frames[1:]],
        iterations=iterations,
    )
    for name, result in zip(RUNS[1:], bootstrap["differences"], strict=True):
        result["run_name"] = name
    result = {
        "baseline_reproduced": True,
        "baseline_max_absolute_prediction_difference": float(
            np.max(np.abs(base.score - frames[1].score))
        ),
        "training_rows": reports[0]["train_rows"],
        "validation_rows": len(base),
        "validation_positives": int(base.target.sum()),
        "training_cohort_sha256": reports[0]["training_cohort_sha256"],
        "validation_cohort_sha256": reports[0]["validation_cohort_sha256"],
        "matched_settings_and_versions": True,
        "test_evaluated": False,
        "bootstrap": bootstrap,
    }
    write_json(reports_dir / "full_comparison_audit.json", result)
    lines = [
        "# Tam veri karşılaştırması: eşleşme ve belirsizlik",
        "",
        (
            "Metadata baseline aynı tahminlerle yeniden üretildi. Eğitim ve doğrulama kimlikleri, "
            "sıraları, etiketleri, XGBoost ayarları ve paket sürümleri eşleşiyor. Test kullanılmadı."
        ),
        "",
        "Aşağıdaki farklar metadata kontrolüne göredir; pozitif fark daha yüksek metrik demektir.",
        "",
        "| Model | Metrik | Fark | Hasta bootstrap %95 aralığı |",
        "| --- | --- | ---: | --- |",
    ]
    for name, delta in zip(RUNS[1:], bootstrap["differences"], strict=True):
        for metric in METRICS:
            v = delta[metric]
            lines.append(
                f"| {name} | {metric} | {v['delta']:+.6f} | "
                f"[{v['ci95_lower']:+.6f}, {v['ci95_upper']:+.6f}] |"
            )
    lines += [
        "",
        (
            f"{bootstrap['valid_iterations']} geçerli tekrar; seed={bootstrap['seed']}; "
            f"{bootstrap['patients']} hasta. Her tekrar hasta bazında örneklenir; hastanın tüm "
            "lezyonlarına aynı ağırlık verilir ve modeller aynı örneklemde karşılaştırılır."
        ),
        "",
        (
            "Aralıklar sabit eğitilmiş modellere koşulludur. Aynı doğrulamada erken durdurma/model "
            "seçimi yapıldığı için seçim belirsizliğini kapsamaz; bağımsız test veya klinik "
            "üstünlük kanıtı değildir. Sıfırı içeren aralık yön konusunda belirsizlik gösterir."
        ),
        "",
    ]
    (reports_dir / "full_comparison_audit.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    evaluate(Path("artifacts"), Path("reports"), args.iterations)


if __name__ == "__main__":
    main()
