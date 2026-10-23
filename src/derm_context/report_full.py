"""Render the completed full-data experiment from audited, aggregate results."""

import json
from pathlib import Path


def render(root: Path) -> None:
    reports = root / "reports"
    audit = json.loads((reports / "full_comparison_audit.json").read_text())
    extraction = json.loads((reports / "embeddinggemma2-70-full-extraction.json").read_text())
    if not audit["baseline_reproduced"] or not audit["matched_settings_and_versions"]:
        raise ValueError("The matched experiment audit must pass before reporting")
    if audit["test_evaluated"] or not extraction["complete"]:
        raise ValueError("Unexpected test evaluation or incomplete extraction")
    names = ["metadata-full-control-v1", "image-eg2-70-full-v1", "fusion-eg2-70-full-v1"]
    labels = [
        "Metadata + morfometri (baseline tekrar)",
        "Yalnız görüntü embedding'i",
        "Embedding + metadata/morfometri",
    ]
    runs = [json.loads((reports / f"{name}.json").read_text()) for name in names]
    for r in runs:
        if not r["all_original_training_rows"] or not r["all_original_validation_rows"]:
            raise ValueError("Full-cohort runs required")
        if r["training_cohort_sha256"] != audit["training_cohort_sha256"]:
            raise ValueError("Training cohort mismatch")
        if r["validation_cohort_sha256"] != audit["validation_cohort_sha256"]:
            raise ValueError("Validation cohort mismatch")
    table = [
        "| Model | Eğitim lezyonu | AP ↑ | ROC-AUC ↑ | pAUC ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, r in zip(labels, runs, strict=True):
        m = r["validation"]
        table.append(
            f"| {label} | {r['train_rows']:,} | {m['average_precision']:.6f} | "
            f"{m['roc_auc']:.6f} | {m['pauc_tpr_ge_0_8_unstandardized']:.6f} |"
        )
    delta = audit["bootstrap"]["differences"][1]["average_precision"]
    direction = "yükseldi" if delta["delta"] > 0 else "yükselmedi"
    conclusion = (
        f"Tam veri deneyinde birleşik modelin ana metriği AP, metadata baseline'ına göre "
        f"**{direction}** (fark {delta['delta']:+.6f}). Eşleştirilmiş hasta bootstrap "
        f"%95 aralığı [{delta['ci95_lower']:+.6f}, {delta['ci95_upper']:+.6f}]. "
    )
    if delta["ci95_lower"] <= 0 <= delta["ci95_upper"]:
        conclusion += "Aralık sıfırı içeriyor; farkın yönü konusunda belirsizlik var."
    elif delta["ci95_lower"] > 0:
        conclusion += "Bu koşullu aralıkta fark pozitif."
    else:
        conclusion += "Bu koşullu aralıkta fark negatif."
    caveat = (
        "Bootstrap sabit eğitilmiş modellere koşulludur; doğrulama üzerinden erken durdurma "
        "ve model seçiminin belirsizliğini kapsamaz. Bağımsız test veya klinik üstünlük "
        "kanıtı değildir. Test bölümü kullanılmadı ve olasılıklar kalibre edilmedi."
    )
    lines = [
        "# Tam veri deneyi — sonuç özeti",
        "",
        conclusion,
        "",
        *table,
        "",
        (
            f"Üç model de aynı {audit['training_rows']:,} eğitim lezyonu ve aynı "
            f"{audit['validation_rows']:,} doğrulama lezyonuyla çalıştı. Doğrulamada "
            f"{audit['validation_positives']} pozitif var. Metadata baseline'ın tahminleri "
            "yeniden üretildi; kimlikler, sıralar, etiketler, ayarlar ve paket sürümleri eşleşti."
        ),
        "",
        (
            "Gemma encoder donduruldu; bütün eğitim verisiyle XGBoost başlıkları eğitildi. "
            "Görüntü bütçesi 70 token, çıktı 768 boyut. Sonuçlar bu temsil ve sabit "
            "XGBoost ayarları için geçerlidir; Gemma fine-tuning deneyi değildir."
        ),
        "",
        "## Yüksek duyarlılıkta sonuç",
        "",
        "| Model | Duyarlılık | Yanlış pozitif | Yanlış negatif |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, r in zip(labels, runs, strict=True):
        op = r["validation"]["operating_point"]
        lines.append(
            f"| {label} | {op['sensitivity']:.4f} | {op['false_positive']:,} | {op['false_negative']} |"
        )
    lines += [
        "",
        "Eşikler doğrulamada ayrı seçildi; bu eşik sonuçları bağımsız test değildir.",
        "",
        "## Hesaplama",
        "",
        (
            f"{extraction['reused_cache']['rows']:,} pilot vektörü yeniden kullanıldı; "
            f"{extraction['new_rows_done']:,} yeni görüntü işlendi. Tam cache "
            f"{extraction['rows_done']:,} vektör içeriyor. Yeni tamamlanan parçaların toplam "
            f"çıkarım süresi {extraction['new_extraction_minutes']:.2f} dakika; bu ölçüm model "
            "yüklemelerini ve yeniden başlatmada yarıda kalan parçayı içermez. Pilotun önceki "
            f"çıkarımı dahil toplam ölçülen hesaplama {extraction['total_compute_minutes_including_reused_pilot']:.2f} dakika."
        ),
        "",
        "| Başlık | Eğitim/kayıt süresi (s) | En iyi tur (0 tabanlı) |",
        "| --- | ---: | ---: |",
    ]
    for name, r in zip(names, runs, strict=True):
        lines.append(f"| {name} | {r['elapsed_train_seconds']:.2f} | {r['best_iteration']} |")
    lines += [
        "",
        "## Yorumlama sınırı",
        "",
        caveat,
        "",
        (
            "[Detaylı metrikler](embedding_full_comparison.md) · "
            "[Hasta bootstrap ve eşleşme denetimi](full_comparison_audit.md) · "
            "[Çıkarım kaydı](embeddinggemma2-70-full-extraction.json) · "
            "[Tekrar çalıştırma](../docs/FULL_DATA_EXPERIMENT.md)"
        ),
        "",
    ]
    (reports / "full_experiment_summary.md").write_text("\n".join(lines))
    readme_path = root / "README.md"
    readme = readme_path.read_text()
    start, end = readme.index("## Mevcut durum"), readme.index("## Kurulum")
    status = [
        "## Mevcut durum",
        "",
        "23 Ekim 2026: tam eğitim verisi deneyi tamamlandı.",
        "",
        conclusion,
        "",
        *table,
        "",
        (
            "Bütün modeller aynı 298.560 eğitim ve 53.629 doğrulama lezyonunu kullandı. "
            "Metadata baseline aynı tahminlerle yeniden üretildi. Gemma donduruldu; XGBoost "
            "başlıkları eğitildi. Test bölümüne dokunulmadı. 26 yazılım testi geçti."
        ),
        "",
        (
            "[Tam deney özeti](reports/full_experiment_summary.md) · "
            "[Hasta bootstrap](reports/full_comparison_audit.md) · "
            "[Tam veri komutları](docs/FULL_DATA_EXPERIMENT.md) · "
            "[Önceki pilot](reports/embedding_comparison.md)"
        ),
        "",
        caveat,
        "",
        "",
    ]
    readme_path.write_text(readme[:start] + "\n".join(status) + readme[end:])
    document = root / "docs/FULL_DATA_EXPERIMENT.md"
    content = document.read_text()
    marker = "## Çalıştırma durumu"
    content = content[: content.index(marker)] + marker + "\n\nTamamlandı. " + conclusion
    content += (
        "\n\n" + "\n".join(table) + "\n\n[Sonuç özeti](../reports/full_experiment_summary.md) · "
    )
    content += "[Metrikler](../reports/embedding_full_comparison.md) · "
    content += "[Belirsizlik ve eşleşme denetimi](../reports/full_comparison_audit.md)\n"
    document.write_text(content)


if __name__ == "__main__":
    render(Path.cwd())
