"""Render the held-out test report from audited aggregate results."""

import json
from pathlib import Path

from derm_context.download import sha256_file

LABELS = {
    "metadata": "Metadata + morfometri",
    "image": "Yalnız görüntü embedding'i",
    "fusion": "Embedding + metadata/morfometri",
}


def render(root=Path(".")):
    report = json.loads((root / "reports/heldout_test_results.json").read_text())
    protocol_path = root / "reports/heldout_test_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if report["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Reported results and frozen protocol differ")
    if not report["frozen_files_verified_before_and_after"] or report["refit"]:
        raise ValueError("Unverified frozen test results")
    if report["test_cohort_sha256"] != protocol["test_cohort_sha256"]:
        raise ValueError("Test cohort differs")
    for model, frozen in zip(report["models"], protocol["models"], strict=True):
        if any(model[key] != value for key, value in frozen.items()):
            raise ValueError("Reported model differs from frozen protocol")
        if model["test"]["operating_point"]["threshold"] != frozen["threshold"]:
            raise ValueError("Reported threshold differs from frozen protocol")
    fusion = report["bootstrap"]["differences"][1]["average_precision"]
    direction = "daha yüksek" if fusion["delta"] > 0 else "daha düşük"
    uncertainty = (
        "Aralık sıfırı içeriyor; bu testte katkının yönü konusunda belirsizlik sürüyor."
        if fusion["ci95_lower"] <= 0 <= fusion["ci95_upper"]
        else (
            "Aralığın tamamı pozitif; bu test grubunda AP artışını destekliyor."
            if fusion["ci95_lower"] > 0
            else "Aralığın tamamı negatif; bu test grubunda AP düşüşünü destekliyor."
        )
    )
    conclusion = (
        f"Birleşik modelin test AP'si metadata baseline'ından **{direction}** "
        f"(fark {fusion['delta']:+.6f}; hasta bootstrap %95 aralığı "
        f"[{fusion['ci95_lower']:+.6f}, {fusion['ci95_upper']:+.6f}]). {uncertainty}"
    )
    table = [
        "| Model | Doğrulama AP | Test AP ↑ | Test ROC-AUC ↑ | Test pAUC ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in report["models"]:
        m = model["test"]
        table.append(
            f"| {LABELS[model['mode']]} | {model['validation']['average_precision']:.6f} | "
            f"{m['average_precision']:.6f} | {m['roc_auc']:.6f} | "
            f"{m['pauc_tpr_ge_0_8_unstandardized']:.6f} |"
        )
    lines = [
        "# Sabit modellerle bağımsız hasta testi",
        "",
        conclusion,
        "",
        (
            f"Test: **{report['test_rows']:,} lezyon, {report['test_patients']} hasta, "
            f"{report['test_positives']} pozitif lezyon** "
            f"({report['test_positive_patients']} pozitif hasta). "
            "Eğitim, doğrulama ve test hastaları birbirinden ayrıdır. Bu aynı SLICE-3D "
            "kaynağında ayrılmış bir iç testtir; başka merkez/veri kaynağından dış doğrulama değildir."
        ),
        "",
        *table,
        "",
        "## Değiştirilmeyen doğrulama eşikleri",
        "",
        "| Model | Sabit eşik | Test duyarlılığı | Test özgüllüğü | Precision | TP | FN | FP | TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in report["models"]:
        op = model["test"]["operating_point"]
        lines.append(
            f"| {LABELS[model['mode']]} | {op['threshold']:.12g} | "
            f"{op['sensitivity']:.4f} | {op['specificity']:.4f} | {op['precision']:.4f} | "
            f"{op['true_positive']} | {op['false_negative']} | "
            f"{op['false_positive']:,} | {op['true_negative']:,} |"
        )
    lines += [
        "",
        (
            "Eşikler testte yeniden seçilmedi. Doğrulamada hedeflenen en az %95 duyarlılık "
            "testte garanti edilmez. Modellerin test duyarlılıkları farklıysa yanlış pozitif "
            "sayıları eşit duyarlılık karşılaştırması olarak yorumlanmamalıdır."
        ),
        "",
        "## Metadata baseline'a göre eşleştirilmiş farklar",
        "",
        "| Model | Metrik | Test farkı | Hasta bootstrap %95 aralığı |",
        "| --- | --- | ---: | --- |",
    ]
    for model, difference in zip(
        report["models"][1:], report["bootstrap"]["differences"], strict=True
    ):
        for metric in ["average_precision", "roc_auc", "pauc_tpr_ge_0_8_unstandardized"]:
            d = difference[metric]
            lines.append(
                f"| {LABELS[model['mode']]} | {metric} | {d['delta']:+.6f} | "
                f"[{d['ci95_lower']:+.6f}, {d['ci95_upper']:+.6f}] |"
            )
    extraction = json.loads((root / "reports/heldout_test_extraction.json").read_text())
    lines += [
        "",
        "## Deney kaydı ve sınırlar",
        "",
        (
            "- Birincil karşılaştırma önceden sabitlendi: birleşik model eksi metadata AP. "
            "Diğer metrikler ikincil; çoklu karşılaştırma düzeltmesi yapılmadı."
        ),
        (
            "- Model, ön işleme ve doğrulama raporlarının SHA-256 kayıtları testten önce "
            "donduruldu; tahminden önce ve sonra değişmedikleri doğrulandı. "
            "Testte eğitim, erken durdurma, kalibrasyon veya eşik optimizasyonu yapılmadı."
        ),
        (
            "- Gemma dondurulmuş encoder: aynı revision, 70 görüntü token'ı, 768 boyut, "
            "batch=16 ve aynı runtime. XGBoost başlıkları 298.560 eğitim lezyonuyla daha önce eğitildi."
        ),
        (
            f"- {report['bootstrap']['valid_iterations']} geçerli eşleştirilmiş hasta bootstrap "
            "tekrarı, seed=42. Her hastanın tüm lezyonları birlikte örneklenir. Aralıklar sabit "
            "eğitilmiş modellere koşulludur; yeniden eğitim ve başka merkezlere genelleme "
            "belirsizliğini kapsamaz."
        ),
        (
            "- AP prevalansa bağlıdır. Doğrulama ve test AP farkı tek başına modelin "
            "iyileştiğini göstermez; esas karşılaştırma aynı test grubundaki modeller arasındadır."
        ),
        "- Olasılıklar kalibre edilmedi; araştırma sonucu klinik kullanım doğrulaması değildir.",
        (
            f"- Test embedding çıkarımı: {extraction['rows_done']:,} görüntü, "
            f"{extraction['extraction_seconds'] / 60:.2f} dakika (model yüklemesi hariç)."
        ),
        (
            "- Bu test artık görülmüştür. Sonraki geliştirmeler için yeniden ayar yapılıp aynı "
            "testte bağımsız doğrulama iddiasında bulunulmamalıdır."
        ),
        "",
        (
            "[Sabit protokol](heldout_test_protocol.json) · [Tüm metrikler](heldout_test_results.json) · "
            "[Çıkarım kaydı](heldout_test_extraction.json) · "
            "[Çalıştırma adımları](../docs/HELDOUT_TEST.md) · "
            "[Önceki doğrulama deneyi](full_experiment_summary.md)"
        ),
        "",
    ]
    (root / "reports/heldout_test_summary.md").write_text("\n".join(lines))
    readme_path = root / "README.md"
    readme = readme_path.read_text()
    start = readme.index("## Mevcut durum")
    end = readme.index("## Kurulum", start)
    status = [
        "## Mevcut durum",
        "",
        (
            "Sabit modeller ve doğrulama eşikleriyle hasta bazında "
            "ayrılmış test tamamlandı. Testte yeniden eğitim veya eşik ayarı yapılmadı."
        ),
        "",
        conclusion,
        "",
        *table,
        "",
        (
            f"Test: {report['test_rows']:,} lezyon / {report['test_patients']} hasta / "
            f"{report['test_positives']} pozitif. Aynı veri kaynağından iç test; dış doğrulama değildir."
        ),
        "",
        (
            "[Test özeti ve sabit eşik sonuçları](reports/heldout_test_summary.md) · "
            "[Test protokolü](reports/heldout_test_protocol.json) · "
            "[Test komutları](docs/HELDOUT_TEST.md) · "
            "[Önceki tam veri doğrulaması](reports/full_experiment_summary.md) · "
            "[Önceki pilot](reports/embedding_comparison.md)"
        ),
        "",
        "",
    ]
    readme = readme[:start] + "\n".join(status) + readme[end:]
    readme = readme.replace(
        "Test bölümünde performans hesaplanmaz.",
        "Eğitim komutu test performansı hesaplamaz; sabit modellerin test değerlendirmesi "
        "ayrı `evaluate_test` komutuyla yapılır.",
    )
    readme = readme.replace(
        "Sonraki hedef: tam eğitim grubunda embedding deneyi, yüksek görüntü token "
        "bütçesi ve hastaya özgü bağlamın ek katkısı.",
        "Tam eğitim ve sabit test karşılaştırmaları tamamlandı. Sonraki araştırmalar: yüksek "
        "görüntü token bütçesi ve hastaya özgü bağlam; yeni bağımsız değerlendirme planı gerektirir.",
    )
    readme_path.write_text(readme)
    print(conclusion)


if __name__ == "__main__":
    render()
