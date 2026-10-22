# Patient Context Dermatology

SLICE-3D lezyon görüntüleri, hasta bilgileri ve aynı hastanın diğer lezyonlarıyla malignite sınıflandırması araştırması. İlk aşama, hasta bazında ayrılmış gerçek veri üzerinde metadata ve görüntüden ölçülmüş morfometri özellikleriyle XGBoost baseline'ıdır.

EmbeddingGemma 2 sonraki görüntü aşamasının encoder adayıdır; ilk tabular eğitimde kullanılmaz. Hasta bağlamı, vaka erişimi ve attention modeli sonraki kilometre taşlarıdır. Ayrıntılar: [yol haritası](docs/ROADMAP.md), [veri kaynağı](docs/DATA_SOURCE.md), [veri gereksinimleri](docs/DATA_REQUIREMENTS.md).

## Mevcut durum

23 Ekim 2026: resmî veri indirildi, 401.059 lezyon ve 1.042 hasta üzerinde hazırlık tamamlandı. İlk XGBoost modeli 298.560 eğitim lezyonunda eğitildi. 53.629 lezyonluk doğrulamada ROC-AUC 0,9359 ve AP 0,0611; test değerlendirilmedi. Yüksek duyarlılıkta yanlış pozitif sayısı yüksek; bu sonuç ilk araştırma baseline'ıdır. Veri sızıntısı ve model kaydetme/yükleme kontrollerini içeren 12 test geçti.

[Veri durum raporu](reports/data_status.md) · [İlk model sonuçları](reports/baseline-v1.md)

## Kurulum

Python 3.12 önerilir. Paket sürümleri `uv.lock` ile sabitlenir. CPU eğitimi yeterlidir; bu aşamada GPU veya API anahtarı gerekmez.

```bash
uv sync --python 3.12 --group dev --locked
uv run derm --help
```

## Veriden modele

Komutları bu repo kökünden sırayla çalıştırın:

```bash
uv run derm download
uv run derm prepare
uv run derm train --config configs/baseline.json --run-name baseline-v1
```

İndirme doğrulanmış mevcut dosyaları tekrar indirmez. `prepare` mevcut bölümlemenin üzerine yazmaz. `train` aynı adlı çalışmanın üzerine yazmaz. Hazır veri ve model varsa yeniden eğitim için farklı `--run-name` seçin.

Hazırlık, resmî dosyaların SHA-256'sını doğrular; görüntü ve etiket kimliklerini eşleştirir; bütün ZIP üyelerine CRC kontrolü, 512 görüntüye piksel decode kontrolü uygular. Hasta bazında %70/%15/%15 ayrımı sabitlenir. Görüntüler ZIP'te kalır, dosya yolu yerine `image_member` ile okunur.

Eğitimde kategorik dönüşümler ve eksik değer işlemleri yalnızca eğitim hastalarında öğrenilir. Erken durdurma ve araştırma eşiği doğrulamada seçilir. Test bölümünde performans hesaplanmaz. AP, ROC-AUC, yüksek duyarlılık bölgesindeki pAUC, Brier ve sabit prevalans referansı raporlanır. `pAUC` normalize edilmemiş 0–0.20 aralığındadır; ROC-AUC ile karıştırılmamalıdır. AP, sklearn `average_precision_score` tanımıdır.

## Çıktılar

```text
configs/baseline.json                 Deney ayarları
data/raw/                            Resmî ZIP ve CSV dosyaları, indirme manifesti
data/processed/metadata.parquet       Temiz özellikler, etiketler, image_member ve split
data/processed/splits.csv             Sabit hasta ayrımı
data/processed/manifest.json          Hazırlık ve checksum kayıtları
artifacts/baseline-v1/model.json       Eğitilmiş XGBoost
artifacts/baseline-v1/preprocessor.joblib
artifacts/baseline-v1/run.json         Ayarlar, sürümler ve metrikler
artifacts/baseline-v1/validation_predictions.parquet
reports/data_audit.json               Toplu veri denetimi
reports/baseline-v1.md                İnsan tarafından okunabilir ilk sonuçlar
reports/baseline-v1.json              Makine tarafından okunabilir ilk sonuçlar
```

Ham veriler, model ağırlıkları, embedding'ler ve örnek bazında tahminler `.gitignore` kapsamındadır. Repo kaynak kodu, ayarlar, toplu raporlar ve dokümantasyon içerir. GitHub uzak deposu ayrıca bağlanabilir; yerel repo kurulumu otomatik yayın yapmaz.

## Tahmin

`input.csv`, seçilen özellik grubundaki kolonları taşımalıdır; etiket gerekmez. Çıktı kalibrasyon uygulanmamış `score` alanıdır.

```bash
uv run derm predict --input data/input.csv --model-dir artifacts/baseline-v1 --output artifacts/new_predictions.csv
```

`morphometry` modelinde renk, sınır ve şekil ölçümleri gerekir. Sadece bir telefon fotoğrafı bu komuta verilemez. Yeni görüntü kullanım akışı, encoder aşamasında ayrıca kurulacaktır.

## Özellik sözleşmesi

`clinical`: yaş, cinsiyet, anatomik bölge, lezyon boyutu. `morphometry`: bunlara ek görüntüden elde edilmiş renk ve şekil ölçümleri. Hasta kimliği, lezyon kimliği, kesin tanı, patoloji, kaynak kurum ve önceden eğitilmiş başka sınıflandırıcıların güven skorları girdiye alınmaz. İzin listesi `src/derm_context/features.py` içindedir.

## Doğrulama

```bash
uv run pytest -q
uv run ruff check src tests
```

Testler hasta sızıntısını, hatalı etiket/kimlikleri, yasak alanların dışlanmasını, görülmemiş kategorileri, pAUC tanımını ve model kaydetme/yükleme/tahmin tutarlılığını denetler. Sentetik test verileri yalnızca yazılım doğrulamasında kullanılır, gerçek başarı raporuna katılmaz.

## Kullanım kapsamı

Araştırma prototipi. Bu tam SLICE-3D sürümü CC-BY-NC 4.0 lisanslıdır. Morfometri ölçümleri, görüntü dağılımı ve etiket doğrulama biçimi için [veri belgesini](docs/DATA_SOURCE.md) okuyun. Sonraki hedef: aynı sabit hasta ayrımında görüntü embedding baseline'ı ve hastaya özgü bağlamın ek katkısı.
