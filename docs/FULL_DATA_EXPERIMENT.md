# Tam eğitim verisiyle EmbeddingGemma 2 karşılaştırması

Bu deney pilotun eğitim örnekleme farkını kaldırır. Bütün modeller **298.560 lezyon, 729 hasta ve 271 pozitif** içeren aynı eğitim grubunu ve **53.629 lezyon, 156 hasta ve 57 pozitif** içeren aynı doğrulama grubunu kullanır. 48.870 lezyonluk test bölümü kullanılmaz.

## Sabit tutulan koşullar

- Veri ve hasta bölme hash'leri; eğitim/doğrulama kimlikleri, etiketleri ve sırası.
- `configs/baseline.json` içindeki XGBoost ayarları: seed=42, derinlik=4, öğrenme oranı=0,04, en fazla 1.000 tur, 60 tur erken durdurma, sınıf ağırlığı sqrt(negatif/pozitif).
- Erken durdurma metriği XGBoost `aucpr`; raporlanan AP sklearn `average_precision_score` tanımıdır. Aynı kavramın birebir aynı sayısal hesabı değildir.
- Aynı paket sürümleri, CPU histogram algoritması ve 8 iş parçacığı. En iyi iterasyon her modelin doğrulamasında ayrı seçilir.
- Görüntü modellerinde aynı dondurulmuş EmbeddingGemma 2 revizyonu, 70 görüntü token bütçesi, 768 boyut, bfloat16 MPS, batch=16 ve L2 normalizasyonu.

Yalnızca özellik grubu değişir: 38 metadata/morfometri özelliği, 768 görüntü özelliği veya bunların 806 boyutlu birleşimi. Gemma ağırlıklarında fine-tuning yapılmaz. Kullanıcının tam veri deneyi, bu görüntü temsili üzerinde XGBoost başlıklarını bütün eğitim örnekleriyle eğitmektir.

## Vektörlerin yeniden kullanılması

Pilotun tamamlanmış 63.900 vektörü SHA-256, model ayarları, veri/bölme kimlikleri ve görüntü kimlikleriyle doğrulanarak yeni cache'e kopyalanır. Yeniden hesaplanacak 288.289 görüntü yalnızca eğitim grubundadır; pilot zaten bütün validation görüntülerini içerir.

Yeni cache 352.189 satırlıdır ve `isic_id` sırasındadır. Vektörler kimlikle eşleştirilir; satır konumları tahmin edilmez. Kaynak cache değiştirilmez. Yeni çıkarım her 512 görüntüde kaydedilir; kesilirse aynı komut devam eder. Sonuç dosyası yaklaşık 1,08 GB olacaktır. Test görüntüleri dahil edilmez.

## Tekrar çalıştırma

Pilot cache varsa:

```bash
uv sync --python 3.12 --extra embeddings --group dev --locked
uv run --extra embeddings python -m derm_context.expand_embeddings \
  --reuse-dir data/embeddings/embeddinggemma2-70-pilot \
  --output-dir data/embeddings/embeddinggemma2-70-full
```

Pilot cache yoksa README'deki download/prepare işlemlerinden sonra tam çıkarımı doğrudan yapabilirsiniz. Bu seçenek bütün 352.189 görüntüyü hesaplar:

```bash
uv run --extra embeddings python -m derm_context.embeddings extract \
  --config configs/embeddinggemma2-70.json --batch-size 16 \
  --embedding-dir data/embeddings/embeddinggemma2-70-full
```

Cache tamamlandıktan sonra:

```bash
uv run --extra embeddings python -m derm_context.train_embeddings \
  --mode metadata --run-name metadata-full-control-v1 \
  --cache-dir data/embeddings/embeddinggemma2-70-full
uv run --extra embeddings python -m derm_context.train_embeddings \
  --mode image --run-name image-eg2-70-full-v1 \
  --cache-dir data/embeddings/embeddinggemma2-70-full
uv run --extra embeddings python -m derm_context.train_embeddings \
  --mode fusion --run-name fusion-eg2-70-full-v1 \
  --cache-dir data/embeddings/embeddinggemma2-70-full
uv run --extra embeddings python -m derm_context.compare \
  baseline-v1 metadata-full-control-v1 image-eg2-70-full-v1 fusion-eg2-70-full-v1 \
  --output-name embedding_full_comparison.md
uv run --extra embeddings python -m derm_context.evaluate_full --iterations 1000
```

`evaluate_full` için önce mevcut baseline'ın README'deki komutla üretilmiş yerel model/tahmin dosyaları gerekir. Sonuç adları mevcutsa modellerin üzerine yazılmaz. Eğitim ve görüntü çıkarımı ayrı süreçlerdedir; macOS'ta XGBoost ile encoder'ın native runtime'larını aynı sürece yüklemeyin.

## Sonucu nasıl değerlendireceğiz?

Ana ölçü AP; ek ölçüler ROC-AUC, yüksek duyarlılık pAUC ve doğrulamada seçilen yüksek duyarlılık eşiğinde yanlış pozitif sayısı. İlk olarak tam metadata kontrolünün önceki baseline'ın tahminlerini yeniden üretmesi zorunludur. Daha sonra görüntü ve birleşik modellerle fark hesaplanır.

1.000 tekrarlı eşleştirilmiş hasta bootstrap'ı AP/ROC-AUC/pAUC farklarının %95 percentile aralıklarını verir. Hastanın tüm lezyonları birlikte örneklenir; modeller aynı örneklemler üzerinde karşılaştırılır. Aralıklar eğitilmiş sabit modellere koşulludur ve doğrulama üzerinden model seçimi belirsizliğini içermez. Bu yüzden bağımsız test kanıtı sayılmaz. Sıfırı içeren aralıklar farkın yönünün belirsiz olduğunu gösterir.

Kod ve toplu sonuçlar GitHub'a eklenir; ham görüntüler, hasta/lezyon kimlikleri, vektörler ve eğitilmiş ağırlıklar yerelde kalır. Veri lisansı CC-BY-NC-4.0; repo kodu MIT'dir.

## Çalıştırma durumu

Tamamlandı. Tam veri deneyinde birleşik modelin ana metriği AP, metadata baseline'ına göre **yükseldi** (fark +0.014232). Eşleştirilmiş hasta bootstrap %95 aralığı [-0.026619, +0.060532]. Aralık sıfırı içeriyor; farkın yönü konusunda belirsizlik var.

| Model | Eğitim lezyonu | AP ↑ | ROC-AUC ↑ | pAUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| Metadata + morfometri (baseline tekrar) | 298,560 | 0.061105 | 0.935891 | 0.148805 |
| Yalnız görüntü embedding'i | 298,560 | 0.039573 | 0.900879 | 0.128966 |
| Embedding + metadata/morfometri | 298,560 | 0.075336 | 0.947379 | 0.159063 |

[Sonuç özeti](../reports/full_experiment_summary.md) · [Metrikler](../reports/embedding_full_comparison.md) · [Belirsizlik ve eşleşme denetimi](../reports/full_comparison_audit.md)
