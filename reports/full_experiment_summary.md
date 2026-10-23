# Tam veri deneyi — sonuç özeti

Tam veri deneyinde birleşik modelin ana metriği AP, metadata baseline'ına göre **yükseldi** (fark +0.014232). Eşleştirilmiş hasta bootstrap %95 aralığı [-0.026619, +0.060532]. Aralık sıfırı içeriyor; farkın yönü konusunda belirsizlik var.

| Model | Eğitim lezyonu | AP ↑ | ROC-AUC ↑ | pAUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| Metadata + morfometri (baseline tekrar) | 298,560 | 0.061105 | 0.935891 | 0.148805 |
| Yalnız görüntü embedding'i | 298,560 | 0.039573 | 0.900879 | 0.128966 |
| Embedding + metadata/morfometri | 298,560 | 0.075336 | 0.947379 | 0.159063 |

Üç model de aynı 298,560 eğitim lezyonu ve aynı 53,629 doğrulama lezyonuyla çalıştı. Doğrulamada 57 pozitif var. Metadata baseline'ın tahminleri yeniden üretildi; kimlikler, sıralar, etiketler, ayarlar ve paket sürümleri eşleşti.

Gemma encoder donduruldu; bütün eğitim verisiyle XGBoost başlıkları eğitildi. Görüntü bütçesi 70 token, çıktı 768 boyut. Sonuçlar bu temsil ve sabit XGBoost ayarları için geçerlidir; Gemma fine-tuning deneyi değildir.

## Yüksek duyarlılıkta sonuç

| Model | Duyarlılık | Yanlış pozitif | Yanlış negatif |
| --- | ---: | ---: | ---: |
| Metadata + morfometri (baseline tekrar) | 0.9649 | 18,281 | 2 |
| Yalnız görüntü embedding'i | 0.9649 | 25,609 | 2 |
| Embedding + metadata/morfometri | 0.9649 | 15,282 | 2 |

Eşikler doğrulamada ayrı seçildi; bu eşik sonuçları bağımsız test değildir.

## Hesaplama

63,900 pilot vektörü yeniden kullanıldı; 288,289 yeni görüntü işlendi. Tam cache 352,189 vektör içeriyor. Yeni tamamlanan parçaların toplam çıkarım süresi 309.33 dakika; bu ölçüm model yüklemelerini ve yeniden başlatmada yarıda kalan parçayı içermez. Pilotun önceki çıkarımı dahil toplam ölçülen hesaplama 350.08 dakika.

| Başlık | Eğitim/kayıt süresi (s) | En iyi tur (0 tabanlı) |
| --- | ---: | ---: |
| metadata-full-control-v1 | 1.35 | 62 |
| image-eg2-70-full-v1 | 21.65 | 35 |
| fusion-eg2-70-full-v1 | 78.24 | 276 |

## Yorumlama sınırı

Bootstrap sabit eğitilmiş modellere koşulludur; doğrulama üzerinden erken durdurma ve model seçiminin belirsizliğini kapsamaz. Bağımsız test veya klinik üstünlük kanıtı değildir. Test bölümü kullanılmadı ve olasılıklar kalibre edilmedi.

[Detaylı metrikler](embedding_full_comparison.md) · [Hasta bootstrap ve eşleşme denetimi](full_comparison_audit.md) · [Çıkarım kaydı](embeddinggemma2-70-full-extraction.json) · [Tekrar çalıştırma](../docs/FULL_DATA_EXPERIMENT.md)
