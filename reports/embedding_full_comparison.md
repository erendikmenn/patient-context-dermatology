# EmbeddingGemma 2 model karşılaştırması

Bütün sonuçlar aynı hasta bazlı doğrulama grubundadır; test değerlendirilmedi.

| Model | Eğitim satırı | Özellik | AP | ROC-AUC | pAUC (0–0.20) | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-v1 | 298,560 | 38 | 0.061105 | 0.935891 | 0.148805 | 0.003649 |
| metadata-full-control-v1 | 298,560 | 38 | 0.061105 | 0.935891 | 0.148805 | 0.003649 |
| image-eg2-70-full-v1 | 298,560 | 768 | 0.039573 | 0.900879 | 0.128966 | 0.003848 |
| fusion-eg2-70-full-v1 | 298,560 | 806 | 0.075336 | 0.947379 | 0.159063 | 0.001780 |

## Doğrulamada seçilen yüksek duyarlılık eşikleri

| Model | Duyarlılık | Özgüllük | Precision | Yanlış pozitif |
| --- | ---: | ---: | ---: | ---: |
| baseline-v1 | 0.9649 | 0.6588 | 0.0030 | 18,281 |
| metadata-full-control-v1 | 0.9649 | 0.6588 | 0.0030 | 18,281 |
| image-eg2-70-full-v1 | 0.9649 | 0.5220 | 0.0021 | 25,609 |
| fusion-eg2-70-full-v1 | 0.9649 | 0.7147 | 0.0036 | 15,282 |

## Yorumlama koşulları

Encoder donduruldu; XGBoost başlıkları eğitildi. Aynı temel XGBoost ayarları kullanıldı, erken durdurma her model için doğrulamada seçildi. Çok boyutlu embedding'in en iyi olası başlığı bulunduğu iddia edilmez. Eşikler aynı doğrulamada seçildiği için eşik metrikleri bağımsız test performansı değildir. Olasılıklar kalibre edilmedi.

Karşılaştırılan modellerin eğitim satırı sayıları eşittir. Embedding modelleri tüm özgün eğitim ve doğrulama satırlarını kullanıyorsa rapor JSON'larında `all_original_training_rows=true` ve `all_original_validation_rows=true` bulunur.
