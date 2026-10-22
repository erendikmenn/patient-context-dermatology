# EmbeddingGemma 2 model karşılaştırması

Bütün sonuçlar aynı hasta bazlı doğrulama grubundadır; test değerlendirilmedi.

| Model | Eğitim satırı | Özellik | AP | ROC-AUC | pAUC (0–0.20) | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-v1 | 298,560 | 38 | 0.061105 | 0.935891 | 0.148805 | 0.003649 |
| metadata-pilot-v1 | 10,271 | 38 | 0.059902 | 0.930258 | 0.143265 | 0.024145 |
| image-eg2-70-pilot-v1 | 10,271 | 768 | 0.042465 | 0.911317 | 0.135757 | 0.015281 |
| fusion-eg2-70-pilot-v1 | 10,271 | 806 | 0.054851 | 0.950578 | 0.160578 | 0.009321 |

## Doğrulamada seçilen yüksek duyarlılık eşikleri

| Model | Duyarlılık | Özgüllük | Precision | Yanlış pozitif |
| --- | ---: | ---: | ---: | ---: |
| baseline-v1 | 0.9649 | 0.6588 | 0.0030 | 18,281 |
| metadata-pilot-v1 | 0.9649 | 0.5776 | 0.0024 | 22,631 |
| image-eg2-70-pilot-v1 | 0.9649 | 0.5485 | 0.0023 | 24,190 |
| fusion-eg2-70-pilot-v1 | 0.9649 | 0.6482 | 0.0029 | 18,846 |

## Yorumlama koşulları

Encoder donduruldu; XGBoost başlıkları eğitildi. Aynı temel XGBoost ayarları kullanıldı, erken durdurma her model için doğrulamada seçildi. Çok boyutlu embedding'in en iyi olası başlığı bulunduğu iddia edilmez. Eşikler aynı doğrulamada seçildiği için eşik metrikleri bağımsız test performansı değildir. Olasılıklar kalibre edilmedi.

Eğitim satırı sayıları farklıdır. Daha küçük eğitim örneği kullanan modelin farkı yalnızca embedding'e bağlanamaz; tablo bu ayrımı açıkça gösterir.
