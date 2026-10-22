# İlk model sonuçları — baseline-v1

Bu sonuçlar hasta bazında ayrılmış **doğrulama** grubundadır. Test grubu açılmadı.

Özellik grubu: `morphometry`. Model çıktıları henüz kalibre edilmemiştir; klinik risk olarak yorumlanmamalıdır.

| Metrik | XGBoost | Sabit prevalans tahmini |
| --- | ---: | ---: |
| Average precision (AP) | 0.061105 | 0.001063 |
| ROC-AUC | 0.935891 | 0.500000 |
| pAUC, TPR ≥ 0.80 (0–0.20) | 0.148805 | 0.020000 |
| Brier (düşük daha iyi) | 0.003649 | 0.001062 |

Doğrulama: 53,629 lezyon; 57 malign. Malignite prevalansı %0.1063.

En iyi ağaç turu (sıfırdan başlayan indeks): 62. Eğitim ve kayıt süresi: 0.8 saniye.

## Araştırma eşiği

Eşik doğrulamada seçildi ve aşağıdaki ölçümler aynı grupta yapıldı; bağımsız test performansı değildir.

Eşik: 0.017769; duyarlılık: %96.49; özgüllük: %65.88; precision: %0.30.
Doğru pozitif: 55; yanlış negatif: 2; yanlış pozitif: 18281; doğru negatif: 35291.

## Sınırlar

Bu tek sabit bölme üzerinde ilk geliştirme deneyidir. Hasta bazında güven aralıkları, merkezler arası doğrulama, kalibrasyon ve final test henüz yapılmadı. Morfometri özellikleri veri setinde hazır bulunan görüntü ölçümleridir. EmbeddingGemma 2 bu eğitimde kullanılmadı.
