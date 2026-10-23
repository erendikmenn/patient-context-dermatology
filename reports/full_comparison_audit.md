# Tam veri karşılaştırması: eşleşme ve belirsizlik

Metadata baseline aynı tahminlerle yeniden üretildi. Eğitim ve doğrulama kimlikleri, sıraları, etiketleri, XGBoost ayarları ve paket sürümleri eşleşiyor. Test kullanılmadı.

Aşağıdaki farklar metadata kontrolüne göredir; pozitif fark daha yüksek metrik demektir.

| Model | Metrik | Fark | Hasta bootstrap %95 aralığı |
| --- | --- | ---: | --- |
| image-eg2-70-full-v1 | average_precision | -0.021531 | [-0.076368, +0.011324] |
| image-eg2-70-full-v1 | roc_auc | -0.035013 | [-0.087589, +0.006300] |
| image-eg2-70-full-v1 | pauc_tpr_ge_0_8_unstandardized | -0.019839 | [-0.061710, +0.015661] |
| fusion-eg2-70-full-v1 | average_precision | +0.014232 | [-0.026619, +0.060532] |
| fusion-eg2-70-full-v1 | roc_auc | +0.011488 | [-0.007836, +0.031900] |
| fusion-eg2-70-full-v1 | pauc_tpr_ge_0_8_unstandardized | +0.010258 | [-0.006602, +0.027469] |

1000 geçerli tekrar; seed=42; 156 hasta. Her tekrar hasta bazında örneklenir; hastanın tüm lezyonlarına aynı ağırlık verilir ve modeller aynı örneklemde karşılaştırılır.

Aralıklar sabit eğitilmiş modellere koşulludur. Aynı doğrulamada erken durdurma/model seçimi yapıldığı için seçim belirsizliğini kapsamaz; bağımsız test veya klinik üstünlük kanıtı değildir. Sıfırı içeren aralık yön konusunda belirsizlik gösterir.
