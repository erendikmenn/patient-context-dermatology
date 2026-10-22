# Veri hazırlama durumu

23 Ekim 2026 tarihinde resmî ISIC kaynağından indirilen tam SLICE-3D verisiyle hazırlık tamamlandı.

Toplam **401,059 lezyon**, **1,042 hasta**, **393 malign lezyon** ve en az bir malign lezyonu olan **259 hasta** var. Malign lezyon oranı **%0.0980**.

| Bölüm | Hasta | Lezyon | Malign lezyon | Malign lezyonu olan hasta |
| --- | ---: | ---: | ---: | ---: |
| train | 729 | 298,560 | 271 | 181 |
| validation | 156 | 53,629 | 57 | 39 |
| test | 157 | 48,870 | 65 | 39 |

Hasta kümeleri arasında çakışma **0**. %70/%15/%15 hedefi hastalara uygulanır; hasta başına lezyon sayıları farklı olduğu için satır oranları aynı değildir. Test sayıları yalnızca veri envanteridir; test performansı hesaplanmadı.

## Bütünlük kontrolleri

- Üç resmî dosyanın URL, boyut ve SHA-256 kayıtları var.
- Görüntü, metadata, ek metadata ve etiket kimlikleri birebir eşleşiyor.
- Bütün ZIP üyeleri CRC kontrolünden geçti.
- Rastgele seçilen 512 görüntü piksel düzeyinde açıldı; bu örneklerde hata yok.
- Tüm 401.059 görüntü henüz piksel düzeyinde açılmadı; görüntü encoder aşamasında kontrol edilecek.
- Tanı, patoloji, hasta kimliği ve kaynak kurum öğrenme özelliklerinden çıkarıldı.

## Eksik girdiler

- Yaş: 2,798 eksik kayıt; eğitim bölümünde öğrenilen medyan ve eksiklik göstergesi kullanılır.
- Cinsiyet: 11,517 eksik kayıt; ayrı kategori.
- Anatomik bölge: 5,756 eksik kayıt; ayrı kategori.

## Yerel dosyalar

- `data/raw/input.zip`: 401.059 JPEG ve temel metadata, lisans ve atıf dosyaları.
- `data/raw/supplement.csv`: ek metadata.
- `data/raw/ground_truth.csv`: malignite etiketleri.
- `data/processed/metadata.parquet`: temiz veri ve sabit bölüm bilgisi.
- `data/processed/splits.csv`: tekrar kullanılacak hasta ayrımı.

## Sonraki veri ihtiyacı

İlk tabular ve görüntü deneyleri için yeni veri toplamaya ihtiyaç yok. Görüntü aşamasında model ağırlıkları ve embedding önbelleği hazırlanacak. Hasta bağlamı için aynı ziyarette erişilebilir lezyonların doğrulanması gerekir; mevcut patient_id tek başına ziyaret zamanını göstermez. Ayrıntılar: [veri gereksinimleri](../docs/DATA_REQUIREMENTS.md).
