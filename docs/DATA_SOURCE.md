# Veri kaynağı ve kullanım

Kaynak: [ISIC Challenge Data](https://challenge.isic-archive.com/data/).

Atıf: International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. International Skin Imaging Collaboration. https://doi.org/10.34970/2024-slice-3d (2024).

Veri lisansı: [Creative Commons Attribution Non Commercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Atıf gerekir; bu tam veri seti ticari kullanım için seçilmemiştir. Depo kaynak kodu ile veri kullanım koşulları birbirinden ayrıdır. Ham veri ve örnek bazındaki çıktılar bu Git deposuna eklenmez. Kaynak sayfada ayrı bir CC-BY permissive alt küme de bulunur; mevcut deney tam kümededir.

Katkıda bulunan kurumlar: Hospital Clínic de Barcelona, Memorial Sloan Kettering Cancer Center, Hospital of Basel, FNQH Cairns, The University of Queensland, Melanoma Institute Australia, Monash University and Alfred Health, University of Athens Medical School, Medical University of Vienna.

## İndirilecek dosyalar

- `ISIC_2024_Training_Input.zip`: JPEG görüntüler ve temel metadata.
- `ISIC_2024_Training_Supplement.csv`: ek metadata; öncelikle veri denetimi ve merkez değerlendirmesi için.
- `ISIC_2024_Training_GroundTruth.csv`: malignite etiketleri.

İndirme kodu resmî ISIC S3 bağlantılarını kullanır. Kaggle kimlik bilgisi gerekmez. Dosya hash kayıtları `data/raw/download_manifest.json` içine yazılır. Hash'ler tekrar indirilen/veri hazırlanan yerel dosyaların bütünlüğünü izler; yayıncının imzaladığı checksum değildir.

## Sınırlar

Görüntüler 3D total body photography'den alınmış standart lezyon kesitleridir. Sıradan telefon fotoğrafı performansına doğrudan genellenemez. Benign örneklerin bir kısmı biyopsi yapılmadan klinisyen değerlendirmesiyle etiketlenmiştir; malign etiketlerle aynı doğrulama sürecine sahip oldukları varsayılmaz. Örnek kimliği veya kaynak kurum tahmin özelliği değildir. Morfometri alanları görüntüden türetilmiş ölçümlerdir ve yeni bir görüntüde otomatik olarak mevcut olmayabilir.

Bu nedenle yalnız görsel yükleyen bir uygulamada metadata baseline'ın bütün özelliklerini sağlamak için ayrıca ölçüm çıkarımı gerekir; baseline bugün veri setindeki hazır ölçümleri kullanır.
