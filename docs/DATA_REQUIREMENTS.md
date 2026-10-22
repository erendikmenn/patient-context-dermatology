# Hangi verilere ihtiyacımız var

## İlk model için gerekli veriler

| Veri | Kullanım | SLICE-3D durumu |
| --- | --- | --- |
| `isic_id` | Görüntü, özellik ve etiket eşleştirmesi | Resmî dosyalarda var |
| `patient_id` | Hastaları eğitim ve test arasında ayırmak | Temel metadatada var; model özelliği yapılmaz |
| Malignite etiketi | Gözetimli öğrenme hedefi | GroundTruth dosyasında `malignant`; projede `target` |
| Yaş, cinsiyet, anatomik bölge | Klinik özellikler | Metadatada; bazı değerler eksik olabilir |
| Lezyon boyutu | Boyut bilgisi | Metadatada hazır ölçüm |
| Renk ve şekil ölçümleri | Morfometri baseline | TBP Lesion Visualizer alanlarında hazır |
| Kaynak ve lisans | Veri denetimi ve ileride merkez bazlı değerlendirme | Ek metadatada; modele verilmez |

İlk eğitim için yeni hasta toplamak, hastane entegrasyonu, metin notu veya API anahtarı gerekmiyor. Morfometri değerleri fotoğrafın kendisi değildir: bir uygulamada bu modelin aynı girdileri alması için ölçümlerin ayrıca üretilmesi gerekir.

## Görüntü modeli için ek gereksinimler

JPEG görüntülerin her biri `isic_id` ile eşleşmeli. ZIP bütünlüğü kontrol edilir; encoder çalışırken her görüntü decode edilir, hatalı görüntü sessizce atılmaz. Doğru RGB dönüşümü ve modelin kendi görüntü ön işleyicisi kullanılır. Embedding önbelleğinde kimlik, model revizyonu, işlemci ayarları ve vektör boyutu tutulur.

EmbeddingGemma 2 ağırlıkları indirildi; sabit revizyonda CPU ve Apple GPU ile gerçek görüntü çıkarımı doğrulandı. İlk pilot 70 görüntü token bütçesi ve 768 boyutlu normalize vektör kullanır. Ayrıntılar ve tekrar çalıştırma komutları [embedding deneyi](EMBEDDING_EXPERIMENT.md) belgesindedir. 401.059 × 768 adet float32 vektör, yalnızca vektörler için yaklaşık 1,23 GB yer tutar; 256 boyutta yaklaşık 0,41 GB. Bu hesap indeks/metadata/model belleğini içermez.

## Hastaya özgü bağlam için gereksinimler

Bir hastanın birden fazla lezyon görüntüsü gerekir. Ayrıca bu görüntülerin tahmin anında birlikte erişilebilir olduğu varsayımını doğrulamak gerekir. `patient_id`, tek başına aynı muayene veya çekim zamanı anlamına gelmez. Ziyaret/çekim zamanları doğrulanamıyorsa ilk bağlam deneyi “veri setindeki çoklu lezyon bağlamı” olarak raporlanır; prospektif aynı-ziyaret performansı iddia edilmez.

İleri gerçek kullanım için hasta kimliğine ek `visit_id`, çekim zamanı, aynı lezyonun tekrar görüntülerini ayıracak `lesion_id` ve görüntü çekim protokolü gerekir. Tek görüntülü hastalarda bağlamın eksik olduğu açık bir özellik olarak temsil edilir. Etiketlerden hastaya referans üretilemez.

## Kullanılmayacak alanlar

Kesin tanı `iddx_*`, melanom kalınlığı, mitotik indeks ve sonuca bağlı başka alanlar girdiye alınmaz. Ek metadatanın tamamı otomatik özellik listesine çevrilmez. İlk baselineda haricî ağların `tbp_lv_nevi_confidence` veya `tbp_lv_dnn_lesion_confidence` skorları da dışarıda kalır. Böylece bu ağların katkısını kendimizin eğittiği modele mal etmeyiz.

## İlk eğitimden sonra ek veri ihtiyacı

Embedding ve bağlam deneyleri için mevcut veri yeterli bir araştırma başlangıcıdır. Klinik genelleme için farklı kurum/cihaz ve cilt görünümü dağılımlarında bağımsız veri gerekir. Bu setten görülmemiş telefon çekimlerine güvenilirlik çıkarılamaz. Ayrı bir merkez testi ve daha sonra gerçek dış doğrulama, model geliştirmeden ayrı değerlendirme aşamalarıdır.
