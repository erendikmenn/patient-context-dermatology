# Dermatoloji projesi yol haritası

Başlangıç tarihi: 23 Ekim 2026. Hedef, 3D tüm vücut fotoğraflarından çıkarılmış bir deri lezyonunun malignite olasılığını, görüntü ve hasta bağlamını kullanarak tahmin etmek. Bu araştırma prototipinin kapsamı mevcut lezyon sınıflandırmasıdır; gelecekte kanser gelişimi veya telefon fotoğraflarında klinik kullanım iddiası taşımaz.

## 1 Veri edinme ve envanter

Kaynak: [ISIC 2024 resmî veri sayfası](https://challenge.isic-archive.com/data/). Tam SLICE-3D seti 401.059 JPEG, temel metadata, ek metadata ve malignite etiketlerini içerir. Tam set CC-BY-NC 4.0; ayrı 217.477 görüntülük permissive set CC-BY 4.0 lisanslıdır. İlk deney tam araştırma setinde yapılır; iki kümenin sonuçları birbirine karıştırılmaz.

Gerekli alanlar: `isic_id`, `patient_id`, ikili `target`; görüntü aşaması için eşleşen JPEG. İlk tabular model yaş, cinsiyet, anatomik bölge, lezyon boyutu ve görüntüden ölçülmüş renk/şekil özelliklerini kullanabilir. Bu nedenle geniş metadata modeli yalnızca klinik bilgilerden oluşmaz. Gerçek serbest metin klinik not bu veri setinde varsayılmaz.

İndirmeler kaynak URL, tarih, SHA-256 ve boyutla kayıt altına alınır. Ham arşiv korunur; görüntüler ilk aşamada ZIP içinden okunarak yüz binlerce dosya açma maliyeti önlenir. Etiket, metadata ve arşiv görüntü kimliklerinin birebir eşleşmesi denetlenir. Kabul: eksik/zıt etiket, yinelenen kimlik veya eksik hasta kimliği varsa hazırlık durur.

## 2 Veri sözleşmesi ve hasta ayrımı

Önce hasta bazında yaklaşık %70 eğitim, %15 doğrulama ve %15 test ayrımı yapılır. Stratifikasyon, hastanın en az bir malign lezyonu olup olmamasına dayanır; lezyon oranları hasta başına farklı görüntü sayıları nedeniyle tam eşit olmayabilir. Seed ve manifest sabitlenir. Aynı hastanın bütün lezyonları tek bölüme gider.

Özellikler açık izin listesinden seçilir. `target`, `iddx_*`, patoloji sonuçları, hasta/lesyon kimlikleri, merkez ve telif alanları öğrenme girdisi değildir. Haricî sınıflandırıcıların nevüs veya lezyon güven skorları ilk baselineda kullanılmaz. Kategorik dönüşümler yalnızca eğitim bölümünde öğrenilir; bilinmeyen kategoriler güvenle ele alınır. Test modeli seçmek veya erken durdurmak için kullanılmaz.

Kabul: hasta kümeleri ayrık, her bölümde iki sınıf mevcut, bütün satırlar tek bölüme atanmış, özellik sözleşmesi kayıtlı. Görüntülerin bütünlüğü ZIP CRC kontrolüyle; model aşamasından önce ayrıca decode kontrolüyle denetlenir.

## 3 İlk model eğitimi

Önce eğitim prevalansını veren sabit tahminleyici referansı hesaplanır. Ardından CPU üzerinde XGBoost histogram modeli eğitilir. `clinical` özellik grubu yaş, cinsiyet, bölge ve boyutu; `morphometry` grubu ek renk/şekil ölçümlerini içerir. İlk çalışma `morphometry` grubuyla yapılır; `clinical` ayrı deney olarak çalıştırılabilir.

Erken durdurma yalnızca doğrulama verisi kullanır. Dengesiz sınıflar için eğitimden hesaplanan sınıf ağırlığı kullanılabilir; bunun çıktıları kalibre edilmiş klinik risk diye sunulmaz. Hiperparametre, bağımlılık sürümleri, veri manifesti, model, ön işleyici ve doğrulama metrikleri saklanır.

Metrikler: average precision (AP), ROC-AUC, duyarlılık >= %80 bölgesinde normalize edilmemiş kısmi AUC, Brier skoru ve doğrulamada %95 duyarlılık için seçilen araştırma eşiğindeki özgüllük/precision/yanlış pozitifler. Eşik performansı aynı doğrulama grubunda seçildiği için iyimser olabilir; final doğrulama için bağımsız test gerekir. Test verisine ilk geliştirme aşamasında bakılmaz.

Kabul: gerçek veriyle tamamlanmış çalışma, yeniden yüklenen modelin aynı tahminleri vermesi, sabit tahminleyiciyle karşılaştırma ve sınıf sayılarının raporlanması. Sayısal performans için önceden başarı garantisi verilmez.

## 4 Görüntü embedding baseline

EmbeddingGemma 2 tokenizer değil görüntü kodlayıcısıdır. Donuk modelden her görüntü için 768 boyutlu vektör alınır; ardından 256 boyut alternatifi ayrıca ölçülür. Model kimliği/revizyonu, işlemci ayarları, görüntü hash bilgisi ve örnek sırası önbelleğe kaydedilir. Tüm veri ölçeğine çıkmadan önce küçük bir örnekle hız, bellek ve geçerli vektör kontrolü yapılır.

Deneyler: yalnız embedding üzerinde lineer başlık/MLP; embedding + metadata; alternatif encoder olarak MedSigLIP. Kodlayıcının seçimi sonuçla belirlenir. Başlangıçta model donuktur; XGBoost üzerinden encoder'a gradyan akmaz. Fine-tuning gerekiyorsa ayrı neural head ve eğitim akışı kullanılır. Apple GPU/CPU destek ve hız ölçümü bu aşamanın ilk işidir.

Kabul: bütün embedding kimlikleri metadata ile eşleşir, NaN/sonsuz değer yok, aynı sabit split kullanılır; metadata baseline'a katkı ayrı gösterilir.

## 5 Hastaya özgü bağlam

Hedef lezyon hariç aynı muayenede mevcut diğer lezyonlardan referans oluşturulur. Merkeze uzaklık, en yakın diğer lezyonlara uzaklık ve göreli boyut ilk özelliklerdir. Tek lezyonlu hastalar için eksik bağlam göstergesi tutulur. Test hastasının etiketleri asla kullanılmaz; diğer etiketsiz görüntülerinin mevcut olması bu deneyin kullanım koşuludur.

`patient_id` tek başına aynı ziyareti kanıtlamaz. Çekim/ziyaret bilgisi doğrulanamıyorsa bu aşama veri setindeki çoklu lezyon bağlamı deneyi olarak sınırlandırılır; aynı anda kullanılabilir görüntülere dayanan prospektif uygulama iddia edilmez. Ayrıntılar `DATA_REQUIREMENTS.md` içindedir.

İleri sürüm: hedef lezyonun diğer lezyonlara attention uyguladığı set modeli. Başlık, projeksiyon ve attention katmanları eğitilir. Küme büyüklüğü, rastgele lezyon örnekleme ve hasta başına ağırlıklandırmanın etkileri raporlanır.

## 6 Eğitim vakalarından bilgi getirme

Arama havuzu yalnızca eğitim hastalarını içerir. Komşu etiket istatistiği eğitim örneklerine verilecekse hasta bazında out-of-fold oluşturulur; kendisini veya aynı hastayı komşu olarak kullanmak yasaktır. Görüntü benzerliği ile metadata uyumu ayrı bileşenlerdir. Farklı encoder uzayları doğrudan karşılaştırılmaz.

Kontroller: erişimsiz model, gerçek komşu, rastgele komşu, etiket bilgisi olmadan yalnız uzaklık. Model değişirse indeks yeniden üretilir. Benzer vaka sunumu nedensel açıklama diye adlandırılmaz.

## 7 Son değerlendirme ve sunum

Model seçimi tamamlandıktan sonra sabit test bir kez açılır. Eşik ve gerekiyorsa kalibrasyon doğrulamada belirlenir. Hasta bazında bootstrap aralıkları, merkezlere göre performans ve yanlış pozitif analizi yapılır. Az sayıdaki malign hasta ve benign etiketlerin farklı doğrulama yöntemleri raporda açıklanır.

Arayüz araştırma girdisi, skor, benzer vakalar ve veri koşullarını gösterir. Kod ve raporlar Git'e; ham veri, bireysel tahminler, ağırlıklar, embedding ve çalışma ortamı Git dışına gider. GitHub yayını için bu yerel depo hazırlandıktan sonra uzak hesap/görünürlük belirlenir.

## Repo bileşenleri

`src/derm_context`: indirme, hazırlama, özellik sözleşmesi, eğitim, değerlendirme ve tahmin. `configs`: değiştirilebilir deney ayarları. `tests`: veri sızıntısı, bölüm ayrımı, metrik ve model serileştirme kontrolleri. `docs`: yol haritası, veri kaynağı ve veri durumu. `reports`: anonim toplu sonuçlar. `data` ve `artifacts`: Git dışında veri ve model çıktıları.

## Aşamaların bağımlılıkları

Veri envanteri → sabit hasta ayrımı → metadata baseline → görüntü baseline → hasta bağlamı → vaka erişimi → bağımsız test. Veri erişiminde sorun varsa sentetik sonuç gerçek performans diye sunulmaz; erişim sorunu ve çalışır durumdaki altyapı ayrı raporlanır.
