# Sabit modellerle test protokolü

Amaç: doğrulamada gözlenen embedding katkısının daha önce değerlendirilmemiş hasta grubunda sürüp sürmediğini ölçmek. Birincil karşılaştırma birleşik model ile metadata modelinin test AP farkıdır. Yalnız görüntü modeli yardımcı karşılaştırmadır. Test aynı SLICE-3D kaynağından ayrılmıştır; başka merkezde dış doğrulama değildir.

## Test açılmadan önce sabitlenenler

- `metadata-full-control-v1`, `image-eg2-70-full-v1`, `fusion-eg2-70-full-v1` model dosyaları ve varsa ön işleme dosyaları.
- Doğrulamada seçilmiş eşikler: sırasıyla 0.01776900142431259, 0.0375378355383873, 0.002028954681009054.
- Doğrulamada seçilmiş en iyi ağaç turları: sırasıyla 62, 35, 276 (sıfır tabanlı). Tahminde açıkça bu turlar kullanılır.
- Önceki eğitim raporları, paket sürümleri, veri/split checksum'ları, encoder revision ve görüntü ayarları.
- Test grubu: 48.870 lezyon / 157 hasta / 65 pozitif lezyon. Hasta ayrımı özgün hazırlık adımından gelir; yeniden bölünmez.
- 1.000 eşleştirilmiş hasta bootstrap tekrarı, seed=42, %95 percentile aralığı. Bir hastanın bütün lezyonları aynı ağırlığı alır. AP, ROC-AUC ve normalize edilmemiş yüksek duyarlılık pAUC farkları raporlanır.

Model/ön işleme/rapor SHA-256 değerleri [protokolde](../reports/heldout_test_protocol.json) tutulur. Bu kayıt test tahmininden önce commit edilmiştir. Eğitim ve doğrulama etiketleri model geliştirmede kullanılmıştır; test etiketleri yalnızca önceden mevcut bölümleme bütünlüğü/sayılarının kontrolünde ve son performans değerlendirmesinde kullanılır.

## Çalıştırma

Repo kökünde, tam veri deneyi tamamlanmış ve yerel model dosyaları mevcutken:

```bash
uv sync --extra embeddings --group dev --locked
# Yalnız ilk değerlendirmeden önce; var olan protokolün üzerine yazmaz:
uv run --extra embeddings python -m derm_context.test_protocol
# Ayrı süreç: macOS'te encoder yüklenmeden önce XGBoost import edilmez.
uv run --extra embeddings python -m derm_context.test_embeddings
uv run --extra embeddings python -m derm_context.evaluate_test
uv run --extra embeddings python -m derm_context.report_test
```

Bu depoda protokol zaten sabittir; aynı yerel model dosyalarıyla devam ediliyorsa ilk komut olan `test_protocol` tekrar çalıştırılmaz. Paylaşılan Git deposu model ağırlıklarını içermez. Başka bir kurulumda yeniden eğitilen dosyaların hash'leri bu kayıttan farklıysa denetim durur; bu yeni çalışmanın aynı dondurulmuş çalışma olduğu iddia edilmemelidir.

`test_embeddings` yalnız test görüntülerini işler, her 512 görüntüde checkpoint alır ve tamamlanan cache'i checksum ile doğrular. Mevcut eğitim/validation cache'ine test satırı eklemez. Veri ve model sürümleri değişirse devam etmez. Encoder frozen, 70 görüntü token'ı, 768 boyut, batch 16, MPS bfloat16; önceki çalışmayla aynı koşullar.

`evaluate_test` model ve ön işlemeyi yalnızca yükler; `fit`/`fit_transform` veya eşik seçimi çağırmaz. Cache sırası, test kimlikleri, model dosyaları ve runtime denetlenir. Test tahminleri `artifacts/heldout-test-v1/` altında yerelde kalır. Mevcut değerlendirme sonuçlarının üzerine yazılmaz. Model ve veriler değerlendirmeden sonra tekrar doğrulanır.

## Yorumlama

Eşikler testte ayarlanmadığı için modellerin test duyarlılıkları farklı olabilir. Doğrulamada hedeflenen %95 duyarlılık testte garanti değildir. Yanlış pozitifler, duyarlılık ve yanlış negatiflerle birlikte okunmalıdır.

Bootstrap aralıkları aynı eğitilmiş modellere koşulludur; yeniden eğitim değişkenliğini ve dış merkez genellemesini kapsamaz. İkincil karşılaştırmalar için çoklu test düzeltmesi uygulanmaz. Pozitif AP farkı, aralık sıfırı içeriyorsa tek başına kesin üstünlük kanıtı değildir. Test artık görülmüş olduğundan gelecekteki model ayarları için yeni bağımsız değerlendirme gerekir.

Ham veriler, embedding'ler, ağırlıklar ve lezyon bazında tahminler Git'e eklenmez. Toplu raporlar `reports/heldout_test_*` altında yayımlanır. Önceki doğrulama raporları tarihsel deney kaydı olarak korunur; oradaki “test kullanılmadı” ifadeleri o aşamayı anlatır.
