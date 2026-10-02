# Karar Günlüğü

Proje planlaması, izlenecek yolu ve bu dokümandaki çözümler yapay zeka destekli araştırmalar sonucunda hazırlandı.
Her adımın sonunda: ne yaptım, neden, hangi alternatifi eledim ve sayısal bulgular.
Bu notlar TECHNICAL.md'nin ham maddesi olarak düşünülebilir.

---

## Faz 0 - Kurulum

- Python 3.12 + venv, paket yapısı `src/fraud_platform` (src layout).
- Tüm yollar ve kolon rolleri `config/settings.yaml`'da; kod kolon adı ezberlemiyor.
- Veri ve modeller repoya girmiyor (`.gitignore`), README'deki adımlarla üretilecek.

Kararlar ve gerekçeler:
- src layout kullandım: paket `pip install -e .` ile kurulmadan import edilemiyor. Böylece notebook, script, test ve API
  aynı kurulu kodu kullanıyor, klasör yapısına bağlı yanlış import olmuyor.
- Case sistemin farklı verilere uyarlanabilir olmasını istiyor. Kolon adlarını, yolları, eşikleri ve ağırlıkları koda değil
  `config/settings.yaml`'a koydum; başka bir işlem verisine geçmek kod değil config değişikliği.
- Config'i Pydantic ile okuyorum: yanlış tipte veya eksik bir ayar varsa sistem açılışta hata veriyor, sessizce yanlış çalışmıyor.
- Veriyi ve eğitilmiş modelleri repoya koymadım (Kaggle kuralları, GitHub dosya boyutu sınırı). README'deki adımlarla yeniden
  üretiliyorlar; bunu temiz bir klasörde baştan kurarak doğruladım.

---

## Adım 1 - Veri yükleme ve kalite

Ham sayılar (notebooks/01_veri_yukleme_kalite.ipynb):
- Left join: 590.540 x 435, identity oranı %24,4, ID tekrarı yok
- Bellek 2,08 GB -> 1,13 GB (float32). TransactionAmt'te 3 hanede kayıp: 0 satır
- Tipler: 375 numeric, 31 categorical (13 high), 26 binary; override = Kaggle kategorik listesi
- Eksiklik: 12 kolon >%90 boş, 26 blok (en büyüğü 46 V kolonu, %77,9 birlikte boş)
- ID hariç 3 çift aynı satır: ardışık ID, aynı saniye, hepsi isFraud=0
- D4/D11/D15'te negatif gün farkı (15/7/15 satır)
- gmail / gmail.com yazım farkı; ülke uzantılı domainler (yahoo.com.mx, hotmail.de ...)
- TransactionAmt skew 14,4 -> log1p sonrası 0,49; tutarların %10,5'i 3 ondalık haneli (fraud %11,7 vs %2,5)
  (ilk ölçüm %12,5 idi; float hatası: 159.95*100 % 1 = 0.99999 -> Adım 3'te tamsayı kontrolüyle düzeltildi)
- Fraud oranı: genel %3,5 | ProductCD=C %11,7 | identity var %7,9 / yok %2,1 | mobile %10,2

Kararlar ve gerekçeler:
- Az sayıda farklı değeri olan sayısal kolonları (çoğu V, C, D) sayısal bıraktım. Bunlar sayaç veya gün farkı; 3'ün 1'den büyük
  olması anlamlı, kategorik yapsaydım bu sıra bilgisi kaybolurdu. Kategorik saydıklarım metin tipindeki kolonlar ve Kaggle
  açıklamasında kategorik olduğu yazan kolonlar (card1-6, addr1-2, id_12-38); bunları config'te override olarak verdim.
- ID dışında birebir aynı 3 çift satır var: ardışık ID, aynı saniye, aynı kart ve tutar. Büyük ihtimalle çift gönderim.
  Hepsi isFraud=0 ve 590 bin satırda 6 satır; silmedim, kalite raporunda gösterdim. Canlı sistemde velocity'yi şişirebilecekleri
  için tekilleştirme gerekir.
- `gmail` ile `gmail.com` aynı domain; Adım 3'teki temizlikte `gmail.com`'a birleştirdim, yoksa frekans ve nadirlik feature'ları
  aynı domaini iki ayrı değer sayardı. DeviceInfo'daki büyük/küçük harf farklarını (Moto/moto) küçük harfe çevirerek birleştirdim.
  Ülke uzantılı domainler (yahoo.com.mx, hotmail.de) için "yabancı e-posta" bayrağı düşünmüştüm; Adım 3'te bu işlemlerde fraud
  oranı ortalamayla aynı çıktı (%3,1), feature yapmadım.
- D4, D11 ve D15'teki negatif gün farklarını veri hatası saydım ve Adım 3'te 0'a kırptım.
- %90'dan fazla boş 12 kolonu silmedim ve doldurmadım. Doldurmak sahte değer üretirdi; boş olmanın kendisi de bilgi taşıyor
  (identity olan işlemlerde fraud %7,9, olmayanlarda %2,1). Column dedektöründe boş değer skora katkı vermiyor, multivariate'de
  -1 olarak ayrı bir bölge oluyor.

## Adım 2 - Profiling

Ham sayılar (notebooks/02_profiling.ipynb):
- Tarih aralığı (varsayım 2017-11-30): 2017-12-01 -> 2018-05-31, 182 gün
- Saat: hacim 05-11 arası çukur (min saat 09: 2.479 işlem), fraud zirvesi saat 07: %10,6 (lift 3,0).
  Yaygın "gece = 00-05" tanımı bu veride tutmuyor; saat muhtemelen UTC, ~5 saat kayma
- Gün: Cmt/Paz en düşük hacim (79,8k / 70,2k) -> referans tarih varsayımıyla tutarlı; fraud oranı günler arası %3,2-3,7
- Outlier: medyan aykırı oranı IQR %10,1 / robust-z %3,7. V148-V156 aykırılarında fraud %95-98 (lift ~28)
- MAD=0 olan kolonlar var (C, D) -> robust_z ortalama mutlak sapmaya düşüyor
- Boşluk sinyali: addr1 boş -> fraud %11,8 vs %2,5 (lift 4,8); identity/D kolonları boş -> lift ~0,2-0,25
- Nadir kombinasyon (5 kolon): işlemlerin %7,6'sı, fraud %4,9 vs %3,4. ProductCD+card4 nadir çiftleri lift 3,0
- Korelasyon: 375 sayısal -> 260 temsilci (|r|>0,95). C1/C2/C4/C6/C8/C10/C11 tek grup, D1~D2
- Cramér's V: id_22~id_26 = 1,0; id_13~DeviceType 0,82
- Tek kolon AUC: en iyi D5 0,743, V258 0,735; 138/375 kolon AUC > 0,6
- uid karşılaştırma (purity_fraud): card1 0,055 | card1+addr1 0,162 | +D1n 0,761 | +P_email 0,869
  geçmişi olan işlem oranı: 0,994 | 0,973 | 0,788 | 0,699
- D1n card1+addr1 içinde medyan 1 farklı değer -> D1 = kartın yaşı yorumu destekleniyor
- Entity (card1+addr1+D1n): 50+ işlemli entity fraud %7,1 vs tek işlemli %2,4; 3+ cihaz %8,9 vs cihazsız %2,2

Kararlar ve gerekçeler:
- uid = card1 + addr1 + (işlem günü - D1): card1 tek başına kişi değil (13.553 değer, purity 0,055). D1'i kartın ilk
  kullanımından beri geçen gün olarak yorumladım; işlem günü - D1 kart başına sabit bir başlangıç günü veriyor (aynı card1+addr1
  içinde medyan 1 farklı değer). Bu tanımla purity 0,761 ve işlemlerin %78,8'inin geçmişi var. P_emaildomain eklemek purity'yi
  0,869'a çıkarıyor ama geçmişli işlem %69,9'a iniyor ve e-posta %16 boş; aynı kişiyi bölebileceği için eklemedim.
  Etiketi sadece tanımları karşılaştırmak için kullandım, hiçbir feature etiketten türemiyor.
  Zayıf yanı: C/H/R ürünlerinde D1 %73-95 oranında 0, bu ürünlerde aynı kişi her gün yeni uid alıyor.
- Saat kaydırma (-5): Hacim göreli 05-11 arasında en düşük, fraud oranı tam bu saatlerde zirvede (07'de %10,6).
  E-ticarette hacmin en düşük olduğu saatler gece geç saatleri; TransactionDT'nin UTC, müşterilerin ağırlıkla ABD doğu saatinde
  olduğunu varsayıp 5 saat kaydırdım. Böylece çukur yerel 00-06'ya oturuyor. Gece 00-06, iş saati 09-18. Yaz saatini hesaba
  katmadım. Otomatik pencere bulmak yerine kaydırmayı seçtim çünkü kurallar "iş saati 09-18" gibi okunabilir kalıyor.
- Boşluk bayrakları: Her kolon için değil, boşluk bloğu başına bir bayrak: addr1 (boşken 4,8 kat risk), dist1, R_emaildomain
  ve M blokları (M1, M4, M6, M7). Ayrıca satırdaki toplam boş sayısı (null_count).

## Adım 3 - Feature engineering

Ham sayılar (notebooks/03_feature_analizi.ipynb):
- 7 builder, 55 feature (53 sayısal + uid + device_key), 590k satırda 5,7 sn
- Leakage: tüm veriden hesapta satırların %63,1'i gelecek bilgisi taşıyor; sadece geçmişte %36,9 ilk işlem (feature boş).
  AUC farkı ihmal edilebilir (amount_to_user_avg 0,516 vs 0,515) -> sadece geçmiş seçildi
- Cihaz anahtarı: DeviceInfo tek başına cihaz değil ("Windows" %40). DeviceInfo+id_31+id_33+id_19+id_20 -> 44.733 değer, max grup %0,8
- D1: C/H/R ürünlerinde %73-95'i 0 -> is_new_card sadece W'de anlamlı (D1<=7: %2,6 vs %1,7);
  aynı sebeple C/H/R'de uid her gün değişiyor (sınırlama)
- Çürüyen varsayımlar: yabancı e-posta uzantısı fraud %3,1 (= ortalama, feature yapılmadı);
  tutar/kullanıcı ort. tüm aralıklarda ~%4 (AUC 0,515); hafta sonu lift 1,01; yeni e-posta lift 1,0
- En güçlü: R_emaildomain_freq 0,675, time_since_last_tx 0,674, null_count 0,671, email_match 0,658, combo_rarity 0,643
- Bayrak lift: addr1_isna 3,37 | amount_is_converted 3,35 | foreign_address 2,92 | mobile 2,91 | email_match 2,76 |
  rapid_repeat 2,65 | new_device 2,32 | night 2,24 | high_value_customer 1,53 (!) | first_tx 0,79 (!) | business_hours 0,91
- Velocity: 24 saatte 0 önceki işlem %2,4 -> 6-10 işlem %8,2. Cihazda 6-20 önceki kullanıcı -> %12

Kararlar ve gerekçeler:
- Sadece geçmiş (past-only): Kullanıcı ortalamasını tüm veriden hesaplasaydım satırların %63,1'inde "geçmiş" gelecekteki
  işlemleri de içerecekti. Sinyal kaybı çok küçük (amount_to_user_avg AUC 0,516 -> 0,515, user_tx_count 0,568 -> 0,560). Asıl
  sebep train/serve tutarlılığı: API'de gelecek görünmez, eğitimde de görünmemeli. Bedeli işlemlerin %36,9'unun kullanıcının ilk
  işlemi olması ve entity feature'larının boş kalması; bunu is_first_tx bayrağıyla işaretledim. Sızıntı olmadığını testle
  sabitledim (son satırları değiştirmek öncekilerin feature'larını değiştirmiyor).
- Cihaz anahtarı: DeviceInfo tek başına cihaz değil, "Windows" değeri tek başına %40 pay alıyor. DeviceInfo + id_31
  (tarayıcı) + id_33 (ekran çözünürlüğü) + id_19 + id_20 birleşimi 44.733 farklı değer veriyor, en büyük grup %0,8.
- Çürüyen varsayımlar: Yabancı e-posta uzantısında (.mx, .de) fraud %3,1, ortalamayla aynı; feature yapmadım. Kişisel tutar
  sapması (tutar / kullanıcı ortalaması) her aralıkta ~%4, AUC 0,515; bu veride zayıf. Hafta sonu lift 1,01, yeni e-posta 1,0.
  Kendi ölçüm hatamı da buldum: "tutarların %12,5'i 3 ondalık haneli" demiştim, float hatasıydı (159.95 x 100 % 1 = 0.99999);
  tamsayı kontrolüyle doğrusu %10,5.
- Adım 6'ya taşıdığım ters bulgular: Yüksek değerli müşteri daha riskli çıktı (lift 1,53), oysa ben bu müşteriler için
  skoru düşürmeyi planlamıştım. İlk işlem daha az riskli (0,79), iş saati biraz daha güvenli (0,91). Bunlar context kurallarını
  varsayıma göre değil ölçüme göre kurmam gerektiğini gösterdi.
- Yeni kart sadece W'de: C/H/R ürünlerinde D1 çoğunlukla 0, bu ürünlerde "yeni kart" anlamsız. W'de D1 <= 7 olanlarda fraud
  %2,6, diğerlerinde %1,7.

## Adım 4 - Dedektörler

Ham sayılar (notebooks/04_degerlendirme.ipynb, scripts/train.py):
- Zaman bazlı bölme: train %60 (354k) / valid %20 / test %20 (118k'şar). Fraud: %3,38 / %3,90 / %3,44
- Tasarım kararları valid'de; ilk denemelerde (column toplama, IF girdisi, entity bileşenleri) son %20'ye bakılarak seçim
  yapılmıştı, üçü de valid'de tekrarlandı ve sıralama aynı kaldı (aşağıdaki sayılar valid)
- Column: satır skoru = tüm kolon skorlarının ortalaması. Valid PR: max 0,046 / top5 0,077 / top10 0,120 / ortalama 0,221 (317 kolonda max doyuyor)
- Multivariate: Isolation Forest, 53 feature + 260 temsilci, kantil dönüşümü + boş=-1. Valid PR: ham+medyan 0,086 -> kantil 0,094
  -> kantil + temsilciler 0,155
- Entity: saat sapması ve yeni e-posta ROC ~0,51 -> çıkarıldı; 5 bileşen eşit ağırlık PR 0,083 -> 0,092
- Temporal: 7 bileşen eşit ağırlık; max'tan iyi (0,090 vs 0,074)
- Valid PR-AUC: column 0,221 | multi 0,155 | entity 0,092 | temporal 0,090
- Test PR-AUC:  column 0,120 | multi 0,100 | entity 0,078 | temporal 0,055 (ROC 0,64-0,77) -> drift
- Test %3 alarm precision: column %21,5 | multi %10,4 | entity %12,5 | temporal %6,1
- Spearman: column~multi 0,81; entity/temporal diğerleriyle 0,32-0,49. Alarm Jaccard: entity diğerleriyle 0,04-0,06
- Test 4.064 fraud: 1.210'u en az bir katmanın top %3'ünde; 721'ini tek katman yakalıyor (column 311, entity 214, multi 119, temporal 77)
- Online == offline: 200 gerçek test işleminde 55 feature'da 0 fark, 60 ms/işlem
- Multivariate açıklaması: kantil yerine kuyruk olasılığı (0/1 bayraklar %100 uç görünüyordu)

Kararlar ve gerekçeler:
- Zaman bazlı bölme: Aynı veride fit edip ölçmek iyimser olur; gerçekte model geçmişte eğitilip gelecekte çalışır. İlk %60
  train, %60-80 valid, son %20 test. Tasarım kararlarını valid'de verdim, test sadece final rapor. Dürüst not: ilk denemelerde üç
  seçimi (column toplama, IF girdisi, entity bileşenleri) son %20'ye bakarak yapmıştım; fark edince üçünü de valid'de tekrarladım,
  sıralama değişmedi. Bu bölme drift'i de gösterdi: column PR-AUC valid 0,221, test 0,120.
- Column: ortalama: 317 kolonda neredeyse her işlemin en az bir uç kolonu var, max doyuyor. Valid PR: max 0,046, top-10
  0,120, ortalama 0,221. Ortalama "kaç kolonda birden tuhaflık var" sorusunu soruyor. Gerekçe olarak yine en uç 3 kolonu gösteriyorum.
- Multivariate: kantil dönüşümü: Her kolonu 0-1 düzgün dağılıma çeviriyor; böylece Isolation Forest tek kolonun ucunu (bu
  column katmanının işi) değil, kolonların birlikte nadir görülen kombinasyonunu yalıtıyor. Valid PR: ham + medyan doldurma 0,086
  -> kantil 0,094 -> kantil + 260 temsilci kolon 0,155. Boş değerleri -1'e koydum, "boş" ayrı bir bölge olarak öğreniliyor.
- Entity ve temporal bileşenleri: Saat sapması ve yeni e-posta valid'de ROC ~0,51 verdi, çıkardım (PR 0,083 -> 0,092). Tutar
  sapması da zayıf ama entity katmanının özü olduğu için tuttum. Kalan 5 bileşen eşit ağırlıkta; bileşen ağırlıklarını optimize
  etmek aşırı uyuma yol açardı. Temporal'da 7 bileşenin eşit ağırlıklı ortalaması max'tan iyi çıktı (0,090 vs 0,074).
- Profil deposu: API için ayrı bir "canlı formül" yazmadım. Depo kullanıcının ve cihazın geçmiş işlemlerini (sadece
  pipeline'ın okuduğu ham kolonları) tutuyor; yeni işlem geçmişin sonuna eklenip aynı pipeline'dan geçiyor. Eğitim ve canlı hesap
  tasarım gereği aynı: 200 gerçek test işleminde 55 feature'da 0 fark, işlem başına ~60 ms.

## Adım 5 - Skor birleştirme

Ham sayılar (notebooks/04_degerlendirme.ipynb bölüm 5):
- Yöntem: valid ikiye bölündü; valid-A ağırlık bulmak, valid-B kıyaslamak için. Normalizasyon referansı = train skorları
- Normalizasyon (eşit ağırlık, valid-B PR): yüzdelik 0,137 | min-max 0,158 | z-sigmoid 0,147 | kuyruk-log 0,170
  -> yüzdelik kuyruğu sıkıştırıyor; kuyruk-log = -log10(1 - yüzdelik) (Fisher benzeri)
- Ağırlıklar (valid-B PR / ROC): eşit 0,170/0,773 | sezgisel 0,163/0,774 | orantılı 0,198/0,776 | grid 0,217/0,755 | max 0,152
- Tek katman valid-B: column 0,213 | multi 0,143 | entity 0,087 | temporal 0,063
- Grid optimumu 4 dilimde: column 0,6-0,8, entity 0,1-0,2, multi 0-0,2, temporal çoğunlukla 0
- Seçilen: performansa orantılı (column .424, multi .293, temporal .159, entity .124)
- Test: raw_anomaly_score PR 0,113 / ROC 0,766 / %3 prec %18,4; column tek başına PR 0,120 / ROC 0,751 / prec %21,5
  -> birleşim PR'de column'u geçmiyor, ROC'ta geçiyor
- Uçtan uca tek işlem: scripts/score_transaction.py, ~450 ms (gerekçelerle)
- Testlerin yakaladığı 2 gerçek hata: küçük veride kantil sayısı > satır sayısı; JSON null DeviceInfo float tiplenip .str.lower() çöküyordu

Kararlar ve gerekçeler:
- Kuyruk-log normalizasyon: En basit yöntem olan düz yüzdelik kuyruğu sıkıştırıyor: %99 ile %99,99 arasındaki fark 0,0099 ve
  fraud'lar tam bu kuyrukta. -log10(1 - yüzdelik) kuyruğu açıyor (%99 -> 2, %99,99 -> 4); p-değerlerini log ölçekte birleştiren
  Fisher yöntemine benziyor. Valid-B'de eşit ağırlıkla PR: yüzdelik 0,137, min-max 0,158, z-sigmoid 0,147, kuyruk-log 0,170.
  Eğitim dağılımının yüzdelik ızgarasını kaydettiğim için API'de tek işlemde de çalışıyor.
- Performansa orantılı ağırlık: Valid'i ikiye böldüm: valid-A'da ağırlığı buldum, valid-B'de kıyasladım (optimize edilen
  seçenek aynı veride ölçülse tanım gereği kazanırdı). Ağırlıklar valid-A'daki tek başına PR kazancıyla orantılı: column .424,
  multivariate .293, temporal .159, entity .124. Valid-B'de orantılı PR 0,198 / ROC 0,776, grid optimumu 0,217 / 0,755, eşit
  0,170, sezgisel ilk tahminim 0,163. Grid'i seçmedim: multivariate ve temporal'ı sıfırlıyor, sadece onların yakaladığı fraud'lar (testte 196)
  kaybolur ve tek bir katmanın drift'ine açık kalır.
- Birleşim column'u neden geçmiyor? Testte ham birleşik skor PR 0,113, column tek başına 0,120. Column zaten 317 kolonu
  topluyor ve multivariate ile çok benzer (Spearman 0,81); entity ve temporal tek başına zayıf. Doğrusal birleşim PR'ye az şey
  ekliyor; kazanç ROC'ta (0,766 vs 0,751), dört açıdan açıklamada ve sonraki adımların katman skorlarını ayrı kullanabilmesinde.
  Context düzeltmesinden sonra birleşik skor column'u geçti (PR 0,125). Daha iyisi etiketle stacking olurdu ama bu artık
  denetimli bir model.

## Adım 6 - Context adjust

Ham sayılar (notebooks/04_degerlendirme.ipynb bölüm 6, scripts/calibrate_context.py):
- 14 kural (11 aktif, 3 kanıtıyla kapalı), 4 kategori: zaman, entity, işlem tipi, coğrafi. Toplam çarpan sınırı [0,5; 2,0]
- Çarpan = koşullu lift^0,5; koşullu lift alarm bölgesinde (en riskli %10) skor dilimlerine göre gözlenen/beklenen fraud (valid-A)
- Ham vs koşullu lift: ürün C 3,05 -> 1,05 | adres boş 3,14 -> 1,04 | kur çevrimi 3,03 -> 1,05 (kapalı, skor zaten görüyor)
  gece 2,69 -> 1,48 | yurt dışı 2,80 -> 2,39 | güvenilir 0,23 -> 0,46 | ilk işlem 0,85 -> 2,33 | yüksek değerli 1,67 -> 0,65
- Kapalı 3 kural alarm bölgesinin %66-74'ünü kapsıyor (alarm listesi zaten büyük ölçüde onlardan oluşuyor)
- İlk deneme: tüm veride koşullu lift PR 0,228; alarm bölgesinde 0,242 ve high_value yön değiştirdi (1,30 -> 0,80)
- Sınır: [0,5;1,5] işlemlerin %25,8'ini kırpıyor (PR 0,2423); [0,5;2,0] %1,5 (PR 0,2440)
- Sabit bütçe %3 - valid-B: TP 477->543, FP 1295->1229, prec %26,9->%30,6, PR 0,198->0,244
                  - test: TP 652->747 (+95), FP 2891->2796 (-95), prec %18,4->%21,1, PR 0,113->0,1245 (column tek başına 0,120'yi geçti)
- Sabit eşik, sadece aşağı çeken 4 kural: valid-B FP 919->680 (-%26, -27 TP) | test FP 3680->3078 (-%16,4, -116 TP)
- Ablation (test ΔPR kapatınca): ilk işlem -0,0059 (+71 FP) | güvenilir -0,0014 | kredi kartı -0,0014 | yeni kart +0,0004 (hafif zararlı)
- Test alarm listesinde kapsam: kredi kartı %76, güvenilir %0,14 (işlemlerin %17'si) 
- Yöntemin bilinen sapması: beklenen oran kuralın kendi işlemlerini içerdiği için gerçek etkiyi biraz küçümser (test_context)

Kararlar ve gerekçeler:
- Koşullu lift: Ham lift skorun zaten gördüğü riski de içeriyor; çarpan vermek aynı riski iki kez saymak olur. Örneğin ürün
  C'nin ham lift'i 3,05 ama aynı skor dilimindeki işlemlere göre 1,05. Çarpanı bu yüzden koşullu lift'ten türettim: kuralın
  kapsadığı işlemlerde gözlenen fraud / aynı skor dilimine göre beklenen fraud, yani "skorun göremediği risk". Ürün C, adres boş ve
  kur çevrimi kurallarını bu yüzden kanıtlarıyla kapalı bıraktım; alarm bölgesinin %66-74'ünü kapsıyorlar.
- Alarm bölgesi: Alarm kararları skorun tepesinde verildiği için lift'i en riskli %10'da ölçtüm. Tüm veride ölçünce yüksek
  değerli müşteri 1,65 (riskli), alarm bölgesinde 0,65 çıktı; çarpan 1,30'dan 0,80'e döndü ve valid-B PR 0,228'den 0,242'ye
  çıktı. Sonuç iş mantığıyla da uyumlu (VIP müşteriye tolerans).
- Çarpan = koşullu lift^0,5: 11 çarpanı ayrı ayrı optimize etmek aşırı uyum olurdu; tek bir sönümleme parametresi kullandım.
  Karekök tek bir kuralın skoru domine etmesini engelliyor. Tek kural için [0,5; 1,5] sınırı var; ilk işlem ve yurt dışı bu
  sınıra takılıyor.
- Toplam sınır [0,5; 2,0]: Sınır bir güvenlik ağı, nadiren devreye girmeli. [0,5; 1,5] işlemlerin %26'sını kırpıyordu (ilk
  işlem x gece gibi kombinasyonlar çalışmıyordu), [0,5; 2,0] %1,5'ini. Alt sınır 0,5: hiçbir bağlam skoru yarısından fazla düşüremez.
- Zorunlu ama etkisiz kurallar: İş saati ve hafta sonu case'in istediği kurallar; uyguladım ve etkisini ölçtüm. Koşullu
  lift'leri 0,90 ve 1,02, çarpanları nötre yakın. Veriyle çelişen bir varsayımı zorla uygulamak yerine ölçüp göstermeyi seçtim.
- En değerli ve zararlı kurallar: En değerlisi geçmişsiz işlem (x1,50): entity ve temporal katman ilk işlemde nötr kaldığı
  için skor riski eksik tahmin ediyor; kapatınca test PR -0,0059 ve %3 bütçede +71 yanlış alarm. Yeni kart (W) iki dönemde de
  hafif zararlı (valid-B +0,0010, test +0,0004); açık bıraktım ve sonucu raporladım.
- FP azaltımını iki senaryoda ölçtüm: Sabit alarm bütçesinde (analist kapasitesi sabit) testte yanlış alarm 2.891 -> 2.796,
  yakalanan fraud 652 -> 747, precision %18,4 -> %21,1. Sabit eşikte tüm kurallarla alarm sayısı artıyor çünkü yukarı çeken
  kurallar fazla; sadece aşağı çeken 4 kural yanlış alarmı testte %16,4 azaltıyor ama 116 fraud kaçıyor. "FP azaldı" iddiasının
  hangi senaryoda geçerli olduğunu ayrı göstermek istedim.

## Adım 7 - Rule engine

Ham sayılar (notebooks/04 bölüm 7, scripts/rule_report.py):
- 14 kural (12 aktif, 2 deneme kuralı kanıtıyla kapalı); YAML ve JSON destekli; 3 çakışma stratejisi; reload
- Skor kurallara alan olarak giriyor (risk_percentile); skor da öncelikli bir kural gibi çakışma çözümüne katılıyor
- Öncelik bantları: BLOCK 90-100 | REVIEW 65-89 | beyaz liste 60 | FLAG 40-59
- Ek isabet testi: skorun %3 bütçesinin hemen altındaki dilim isabeti %11. Bütçe dışı isabet: e-posta %21,4, gece/yurt dışı %20,1,
  velocity patlaması %18,2 | cihaz paylaşımı %7,7, velocity yüksek %5,7, harcama %8,3, yeni cihaz %9,1 -> FLAG
- Taslak (hepsi REVIEW): alarm %8, precision %19,4 < aynı hacimde skor %20,8. Bantlı: alarm %4,6, prec %28,3 > skor %27,6
- Reddedilen kurallar: gece VE yurt dışı VE >=500 -> valid'de 0 tetiklenme; kişisel sapma 10x -> fraud %1,9 (< ort.)
- Valid kararlar: BLOCK %0,35 (prec %67,6) | REVIEW %4,25 (%25,1) | FLAG %6,1 (%7,6) | ALLOW %89,3 (%2,4)
- TEST OLAYI: tek meşru entity (15775_330_129: 1.414 işlem, ProductCD S, 17 farklı tutar, 0 fraud) R001'i 1.245 kez tetikledi;
  BLOCK isabeti %8,3. Valid fraud patlamaları hep <=70 geçmiş işlemli -> koruma user_tx_count < 100 (valid'de 0 değişiklik).
  Koruma sonrası test BLOCK 1.718 -> 416, isabet %8,3 -> %34,1, TP aynı (142). Sorun testte bulundu -> düzeltme sonrası test iyimser
- Test (korumalı): BLOCK %34,1 | REVIEW+BLOCK alarm %5,3, prec %20,2, recall %31,4 (aynı hacim skor %20,4 / %31,7 -> başa baş)
- R001 testte 89 tetiklenme, 0 fraud (testte velocity fraud'u yok); R003/R004 isabeti testte düştü (%22->%10, %26->%15; e-posta listesi valid'den -> aşırı uyum)
- Strateji: bantlı tasarımda 3 strateji aynı karar (dayanıklı). Dosya ters + first_match: BLOCK 417 -> 0, recall %33 -> %15
- 118 bin işlem vektörel değerlendirme 0,1 sn; toplu == tek işlem (testle doğrulandı)

Kararlar ve gerekçeler:
- Skor bir kural olarak: Anomali skoru kurallara alan olarak giriyor: risk yüzdeliği >= 0,97 -> REVIEW (öncelik 70), >= 0,999 ->
  BLOCK (95). Böylece skor ve iş kuralları tek bir çakışma çözümünde birleşiyor: yüksek öncelikli bir iş kuralı skoru geçersiz
  kılabiliyor, beyaz liste kılamıyor.
- Öncelik bantları: BLOCK 90-100, REVIEW 65-89, beyaz liste 60, FLAG 40-59. Bant olmadan yüksek öncelikli bir FLAG kuralı
  skorun REVIEW kararını ezebiliyordu. Bantlar ve önceliğe göre sıralı dosyayla üç çakışma stratejisi aynı kararı veriyor; kural
  seti strateji seçimine dayanıklı.
- Aksiyonu ek isabet testiyle seçtim: Skorun %3 bütçesinin hemen altındaki dilimde isabet %11; bu, bütçeyi büyütmenin
  marjinal değeri. Bir kural bütçe dışına bundan daha isabetli alarm ekliyorsa REVIEW (e-posta %21,4, gece/yurt dışı %20,1),
  eklemiyorsa FLAG (cihaz paylaşımı %7,7, velocity %5,7, harcama %8,3, yeni cihaz %9,1). BLOCK'u sadece çok yüksek isabetli
  kurallara verdim, yanlış blok müşteriyi doğrudan etkiliyor. İlk taslakta hepsi REVIEW'du: alarm %8, precision %19,4, aynı
  hacimde skorun (%20,8) gerisinde. Bantlı sürümde alarm %4,6, precision %28,3, aynı hacimde skor %27,6.
- Eşikleri dağılıma göre koydum: 500 USD %96. yüzdelik, son 1 saatte >=10 işlem %99,9, >=5 işlem %99'un üstü, 24 saatte
  2.000 USD ~ %99, tutar / kullanıcı ortalaması >=3 ~ %95.
- Test olayı ve koruma: İlk kural setinde test döneminde BLOCK isabeti %67,6'dan %8,3'e düştü. Sebebini aradım: tek bir
  entity (1.414 işlem, hepsi ProductCD S, 17 farklı tutar, hiç fraud yok; abonelik ya da entegrasyon hesabı gibi) velocity
  kuralını 1.245 kez tetiklemiş. user_tx_count < 100 korumasını ekledim: yerleşik yüksek hacimli bir hesabın velocity'si kendisi
  için olağan, otomatik blok yerine incelemeye düşmeli. Valid'deki gerçek fraud patlamalarının hepsi 70'ten az geçmiş işlemli
  kullanıcılarda, koruma valid'de hiçbir kararı değiştirmiyor. Testte BLOCK 1.718 -> 416, isabet %34,1, yakalanan fraud aynı.
  Sorunu testte bulduğum için düzeltme sonrası test sonucu iyimser; iki durumu da raporladım. Bu olaydan bilgi tabanına iki
  politika ekledim: onaylı yüksek hacimli hesaplar (FP-25) ve blok devre kesici (FP-41).
- Reddettiğim kurallar: "Gece VE yurt dışı VE >=500" valid'de hiç tetiklenmedi, VEYA'lı halini (R003) kullandım.
  "Kişisel sapma 10x" kuralında fraud %1,9, ortalamanın altında. İkisini de kanıtıyla kapalı bıraktım.
- Testte zayıflayan kurallar: Velocity patlaması testte 89 kez tetiklendi ve hiç fraud yakalamadı (test döneminde bu tip
  fraud yok). Riskli e-posta listesini valid'den seçtiğim için testte isabet %26'dan %15'e düştü; bu liste düzenli güncellenmeli.

## Adım 8 - RAG

Ham sayılar (notebooks/05_rag.ipynb, scripts/eval_rag.py):
- Bilgi tabanı: 7 doküman, 18 kurgusal politika (19 parça, 30-72 kelime); 12 kuralın policy_ref'inin hepsi bilgi tabanında
- Embedding: bge-m3 (Ollama, 1024 boyut, çok dilli); taban çizgisi TF-IDF karakter 3-5 gram. FAISS IndexFlatIP (kosinüs)
- Retrieval (18 soru, politika kelimeleri tekrar edilmeden): TF-IDF hit@1 0,72 / hit@3 0,89 / MRR 0,80 | bge-m3 0,83 / 1,00 / 0,91
- Üretim (qwen2.5:3b, sıcaklık 0): doğru atıf %94, atıf doğrulandı %89, kurgusal bilgi RAG %72 vs bağlamsız %6,
  cevapsız sorularda "bilgi yok" %100 (bağlamsız model kripto limiti için "10.000-50.000 dolar" uydurdu)
- Prompt iterasyonu: v1 kaçış ("bilgi yoksa söyle") fazla kullanılıyordu; v2 fact %50 -> %72, atıf %89 -> %94 (aynı sette, hafif iyimser)
- Sıcaklık 0'da bir kez tekrar döngüsü -> 180 sn zaman aşımı -> num_predict 400 + repeat_penalty 1,1
- İşlem açıklaması: önce eski prompt ALLOW'a BLOCK öneriyordu; "kararı motor verdi, sen açıkla" prompt'uyla 14/14 motorla uyumlu, 13/14 atıf doğrulandı
- Atıf doğrulama sadece FP-xx kodları (model kural id'si R002'yi de listeliyordu)
- Windows: faiss.write_index ASCII olmayan yolu ("msı") açamıyor -> serialize_index ile Python'dan yazma (testli)
- Gecikme: soru ~1,4 sn, işlem açıklaması ~6-8 sn; Ollama kapalıysa deterministik fallback
- Bilinen zaaflar: sayı bozma (0,999 -> "%0,9996"), çelişki (FP-25), politikayı yanlış bağlamda uygulama (FP-03 6 saat) -> atıf doğru, içerik yanlış

Kararlar ve gerekçeler:
- Bilgi tabanı: 7 doküman, 18 kurgusal politika yazdım; politika kodları kuralların policy_ref alanıyla birebir eşleşiyor
  (12/12). Ayrıntıları bilerek kurgusal yaptım (cihazın 48 saat gözetimde kalması, yeni kartta 1.500 USD günlük limit gibi):
  LLM bunları önceden bilemez, cevapta geçiyorsa RAG'in çalıştığını gösterir.
- bge-m3: İçerik Türkçe olduğu için çok dilli bir model seçtim, LLM ile aynı yerel serviste (Ollama) çalışıyor. TF-IDF'yi
  taban çizgisi olarak tuttum: hit@3 0,89'a karşı bge-m3 1,00. TF-IDF kelime örtüşmesi olmayan sorularda kaçırıyor. İndirme
  gerektirmediği için testlerde ve yedek olarak duruyor.
- Başlık bazlı parçalama: Her politika tek bir kavram ve 30-72 kelime; her başlığı bir parça yaptım, politika kodu metadata'da.
- Hibrit getirme: İşlem açıklamasında tetiklenen kuralın politikası zaten belli, aramaya gerek yok: policy_ref ile doğrudan
  getiriyorum. Vektör araması kuralın görmediği bağlamı (gece, cihaz gibi) ek politikayla tamamlıyor. Modelin 4.096 token
  bağlamına sığması için en fazla 5 parça.
- Halüsinasyon kontrolü: Cevaptaki her FP-xx kodu getirilen parçalarda olmalı; değilse citations_verified false. Gerçek
  değerlendirmede 18 cevabın 2'si doğrulamadan geçmedi. Sınırı: atfı kontrol ediyor, içeriğin doğruluğunu değil.
- Karar LLM'de değil: İlk prompt'ta LLM, motorun ALLOW dediği bir işleme BLOCK önerdi. Karar deterministik ve denetlenebilir
  olmalı; kararı kural motoru veriyor, LLM açıklıyor. LLM farklı bir aksiyon önerirse llm_agrees_with_engine=false olarak
  işaretleniyor. Prompt'a motorun kararını yazınca 14 işlemlik örneklemde 14/14 uyum oldu.
- Prompt iterasyonu: İlk sürümdeki "bilgi yoksa söyle" kaçışını 3B model bağlamda cevap varken bile kullanıyordu. İkinci
  sürüm önce ilgili politikayı bulup sayıları aynen aktarmasını istiyor. Kurgusal bilgi %50 -> %72, doğru atıf %89 -> %94. Prompt'u
  aynı 20 soruda iyileştirdiğim için bu skorlar biraz iyimser.
- Pratikte öğrendiklerim: Sıcaklık 0'da model bir kez tekrar döngüsüne girip 180 sn zaman aşımına uğradı; üretim sınırı
  (400 token) ve tekrar cezası ekledim. Ollama kapalıysa sistem çökmüyor, kural açıklamalarından deterministik bir açıklama dönüyor.

## Adım 9 - Agent'lar

Ham sayılar (notebooks/06_agentlar.ipynb, scripts/run_agents_demo.py, scripts/eval_planner.py):
- 6 agent: orchestrator, data, feature, scoring, rule, investigator; 8 görev. Message + MessageBus (Mediator), her mesaj izde
- Yetenek kaydı: görev, ön koşul (requires), girdi (inputs), çıktı (provides); orchestrator'da sabit görev tablosu yok
- Agent-to-agent: investigator -> feature_agent get_entity_profile; profil LLM açıklamasına giriyor ("5 farklı cihaz, 24 saatte 10 işlem")
- Planlayıcı v1 (tam liste) LLM %67 -> v2 (tek hedef görev, ön koşulları onarım ekler) %83 (12 istek)
- 16 istek (4 holdout): hybrid %88 | llm %75 | rules %81; holdout: rules %25, llm %50, hybrid %50
- Kalan LLM hatası: "Bu işlem dolandırıcılık mı?" -> genel politika sorusu sanılıyor
- Dinamik dağıtım: ALLOW kararında soruşturma atlanıyor (8,1 sn -> 0,4 sn); doğrulama hatasında akış duruyor; agent hatası hata cevabı
- Testin yakaladığı hatalar: onarım, işlem gerektiren ön koşulları geri ekliyordu (politika sorusu çöküyordu);
  feasible() döngüsel bağımlılıkta sonsuz özyineleme
- Tam akış süresi: doğrulama 18 ms, feature 105 ms, skor 280 ms, kural 12 ms, soruşturma ~7,5 sn (LLM)

Kararlar ve gerekçeler:
- Kendi hafif yapım: Bir agent framework'ü yerine küçük bir yapı kurdum: mesaj, bus ve iz tek yerde, design pattern'ler net,
  dış bağımlılık yok ve agent'ları stub'larla test edebiliyorum. Ölçekte kuyruk tabanlı bir yapıya geçilebilir, arayüz (gönder,
  yönlendir, iz) aynı kalır.
- Message bus (Mediator): Agent'lar birbirini tanımıyor, her mesaj bus'tan geçiyor. Yönlendirme, kayıt, süre ölçümü ve hata
  yakalama tek yerde; iz /explain cevabında da dönüyor.
- Yetenek kaydı: Her agent görevlerini, ön koşullarını, girdi ve çıktılarını beyan ediyor; orchestrator'da sabit bir görev
  tablosu yok, yeni agent eklemek yeni bir sınıf demek. Planlayıcı bu açıklamaları LLM'e araç listesi olarak da veriyor.
- Hibrit planlama: 16 istekte ölçtüm: hibrit %88, sadece LLM %75, sadece anahtar kelime %81. Anahtar kelimeler ayarlandığı
  12 istekte %100 ama görülmemiş 4 ifadede %25; LLM orada %50. Hibrit hem deterministik ve ucuz, hem belirsiz isteklerde LLM'e
  düşüyor. Planın nereden geldiği (rules / llm / default) izde görünüyor.
- Tek hedef görev: İlk sürümde LLM'den tam görev listesi istedim, 3B model gereksiz görevler ekliyordu (%67). Sadece hedef
  görevi seçtirip ön koşulları deterministik onarıma bıraktım (%83). Küçük modelde "listele" yerine "seç" daha iyi çalıştı.
- Plan onarımı: Bilinmeyen görev atılıyor, işlem yoksa işlem gerektiren görevler atılıyor, eksik ön koşullar ekleniyor, sıra
  topolojik, döngü tespiti var. Geçerli ama yanlış niyeti düzeltemiyor.
- Dinamik dağıtım: Karar ALLOW ise maliyetli LLM soruşturmasını atlıyorum (8,1 sn -> 0,5 sn), istenirse zorlanabiliyor.
  Doğrulama başarısızsa akış duruyor; bir agent hata verirse sistem çökmeden kısmi sonuç ve hata dönüyor.
- Agent-to-agent: Investigator, açıklamadan önce FeatureAgent'tan kullanıcı profilini istiyor (orchestrator'dan geçmeden,
  aynı konuşma kimliğiyle); profil LLM bağlamına giriyor.
- Karar yine kural motorunda: LLM sadece belirsiz isteğin planında ve açıklamada kullanılıyor.

## Adım 10 - API

Ham sayılar (tests/test_api.py, uvicorn + Swagger kontrolü):
- 8 endpoint: /health, /score, /explain, /rules, /rules/evaluate, /rules/reload, /rag/query, /agents/run
- dependency_injector container: 17 provider, ağırlar Singleton; Selector (embedder: ollama | tfidf); FRAUD_SETTINGS ile başka config
- Açılış (warm_up) 1,5 sn; /score ~0,7-0,8 sn (ID ile okuma 0,38 sn dahil); /rules/evaluate ~0,4 sn; /rag/query ~1-10 sn; /explain ~8-10 sn
- İşlem girişi: JSON (kısmi; eksik alan boş) veya transaction_id (Repository, merged.parquet; gerçekte DB)
- /rules/evaluate fields: skorlamadan kural what-if analizi; strateji istek başına
- Dayanıklılık: bileşen yüklenemezse API açılır, /health degraded; LLMError -> 503, artifact yok -> 503 (ipucuyla), ID yok -> 404, şema -> 422
- Bulunan hatalar: kısmi JSON'da dedektörlerin ham kolonları yoktu (KeyError) -> detector.input_columns() ile boş tamamlama;
  pd.NA object tipine çevirip float dönüşümünü bozuyordu -> np.nan; ~280 kolonu tek tek eklemek PerformanceWarning -> tek concat
- Testler: 14 API testi (DI override ile stub'lar; 1 gerçek artifact entegrasyon testi, artifact yoksa atlanır); toplam 122 test

Kararlar ve gerekçeler:
- dependency_injector container: Nesnelerin nasıl oluşturulduğu tek bir yerde. Ağır bileşenler (modeller, profil deposu,
  FAISS index, LLM istemcisi) Singleton, açılışta bir kez yükleniyor. Ayarlar Pydantic Settings'ten geliyor, FRAUD_SETTINGS ile
  başka bir config verilebiliyor. Embedder'ı config'e göre Selector seçiyor (bge-m3 ya da TF-IDF).
- İnce router'lar: İş mantığı servislerde (ScoringService, ExplainService); router sadece doğrulayıp servisi çağırıyor. Aynı
  servisleri agent'lar ve script'ler de kullanıyor.
- İki giriş yolu: Gerçek kullanımda işlem JSON olarak gelir; ama 435 alanı elle göndermek demo için pratik değil, bu yüzden
  transaction_id ile kayıtlı işlemi okuyan bir Repository de ekledim (prototipte parquet, gerçekte veritabanı). Kısmi JSON'da
  verilmeyen alanlar boş sayılıyor; bunun skoru değiştirebileceğini dokümanda belirttim.
- Dayanıklılık: Açılışta her bileşeni ayrı deniyorum; biri yüklenemezse (index yok, Ollama kapalı) API yine açılıyor,
  /health "degraded" diyor, ilgili endpoint 503 dönüyor. Skorlama LLM'e bağlı değil. Hata kodları: 422 doğrulama, 404 olmayan ID,
  503 LLM ya da artifact yok.
- Test: Container override ile servisleri, LLM'i ve RAG'i stub'larla değiştirip API'yi model ve Ollama olmadan test ediyorum;
  gerçek artifact'larla bir entegrasyon testi de var (artifact yoksa atlanıyor). Testler gerçek hatalar yakaladı: kısmi JSON'da
  dedektörlerin ham kolonları eksikti, pd.NA sayısal dönüşümü bozuyordu.
- Temiz kurulum: Repoyu boş bir klasöre kopyalayıp README'yi harfi harfine izledim: tüm adımlar çalıştı, sonuçlar birebir
  aynı çıktı, 122 test geçti.
