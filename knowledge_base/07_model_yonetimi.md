# Model ve Kural Yönetimi Politikası

## FP-40 Model Performans İzleme

Anomali modeli ve kural seti haftalık olarak izlenir. Alarm bütçesindeki (en riskli %3) isabet oranı üst üste iki
hafta %15'in altına düşerse model yeniden eğitilir. Skor dağılımındaki kayma, eğitim dönemine göre population
stability index (PSI) ile ölçülür; PSI 0,25'i aşarsa model risk komitesine bildirilir.

## FP-41 Blok Devre Kesici

Bir saatlik BLOCK kararı sayısı son 30 günün saatlik ortalamasının 5 katını aşarsa devre kesici devreye girer:
otomatik BLOCK kararları 2 saat boyunca REVIEW'a düşürülür ve nöbetçi analiste bildirim gönderilir. Bu mekanizma
tek bir kuralın veya veri hatasının çok sayıda meşru işlemi bloke etmesini engeller.

## FP-42 Kural Değişiklik Yönetimi

Yeni bir kural veya eşik değişikliği, en az 4 haftalık geçmiş veri üzerinde isabet ve kapsam ölçülmeden devreye
alınamaz. Kural bir analist tarafından yazılır, ikinci bir analist tarafından onaylanır ve kural dosyası sürüm
kontrolünde tutulur. Devreye alınan her kuralın ilk 2 haftası "gölge mod"da çalıştırılır: karar üretir ama uygulanmaz,
sadece kaydedilir.
