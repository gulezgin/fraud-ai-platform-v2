# Velocity ve Otomatik Saldırı Politikası

## FP-03 İşlem Hızı (Velocity) Kontrolü

Bir kullanıcının son 1 saat içinde 10 veya daha fazla işlem yapması bot veya kart test saldırısı şüphesi olarak
değerlendirilir ve işlem bloke edilir. Bu kontrol yalnızca geçmişi 100 işlemden az olan kullanıcılara uygulanır;
yerleşik yüksek hacimli hesaplar FP-25 kapsamında değerlendirilir. Son 1 saatte 5 ile 9 arasında işlem yapan
kullanıcılar FLAG olarak izlemeye alınır. Velocity blokları sonrasında kullanıcının kartı 6 saat boyunca yeni
işleme kapatılır.

## FP-04 Ardışık İşlem ve Harcama Patlaması

Aynı kullanıcının önceki işleminden 60 saniyeden kısa süre sonra 200 USD üzeri yeni bir işlem yapması ardışık hızlı
işlem olarak işaretlenir. Son 24 saatte en az 5 işlemde toplam 2000 USD üzerinde harcama "harcama patlaması" kabul
edilir. Her iki durumda işlem FLAG olarak izlenir; aynı gün içinde aynı kullanıcıdan üçüncü bir FLAG gelirse günlük
harcama limiti, son 30 günlük ortalama günlük harcamanın 2 katı ile sınırlandırılır.
