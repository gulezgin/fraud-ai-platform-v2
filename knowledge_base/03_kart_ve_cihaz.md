# Kart ve Cihaz Güvenliği Politikası

## FP-07 Yeni Kart Politikası

Kart aktivasyonundan sonraki ilk 7 gün içinde 500 USD üzerindeki işlemler yeni kart riski taşır ve FLAG olarak
izlenir. Bu süre içinde kart başına günlük toplam limit 1500 USD'dir. Ürün grubu W dışındaki işlemlerde kart yaşı
bilgisi güvenilir olmadığından bu politika yalnızca W ürün grubuna uygulanır.

## FP-09 Yeni Cihaz ve Hesap Ele Geçirme Şüphesi

Geçmişi olan bir kullanıcının ilk kez görülen bir cihazdan, geçmiş ortalama tutarının 3 katı veya üzerinde işlem
yapması hesap ele geçirme (account takeover) belirtisi olabilir. İşlem FLAG olarak izlenir ve yeni cihaz 72 saat
boyunca "doğrulanmamış cihaz" statüsünde tutulur. Bu sürede aynı cihazdan yapılan şifre veya e-posta değişikliği
talepleri reddedilir.

## FP-12 Cihaz Paylaşımı ve Kart Test Saldırısı

Aynı cihaz parmak izi üzerinden daha önce 5 veya daha fazla farklı kart kullanılmış olması kart test (card testing)
saldırısı göstergesidir. Anomali skoru da yüksekse (risk yüzdeliği 0,90 ve üzeri) işlem FLAG olarak izlenir ve cihaz
parmak izi 48 saat boyunca gözetim listesine alınır. Gözetim listesindeki bir cihazdan yeni bir kartla yapılan ilk
işlem otomatik olarak REVIEW'a gönderilir.
