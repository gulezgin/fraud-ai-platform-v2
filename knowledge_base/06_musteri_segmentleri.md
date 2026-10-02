# Müşteri Segmentleri Politikası

## FP-20 Güvenilir Müşteri Beyaz Listesi

Kartı en az 180 günlük olan, en az 30 gündür işlem geçmişi bulunan ve bilinen bir cihazdan işlem yapan müşterilerin
100 USD altındaki işlemleri doğrudan onaylanır (ALLOW). Beyaz liste yalnızca izleme (FLAG) seviyesindeki kuralları
geçersiz kılar; REVIEW ve BLOCK kararlarını kaldıramaz. Böylece ele geçirilmiş güvenilir bir hesapta model alarmı
susturulmaz.

## FP-21 Yüksek Değerli Müşteriler

Geçmiş ortalama işlem tutarı müşteri tabanının ilk %10'unda olan müşteriler yüksek değerli müşteri sayılır. Bu
müşterilerin yüksek tutarlı işlemleri için anomali skoru 0,8 çarpanıyla düşürülür ve REVIEW kararlarında öncelikli
analist kuyruğu kullanılır (en geç 1 saat içinde karar).

## FP-25 Onaylı Yüksek Hacimli Hesaplar

Abonelik, entegrasyon veya toptan alım yapan hesaplar kısa sürede çok sayıda işlem üretebilir. Geçmişi 100 işlemi
aşan hesaplar velocity ve model tabanlı otomatik blok kurallarından muaftır; bu hesaplarda yüksek riskli işlemler
blok yerine REVIEW'a gönderilir. Muafiyetin gerekçesi, tek bir meşru hesabın yüzlerce işleminin toplu olarak bloke
edilmesini önlemektir. Fraud ekibi bu hesapları her çeyrekte yeniden gözden geçirir.
