# Genel Risk Değerlendirme Politikası

Bu doküman örnek (kurgusal) bir ödeme kuruluşunun fraud yönetimi politikalarını içerir. Kodlar kural motorundaki
`policy_ref` alanlarıyla eşleşir.

## FP-01 Model Tabanlı Anomali Alarmları

Anomali skoru, işlemin geçmiş davranışa göre ne kadar olağan dışı olduğunu gösterir ve tek başına dolandırıcılık
kanıtı sayılmaz. Risk yüzdeliği 0,97 ve üzerindeki işlemler (en riskli %3) fraud analistinin inceleme kuyruğuna
REVIEW olarak düşer. Risk yüzdeliği 0,999 ve üzerindeki işlemler, kullanıcının geçmişi 100 işlemden azsa otomatik
olarak bloke edilir (BLOCK). Model alarmıyla bloke edilen işlemlerde analist, blok kararını 2 saat içinde teyit etmek
veya kaldırmak zorundadır; teyit edilmeyen bloklar otomatik olarak kalkar.

## FP-30 Karar Seviyeleri ve İnceleme Süreleri

Sistem dört karar üretir. BLOCK: işlem reddedilir, kart geçici olarak dondurulur. REVIEW: işlem bekletilir ve
analist 4 saat içinde karar verir; 4 saat içinde karar verilmezse işlem onaylanır ve kayda "süre aşımı" notu düşülür.
FLAG: işlem onaylanır ancak 24 saat boyunca izleme listesinde tutulur, aynı müşteriden ikinci bir FLAG gelirse
REVIEW'a yükseltilir. ALLOW: işlem doğrudan onaylanır. Birden fazla kural çeliştiğinde en yüksek öncelikli kuralın
kararı uygulanır; eşitlikte daha ağır karar seçilir.

## FP-31 Müşteri İletişimi ve Doğrulama

REVIEW veya BLOCK kararı verilen işlemlerde müşteriye kayıtlı telefonuna SMS ile tek kullanımlık doğrulama kodu
gönderilir. Müşteri kodu 10 dakika içinde onaylarsa REVIEW kararı ALLOW'a çevrilebilir; BLOCK kararları için SMS
onayı yeterli değildir, müşteri çağrı merkezi üzerinden kimlik doğrulaması yapmalıdır. Müşteriye hiçbir koşulda
hangi kuralın tetiklendiği veya risk skoru bildirilmez.
