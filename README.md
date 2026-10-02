# Fraud AI Platform

IEEE-CIS Fraud Detection verisi üzerinde geliştirdiğim fraud / anomali tespit platformu prototipi. İçinde veri profilleme,
sadece geçmiş verisinden hesaplanan feature'lar, dört katmanlı anomali tespiti, skor birleştirme, iş bağlamına göre
düzeltme, YAML tabanlı kural motoru, politika dokümanları üzerinde RAG ve multi-agent orkestrasyonu var. Hepsi FastAPI ile
sunuluyor; LLM ve embedding modelleri Ollama ile yerelde çalışıyor.

Bir işlem sisteme girdiğinde sonunda bir risk skoru, bir karar (BLOCK / REVIEW / FLAG / ALLOW) ve bu kararın politikalara
dayanan bir açıklaması çıkıyor.

Tasarım kararları, ölçümler ve sınırlamalar için: [docs/TECHNICAL.md](docs/TECHNICAL.md)

## Sonuçlar

Aşağıdaki sayılar test döneminden; yani verinin zaman olarak son %20'si, modellerin eğitimde hiç görmediği kısım.

- Context düzeltmesinden sonraki skor: PR-AUC 0,125, ROC-AUC 0,778 (fraud oranı %3,4). En iyi tek katmanı geçiyor.
- Context düzeltmesi, aynı alarm sayısında (%3) 95 yanlış alarmın yerine gerçek fraud koyuyor.
- Kural motoru: 12 aktif kural, 3 çakışma stratejisi. BLOCK isabeti valid'de %67,6, testte %34,1.
- RAG (bge-m3 + qwen2.5:3b): doğru politika ilk 3 sonuçta %100, doğru atıf %94, bilgi tabanında olmayan sorularda "bilgi
  yok" %100.
- Agent planlayıcı 16 istekte %88 doğru plan çıkarıyor.
- 200 gerçek işlemde canlı ve toplu feature hesabı arasında fark yok.
- 122 test var; çoğu Ollama ya da veri gerektirmiyor.

## Gereksinimler

- Python 3.11 veya üstü (3.12 ile geliştirdim)
- 16 GB RAM önerilir (tam veride ölçtüğüm tepe bellek: `prepare_data.py` 5,0 GB, `train.py` 6,7 GB)
- ~2 GB disk (ham veri 0,7 GB, işlenmiş veri ve modeller 0,3 GB, Python paketleri 0,85 GB), Ollama modelleri için ayrıca
  ~3,1 GB
- [Ollama](https://ollama.com): sadece RAG, agent'lar ve açıklama için gerekli; skorlama ve kurallar LLM olmadan çalışıyor

## Kurulum

```bash
git clone https://github.com/gulezgin/fraud-ai-platform-v2.git
cd fraud-ai-platform-v2
python -m venv .venv
```

Sanal ortamı aktif etmek için Windows'ta `.venv\Scripts\activate`, Mac/Linux'ta `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
pip install -e .
```

## Veri

[Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) sayfasında yarışma kurallarını kabul
edip `train_transaction.csv` ve `train_identity.csv` dosyalarını `data/raw/` klasörüne koyun. Veri lisans nedeniyle repoda yok.

## Yerel LLM

Ollama kurulduktan sonra iki modeli indirin:

```bash
ollama pull qwen2.5:3b
```
```bash
ollama pull bge-m3
```

`qwen2.5:3b` (~1,9 GB) cevapları üreten model, `bge-m3` (~1,2 GB) çok dilli embedding modeli. İkisi 4 GB'lık bir ekran kartına
birlikte sığıyor; ekran kartı yoksa CPU'da daha yavaş çalışıyor.

## Çalıştırma

Script'leri sırayla çalıştırın. Süreler 8 çekirdekli, 32 GB RAM'li bir dizüstü bilgisayarda ölçüldü.

```bash
python scripts/prepare_data.py
```
Tabloları birleştirir, şemayı, kalite raporunu ve profili çıkarır, 55 feature üretir (~2 dk). Çıktılar `data/processed/` ve
`artifacts/` altına yazılır.

```bash
python scripts/train.py
```
4 anomali dedektörünü ve skor birleştiriciyi eğitir, profil deposunu oluşturur, valid ve test sonuçlarını yazdırır (~1 dk).

```bash
python scripts/build_index.py
```
Politika dokümanlarını bge-m3 ile vektöre çevirip FAISS index'i oluşturur (birkaç saniye, Ollama açık olmalı).

Tek bir işlemi baştan sona skorlayıp kural kararını görmek için:

```bash
python scripts/score_transaction.py 3554906
```

İsteğe bağlı değerlendirme script'leri:

- `python scripts/calibrate_context.py`: context kurallarının kanıt tablosu (koşullu lift, önerilen çarpan)
- `python scripts/rule_report.py`: kural başına tetiklenme, isabet ve karar dağılımı (valid ve test)
- `python scripts/eval_rag.py`: RAG değerlendirmesi (bge-m3 ve TF-IDF karşılaştırması, bağlamlı ve bağlamsız cevaplar)
- `python scripts/eval_planner.py`: agent planlayıcının doğruluğu
- `python scripts/run_agents_demo.py --id 3554906 --goal "Bu işlemi değerlendir ve kararın nedenini açıkla"`: agent'ların
  planı, adımları ve mesaj izi

## API

```bash
uvicorn fraud_platform.api.main:app --port 8000
```

Açılışta modeller, profil deposu ve FAISS index bir kez yükleniyor (birkaç saniye); hazır olunca `/health` cevap veriyor.
Swagger arayüzü <http://localhost:8000/docs> adresinde, her endpoint'te hazır bir örnek istek var. İşlem iki şekilde
verilebilir: `transaction` alanıyla JSON olarak (verilmeyen alanlar boş sayılır) ya da `transaction_id` ile (kayıtlı işlemin
bütün alanları veri setinden okunur). Skorlamada sadece işlemden önceki geçmiş kullanılıyor.

| Endpoint | Ne yapar |
|---|---|
| `GET /health` | Bileşenler yüklendi mi, LLM erişilebilir mi |
| `POST /score` | 4 katmanın skoru, birleşik skor, context düzeltmesi, gerekçeler |
| `POST /explain` | Skor, en etkili feature'lar, kurallar, RAG açıklaması ve agent izi |
| `POST /rules/evaluate` | Karar, tetiklenen kurallar ve çakışma çözümü (`fields` ile skorlamadan deneme yapılabilir) |
| `GET /rules`, `POST /rules/reload` | Kural seti; YAML dosyasını kod değişmeden yeniden yükleme |
| `POST /rag/query` | Politika sorusuna kaynak gösteren cevap |
| `POST /agents/run` | Doğal dilde bir isteği agent'lara verir |

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{"transaction_id": 3554906}'
```
```bash
curl -X POST localhost:8000/explain -H "Content-Type: application/json" -d '{"transaction_id": 3554906}'
```
```bash
curl -X POST localhost:8000/rules/evaluate -H "Content-Type: application/json" -d '{"fields": {"risk_percentile": 0.995, "user_tx_count": 4, "tx_count_1h": 12, "TransactionAmt": 820, "is_night": 1}, "strategy": "most_severe"}'
```
```bash
curl -X POST localhost:8000/rag/query -H "Content-Type: application/json" -d '{"question": "Kart test saldırısında cihaz ne kadar süre gözetimde kalır?"}'
```
```bash
curl -X POST localhost:8000/agents/run -H "Content-Type: application/json" -d '{"goal": "Sadece risk skorunu ver", "transaction_id": 3554906}'
```

Bu `curl` örnekleri bash için (Mac/Linux, Windows'ta Git Bash ya da WSL). Windows PowerShell'de `curl` başka bir komutun takma
adı ve tırnakları farklı işliyor; orada Swagger arayüzünü kullanmak daha kolay.

Ollama kapalıysa API yine açılıyor: skorlama ve kurallar çalışıyor, LLM gerektiren endpoint'ler 503 dönüyor, `/health`
durumu gösteriyor.

## Notebook'lar

- `01_veri_yukleme_kalite`: birleştirme, otomatik şema, eksik veri, kalite, dağılımlar (Adım 1)
- `02_profiling`: zaman, aykırı değerler, boşluk sinyali, nadir kombinasyonlar, korelasyon, kullanıcı tanımı (Adım 2)
- `03_feature_analizi`: sızıntı kararı, feature listesi, feature'ların fraud ile ilişkisi (Adım 3)
- `04_degerlendirme`: dedektörler, skor birleştirme, context ve kural motoru (Adım 4-7)
- `05_rag`: bilgi tabanı, retrieval ve cevap değerlendirmesi, işlem açıklaması (Adım 8)
- `06_agentlar`: agent'lar, planlayıcı değerlendirmesi, örnek senaryolar (Adım 9)

Notebook'lar çıktılarıyla birlikte kayıtlı. Yeniden çalıştırmak için önce yukarıdaki script'lerin çalışmış olması gerekiyor.

## Testler

```bash
pytest
```

Testler için Ollama ya da eğitilmiş model gerekmiyor (sahte LLM, TF-IDF embedder ve container override kullanılıyor). Gerçek
model dosyalarıyla çalışan entegrasyon testi, dosyalar yoksa atlanıyor.

## Proje yapısı

```
config/            settings.yaml (yollar, eşikler, ağırlıklar), context_rules.yaml, rules.yaml
knowledge_base/    kurgusal fraud politikaları ve eval/ altında değerlendirme soruları
notebooks/         analiz ve değerlendirme notebook'ları
scripts/           veri hazırlığı, eğitim, index, değerlendirme, demo
src/fraud_platform/
  config.py        ayarlar (Pydantic)
  data/            yükleme, şema, kalite, profilleme
  features/        feature pipeline ve tek işlem için feature üretimi
  store/           kullanıcı profil deposu
  detection/       4 dedektör ve skor birleştirici
  conditions/      context ve kural motorunun ortak koşul dili
  context/         context düzeltmesi ve kalibrasyon
  rules/           kural motoru ve çakışma stratejileri
  llm/, rag/       Ollama istemcisi, parçalama, embedding, FAISS, RAG
  agents/          mesajlar, bus, agent'lar, planlayıcı, orchestrator
  services/        skorlama ve açıklama servisleri, işlem deposu
  api/             FastAPI uygulaması
  container.py     dependency_injector container'ı
tests/             testler
docs/              TECHNICAL.md, notes.md (çalışma notlarım), images/
```
