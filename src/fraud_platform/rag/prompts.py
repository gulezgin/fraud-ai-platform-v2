"""Prompt şablonları. Küçük yerel model (3B) için kısa, net talimat ve sabit çıktı biçimi.

v1'de "bilgi yoksa söyle" kaçışı küçük model için fazla çekiciydi: bağlamda cevap varken bile kullanıyordu.
v2: önce ilgili politikayı bulup sayı/süre/eşikleri aynen aktarması isteniyor, kaçış sadece hiçbir politika ilgili
değilse. Değerlendirme (scripts/eval_rag.py): kurgusal bilgi aktarımı %50 -> %72, doğru atıf %89 -> %94 (sıcaklık 0).
"""

SYSTEM = (
    "Sen bir ödeme kuruluşunda fraud analistisin. Soruyu SADECE verilen politika metinlerini kullanarak Türkçe cevapla. "
    "Önce soruyla ilgili politikayı bul; o politikadaki somut sayıları, süreleri, eşikleri ve yapılacak işlemleri aynen "
    "aktar. Politikada yazmayan hiçbir bilgi ekleme. Her cümlenin sonuna dayandığın politika kodunu köşeli parantezle yaz, "
    "örneğin [FP-07]. Verilen politikaların HİÇBİRİ soruyla ilgili değilse sadece 'Politikalarda bu konuda bilgi yok.' yaz."
)

QA = """### Politikalar
{context}

### Soru
{question}

### Cevap
2-3 cümle. Politikadaki sayı ve süreleri aynen kullan. Her cümlenin sonunda [FP-xx]."""

EXPLAIN = """### Politikalar
{context}

### İşlem değerlendirmesi
{summary}

### Görev
Kararı kural motoru verdi: {decision}. Senin görevin karar vermek değil, bu kararın NEDENİNİ analiste politikalara
dayanarak açıklamak. Tetiklenen kural yoksa bunu söyle ve risk yüzdeliğine göre neden {decision} olduğunu açıkla.
Politikadaki somut süre ve adımları aynen aktar; işlem bilgisinde olmayan bir şey uydurma.
recommended_action kural motorunun kararıyla aynı olmalı; yalnızca bir politika açıkça farklı bir işlem gerektiriyorsa değiştir.
Aşağıdaki JSON biçiminde cevap ver:
{{"explanation": "2-4 cümle Türkçe açıklama, her cümlenin sonunda [FP-xx]",
  "recommended_action": "{decision}",
  "next_steps": ["politikadan somut bir adım"],
  "citations": ["FP-xx"]}}"""

NO_CONTEXT = """### Soru
{question}

### Cevap (kısa)"""


def format_context(chunks) -> str:
    return "\n\n".join(f"[{c.policy_code or c.source}] {c.title}\n{c.text}" for c in chunks)
