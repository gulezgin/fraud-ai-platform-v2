"""Görev planlayıcı: kullanıcı isteğinden hangi agent görevlerinin hangi sırayla çalışacağına karar verir.

LLM'e mevcut görevler (agent'ların beyan ettiği yetenekler) ve istek verilir, JSON plan alınır. Küçük model hatalı plan
üretebileceği için plan doğrulanır ve onarılır: bilinmeyen görev atılır, işlem yoksa işlem gerektiren görev atılır,
eksik ön koşullar eklenir, sıra bağımlılıklara göre düzeltilir. LLM'e ulaşılamaz veya plan boş kalırsa anahtar kelimeli
yedek plan kullanılır. Hangi yoldan gelindiği (source) ve yapılan onarımlar izde görünür.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from fraud_platform.agents.base import Capability
from fraud_platform.llm.base import BaseLLM, LLMError

# v1 tam görev listesi istiyordu; 3B model gereksiz görev ekliyordu (12 istekte %67 doğru). v2 tek bir "hedef görev"
# seçtiriyor (niyet sınıflandırması), ön koşulları onarım ekliyor. Ölçüm: scripts/eval_planner.py
PLAN_PROMPT = """Bir fraud analiz sisteminde şu görevler var. Bir görev seçildiğinde ön koşulları otomatik çalışır.
{tasks}

Kullanıcı isteği: "{goal}"
{tx_note}

Bu isteği karşılayan TEK görevi seç: isteğin asıl sorduğu şeyi üreten görev. Daha fazlasını seçme.
Sadece JSON döndür: {{"task": "görev_adı", "reason": "kısa Türkçe gerekçe"}}"""

EXPLAIN_WORDS = re.compile(r"açıkla|neden|niye|gerekçe|soruştur|incele|anlat|politika.*göre|detay", re.IGNORECASE)
DECISION_WORDS = re.compile(r"karar|kural|onay|blok|engelle", re.IGNORECASE)
SCORE_WORDS = re.compile(r"skor|risk|puan|anomali", re.IGNORECASE)
FEATURE_WORDS = re.compile(r"feature|özellik", re.IGNORECASE)
VALIDATE_WORDS = re.compile(r"geçerli|doğrula|eksik alan|alanlar", re.IGNORECASE)
DATA_WORDS = re.compile(r"veri ?set|veri kalite|dataset|kaç satır|eksik veri|eğitim veri", re.IGNORECASE)


@dataclass
class Plan:
    tasks: list[str]
    source: str                                   # rules | llm | default
    reason: str = ""
    repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tasks": self.tasks, "source": self.source, "reason": self.reason, "repairs": self.repairs}


class TaskPlanner:
    """mode:
    - hybrid: önce anahtar kelime yönlendiricisi; eşleşme yoksa LLM; LLM yoksa / başarısızsa varsayılan
    - llm: önce LLM; başarısızsa anahtar kelime, o da yoksa varsayılan
    - rules: sadece anahtar kelime + varsayılan
    """

    MODES = ("hybrid", "llm", "rules")

    def __init__(self, llm: BaseLLM | None = None, mode: str = "hybrid"):
        if mode not in self.MODES:
            raise ValueError(f"bilinmeyen planlama modu '{mode}', geçerli: {self.MODES}")
        self.llm = llm
        self.mode = mode

    def plan(self, goal: str, caps: dict[str, Capability], has_transaction: bool) -> Plan:
        notes = []
        order = {"hybrid": ("rules", "llm"), "llm": ("llm", "rules"), "rules": ("rules",)}[self.mode]
        for step in order:
            if step == "rules":
                task = self.route(goal, has_transaction)
                if task:
                    plan = self.repair([task], caps, has_transaction, source="rules")
                    if plan.tasks:
                        plan.reason = f"anahtar kelime eşleşmesi -> {task}"
                        return plan
                notes.append("anahtar kelime eşleşmedi")
            elif self.llm is None:
                notes.append("LLM tanımlı değil")
            else:
                try:
                    out = self.llm.generate_json(self._prompt(goal, caps, has_transaction))
                    task = out.get("task")
                    plan = self.repair([task.strip()] if isinstance(task, str) else [], caps, has_transaction)
                    if plan.tasks:
                        plan.reason = str(out.get("reason", ""))
                        return plan
                    notes.append(f"LLM geçersiz görev seçti: {task}")
                except LLMError as e:
                    notes.append(f"LLM'e ulaşılamadı: {e}")
        plan = self.repair([self.default(has_transaction)], caps, has_transaction, source="default")
        plan.reason = "; ".join(notes)
        return plan

    @staticmethod
    def _prompt(goal: str, caps: dict[str, Capability], has_transaction: bool) -> str:
        tasks = "\n".join(f"- {name}: {c.description}" + (f" (önce: {', '.join(c.requires)})" if c.requires else "")
                          for name, c in caps.items())
        tx_note = "İstekle birlikte bir işlem (transaction) verildi." if has_transaction else "İstekte işlem yok."
        return PLAN_PROMPT.format(tasks=tasks, goal=goal, tx_note=tx_note)

    @staticmethod
    def route(goal: str, has_transaction: bool) -> str | None:
        """Deterministik niyet yönlendirici; eşleşme yoksa None (karar LLM'e kalır)."""
        if not has_transaction:
            return "dataset_summary" if DATA_WORDS.search(goal) else None
        for pattern, task in [(EXPLAIN_WORDS, "investigate"), (DECISION_WORDS, "evaluate_rules"), (SCORE_WORDS, "score"),
                              (FEATURE_WORDS, "build_features"), (VALIDATE_WORDS, "validate_transaction")]:
            if pattern.search(goal):
                return task
        return None

    @staticmethod
    def default(has_transaction: bool) -> str:
        # işlem varsa tam değerlendirme (ön koşullar onarımla eklenir), yoksa politika sorusu
        return "investigate" if has_transaction else "answer_policy_question"

    @staticmethod
    def feasible(task: str, caps: dict[str, Capability], has_transaction: bool, stack: tuple[str, ...] = ()) -> bool:
        """Görev ve tüm ön koşulları mevcut girdiyle çalışabilir mi (ör. işlem yoksa score da yapılamaz)."""
        if task in stack:
            raise ValueError(f"döngüsel bağımlılık: {' -> '.join((*stack, task))}")
        if task not in caps:          # ön koşulu hiçbir agent sağlamıyor
            return False
        c = caps[task]
        if c.needs_transaction and not has_transaction:
            return False
        return all(TaskPlanner.feasible(dep, caps, has_transaction, (*stack, task)) for dep in c.requires)

    @staticmethod
    def repair(raw: list[str], caps: dict[str, Capability], has_transaction: bool, source: str = "llm") -> Plan:
        repairs, kept = [], []
        for t in dict.fromkeys(raw):
            if t not in caps:
                repairs.append(f"bilinmeyen görev atıldı: {t}")
            elif not TaskPlanner.feasible(t, caps, has_transaction):
                repairs.append(f"işlem olmadığı için atıldı: {t}")
            else:
                kept.append(t)

        ordered: list[str] = []

        def visit(t: str, stack: tuple[str, ...] = ()) -> None:
            if t in ordered:
                return
            if t in stack:
                raise ValueError(f"döngüsel bağımlılık: {' -> '.join((*stack, t))}")
            for dep in caps[t].requires:
                if dep not in kept and dep not in ordered:
                    repairs.append(f"ön koşul eklendi: {dep} ({t} için)")
                visit(dep, (*stack, t))
            ordered.append(t)

        for t in kept:
            visit(t)
        if [t for t in ordered if t in kept] != kept:
            repairs.append("sıra bağımlılıklara göre düzeltildi")
        return Plan(ordered, source, repairs=repairs)
