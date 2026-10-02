"""Multi-agent demo: doğal dildeki isteği orchestrator'a verir, planı, adımları, kararı ve mesaj izini basar.

    python scripts/run_agents_demo.py --id 3554906 --goal "Bu işlemi değerlendir ve kararın nedenini açıkla"
    python scripts/run_agents_demo.py --id 3554906 --goal "Sadece risk skorunu ver"
    python scripts/run_agents_demo.py --goal "Kart test saldırısında cihaz ne kadar süre gözetimde kalır?"

İz artifacts/agent_traces/<conversation_id>.json olarak kaydedilir.
"""
import argparse
import json
import logging
import time

import pandas as pd

from fraud_platform.agents.factory import build_agent_system
from fraud_platform.config import get_settings
from fraud_platform.rag.factory import build_llm, load_pipeline
from fraud_platform.rules.engine import RuleEngine
from fraud_platform.services.scoring_service import ScoringService


def load_transaction(settings, tx_id: int) -> dict:
    raw = pd.read_parquet(settings.merged_path, filters=[(settings.data.id_column, "==", tx_id)])
    if raw.empty:
        raise SystemExit(f"{tx_id} bulunamadı")
    row = raw.drop(columns=[settings.data.target_column]).iloc[0]
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--id", type=int, default=None, help="TransactionID (yoksa işlemsiz istek)")
    parser.add_argument("--force-investigation", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    s = get_settings()
    llm = build_llm(s)
    orchestrator = build_agent_system(s, ScoringService.from_artifacts(s), RuleEngine(s.rules.rules_file),
                                      load_pipeline(s, llm), llm)
    tx = load_transaction(s, args.id) if args.id else None

    start = time.time()
    result = orchestrator.run(args.goal, transaction=tx, force_investigation=args.force_investigation)
    elapsed = time.time() - start

    print(f"\nİstek: {args.goal}\nDurum: {result['status']}  ({elapsed:.1f} sn)")
    plan = result.get("plan", {})
    print(f"Plan [{plan.get('source')}]: {' -> '.join(plan.get('tasks', []))}  ({plan.get('reason', '')})")
    for r in plan.get("repairs", []):
        print(f"  onarım: {r}")
    print("\nMesaj izi:")
    for e in result["trace"]:
        arrow = "->" if e["kind"] == "request" else "<-"
        a, b = (e["from"], e["to"]) if e["kind"] == "request" else (e["to"], e["from"])
        ms = f"{e['duration_ms']:>8.1f} ms" if e["duration_ms"] is not None else ""
        print(f"  {a:>18s} {arrow} {b:<18s} {e['task']:<24s} {e['status']:<5s} {ms}")
    for step in result.get("steps", []):
        if step["status"] == "skipped":
            print(f"\n  atlandı: {step['task']} ({step['reason']})")

    if result.get("decision"):
        print(f"\nKarar: {result['decision']}  risk yüzdeliği {result['risk_percentile']}")
    if result.get("investigation"):
        inv = result["investigation"]
        print(f"Açıklama: {inv['explanation']}\nAtıflar: {inv['citations']} doğrulandı: {inv['citations_verified']}")
    if result.get("answer"):
        print(f"Cevap: {result['answer']['answer']}")
    if result.get("dataset"):
        print(f"Veri seti: {result['dataset']}")
    if result.get("error"):
        print(f"Hata: {result['error']}")

    out = s.paths.artifacts_dir / "agent_traces" / f"{result['conversation_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nİz kaydedildi: {out}")


if __name__ == "__main__":
    main()
