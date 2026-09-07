"""
eval/scripts/run_ragas_eval.py

Phase C2 — RAGAS faithfulness (+ optional answer_relevancy) with Groq judge.

Requires:
  - ragas==0.3.9
  - eval/scripts/_ragas_import_fix.py  (fixes langchain_community VertexAI import)

Usage:
  python eval/scripts/run_ragas_eval.py --lang hi --limit 10 --faithfulness-only
  python eval/scripts/run_ragas_eval.py --lang en --limit 10 --faithfulness-only
  python eval/scripts/run_ragas_eval.py --limit 15 --faithfulness-only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# RAGAS import fix — MUST run before importing ragas
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _ragas_import_fix import apply as _apply_ragas_fix

_apply_ragas_fix()
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from pipeline.generation.generate import generate_answer
from pipeline.retrieval.retriever import retrieve

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_dataset" / "golden_questions.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_golden(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else (data.get("items") or [])


def build_rows(items: list[dict]) -> list[dict]:
    rows = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        lang = item.get("language") or "en"
        print(f"[{i}/{len(items)}] {item.get('id', i)} ({lang})")

        chunks = retrieve(query=q, language=lang, top_k=5)
        contexts = [c["text"] for c in chunks if c.get("text")]

        gen = generate_answer(query=q, retrieved_chunks=chunks, language=lang)
        answer = gen.get("answer") or ""

        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "id": item.get("id"),
                "language": lang,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", choices=["en", "hi", "mr"], default=None)
    parser.add_argument(
        "--faithfulness-only",
        action="store_true",
        help="Only run faithfulness (skip answer_relevancy / embeddings)",
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY missing in .env")
        sys.exit(1)

    if not GOLDEN_PATH.exists():
        print(f"ERROR: golden file not found: {GOLDEN_PATH}")
        sys.exit(1)

    items = [i for i in load_golden(GOLDEN_PATH) if i.get("question")]
    if args.lang:
        items = [i for i in items if i.get("language") == args.lang]
    if args.limit:
        items = items[: args.limit]

    if not items:
        print("No items to evaluate after filters.")
        sys.exit(1)

    print(f"Running pipeline on {len(items)} items...")
    rows = build_rows(items)

    judge = ChatOpenAI(
        model=os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
    )
    ragas_llm = LangchainLLMWrapper(judge)

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
        }
    )

    metrics = [faithfulness]
    if not args.faithfulness_only:
        metrics.append(answer_relevancy)

    print("Running RAGAS evaluate()...")
    result = evaluate(dataset, metrics=metrics, llm=ragas_llm)

    try:
        summary = dict(result)
    except Exception:
        summary = {"result": str(result)}

    try:
        per_row = result.to_pandas().to_dict(orient="records")
    except Exception:
        per_row = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"ragas_eval_{ts}.json"
    payload = {
        "summary": summary,
        "n": len(rows),
        "lang_filter": args.lang,
        "faithfulness_only": args.faithfulness_only,
        "per_row": per_row,
        "meta_rows": [{"id": r["id"], "language": r["language"]} for r in rows],
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("RAGAS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()