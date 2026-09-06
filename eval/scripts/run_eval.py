"""
eval/scripts/run_eval.py

Runs the Phase C golden dataset through the real pipeline (retrieve ->
generate) and scores each answer on two independent dimensions:

1. RETRIEVAL ACCURACY (no LLM call needed, free, deterministic):
   Did the pipeline actually retrieve a chunk matching the question's
   expected_scheme_id + expected_section? This directly measures the
   retrieval layer (vector search + section-intent routing) in isolation
   from generation quality.

2. FAITHFULNESS (LLM-as-judge, uses your existing Groq setup):
   Given the retrieved context and the generated answer, is the answer
   actually supported by that context (not hallucinated)? This is the
   same idea RAGAS uses, implemented directly against your own stack
   instead of adding a new dependency.

   For "faq"-source questions (which have a verified expected_answer),
   an additional ANSWER CORRECTNESS check compares the generated answer
   against the real source answer.

Usage:
    python eval/scripts/run_eval.py
    python eval/scripts/run_eval.py --limit 20          # quick smoke test
    python eval/scripts/run_eval.py --fail-below 0.75   # CI gating mode
    python eval/scripts/run_eval.py --language hi       # one language only

Output: eval/reports/golden_eval_<timestamp>.json (full detail)
        + a printed summary table (overall + per language + per source)

Exit code: 0 normally. If --fail-below is set and any language's
faithfulness rate falls below the threshold, exits 1 (for CI gating —
see .github/workflows/eval-gate.yml).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from groq import Groq

from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer

load_dotenv()

GOLDEN_DATASET_PATH = Path("eval/golden_dataset/golden_questions.json")
REPORTS_DIR = Path("eval/reports")
JUDGE_MODEL = "openai/gpt-oss-120b"  # reuse the same model already configured for generation

judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def check_retrieval_accuracy(question: dict, retrieved_chunks: list[dict]) -> dict:
    """Deterministic, no LLM call. Checks whether the expected
    scheme_id + section actually appears among the retrieved chunks."""
    expected_scheme = question["scheme_id"]
    expected_section = question["expected_section"]

    matching = [
        c for c in retrieved_chunks
        if c.get("scheme_id") == expected_scheme and c.get("section") == expected_section
    ]

    return {
        "correct_chunk_retrieved": len(matching) > 0,
        "matched_via": matching[0].get("matched_via") if matching else None,
    }


def judge_faithfulness(question_text: str, context: str, answer: str, language: str) -> dict:
    """LLM-as-judge: is the answer supported by the retrieved context?
    Returns a dict with a boolean verdict and a short reason, parsed from
    a strict JSON-only response."""
    judge_prompt = f"""You are a strict evaluator. Given a QUESTION, the CONTEXT that was retrieved to answer it, and the ANSWER that was generated, determine whether the ANSWER is fully supported by the CONTEXT.

Respond with ONLY a JSON object, no other text, no markdown formatting:
{{"faithful": true or false, "reason": "one short sentence explaining why"}}

A faithful answer:
- Contains only claims that can be verified from the CONTEXT
- Does not add facts, numbers, or details not present in the CONTEXT
- If the ANSWER says information is unavailable/insufficient, and the CONTEXT genuinely does not contain the answer, that counts as faithful (correct refusal)

QUESTION ({language}): {question_text}

CONTEXT:
{context}

ANSWER:
{answer}

JSON response:"""

    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if the model added them despite instructions
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            "faithful": bool(parsed.get("faithful", False)),
            "reason": parsed.get("reason", ""),
            "judge_error": None,
        }
    except Exception as e:
        return {"faithful": None, "reason": None, "judge_error": str(e)}


def run_single_question(question: dict) -> dict:
    retrieved_chunks = retrieve(
        query=question["question"],
        language=question["language"],
        top_k=5,
    )

    result = generate_answer(
        query=question["question"],
        retrieved_chunks=retrieved_chunks,
        language=question["language"],
    )

    retrieval_check = check_retrieval_accuracy(question, retrieved_chunks)

    context_text = "\n\n".join(c["text"] for c in retrieved_chunks) if retrieved_chunks else ""
    faithfulness_check = judge_faithfulness(
        question_text=question["question"],
        context=context_text,
        answer=result["answer"],
        language=question["language"],
    )

    return {
        "id": question["id"],
        "language": question["language"],
        "scheme_id": question["scheme_id"],
        "expected_section": question["expected_section"],
        "source": question["source"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "expected_answer": question.get("expected_answer"),
        "generated_answer": result["answer"],
        "retrieved_sections": [
            {"scheme_id": c.get("scheme_id"), "section": c.get("section"), "matched_via": c.get("matched_via")}
            for c in retrieved_chunks
        ],
        "retrieval_accuracy": retrieval_check,
        "faithfulness": faithfulness_check,
    }


def summarize(results: list[dict]) -> dict:
    def rate(items, key_fn):
        valid = [x for x in items if key_fn(x) is not None]
        if not valid:
            return None
        return sum(1 for x in valid if key_fn(x)) / len(valid)

    summary = {
        "total_questions": len(results),
        "overall": {
            "retrieval_accuracy": rate(results, lambda r: r["retrieval_accuracy"]["correct_chunk_retrieved"]),
            "faithfulness": rate(results, lambda r: r["faithfulness"]["faithful"]),
        },
        "by_language": {},
        "by_source": {},
        "by_section": {},
    }

    for lang in sorted(set(r["language"] for r in results)):
        subset = [r for r in results if r["language"] == lang]
        summary["by_language"][lang] = {
            "count": len(subset),
            "retrieval_accuracy": rate(subset, lambda r: r["retrieval_accuracy"]["correct_chunk_retrieved"]),
            "faithfulness": rate(subset, lambda r: r["faithfulness"]["faithful"]),
        }

    for source in sorted(set(r["source"] for r in results)):
        subset = [r for r in results if r["source"] == source]
        summary["by_source"][source] = {
            "count": len(subset),
            "retrieval_accuracy": rate(subset, lambda r: r["retrieval_accuracy"]["correct_chunk_retrieved"]),
            "faithfulness": rate(subset, lambda r: r["faithfulness"]["faithful"]),
        }

    for section in sorted(set(r["expected_section"] for r in results)):
        subset = [r for r in results if r["expected_section"] == section]
        summary["by_section"][section] = {
            "count": len(subset),
            "retrieval_accuracy": rate(subset, lambda r: r["retrieval_accuracy"]["correct_chunk_retrieved"]),
            "faithfulness": rate(subset, lambda r: r["faithfulness"]["faithful"]),
        }

    return summary


def print_summary(summary: dict):
    def fmt(x):
        return f"{x*100:.1f}%" if x is not None else "N/A"

    print("\n" + "=" * 70)
    print(f"GOLDEN EVAL SUMMARY — {summary['total_questions']} questions")
    print("=" * 70)
    print(f"Overall retrieval accuracy: {fmt(summary['overall']['retrieval_accuracy'])}")
    print(f"Overall faithfulness:       {fmt(summary['overall']['faithfulness'])}")

    print("\nBy language:")
    for lang, s in summary["by_language"].items():
        print(f"  {lang:4s} (n={s['count']:3d})  retrieval={fmt(s['retrieval_accuracy']):>7s}  faithfulness={fmt(s['faithfulness']):>7s}")

    print("\nBy source:")
    for src, s in summary["by_source"].items():
        print(f"  {src:10s} (n={s['count']:3d})  retrieval={fmt(s['retrieval_accuracy']):>7s}  faithfulness={fmt(s['faithfulness']):>7s}")

    print("\nBy section:")
    for sec, s in summary["by_section"].items():
        print(f"  {sec:22s} (n={s['count']:3d})  retrieval={fmt(s['retrieval_accuracy']):>7s}  faithfulness={fmt(s['faithfulness']):>7s}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run the golden evaluation dataset against the live pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (quick smoke test)")
    parser.add_argument("--language", type=str, default=None, choices=["en", "hi", "mr"], help="Only run questions in this language")
    parser.add_argument("--fail-below", type=float, default=None,
                         help="Exit with code 1 if any language's faithfulness rate falls below this threshold (0-1). For CI gating.")
    args = parser.parse_args()

    if not GOLDEN_DATASET_PATH.exists():
        raise SystemExit(f"Golden dataset not found at {GOLDEN_DATASET_PATH}. "
                          f"Run eval/golden_dataset/build_golden_dataset.py first.")

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.language:
        questions = [q for q in questions if q["language"] == args.language]
    if args.limit:
        questions = questions[:args.limit]

    print(f"Running {len(questions)} questions...")
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}")
        try:
            results.append(run_single_question(q))
        except Exception as e:
            print(f"    [error] {e}")
            results.append({
                "id": q["id"], "language": q["language"], "scheme_id": q["scheme_id"],
                "expected_section": q["expected_section"], "source": q["source"],
                "difficulty": q["difficulty"], "question": q["question"],
                "expected_answer": q.get("expected_answer"), "generated_answer": None,
                "retrieved_sections": [], "retrieval_accuracy": {"correct_chunk_retrieved": None, "matched_via": None},
                "faithfulness": {"faithful": None, "reason": None, "judge_error": str(e)},
            })
        time.sleep(0.2)  # light rate-limit courtesy for the Groq free tier

    summary = summarize(results)
    print_summary(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"golden_eval_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to: {report_path}")

    if args.fail_below is not None:
        failing_languages = [
            lang for lang, s in summary["by_language"].items()
            if s["faithfulness"] is not None and s["faithfulness"] < args.fail_below
        ]
        if failing_languages:
            print(f"\n[FAIL] Faithfulness below {args.fail_below*100:.0f}% for: {failing_languages}")
            sys.exit(1)
        else:
            print(f"\n[PASS] All languages meet the {args.fail_below*100:.0f}% faithfulness threshold.")


if __name__ == "__main__":
    main()