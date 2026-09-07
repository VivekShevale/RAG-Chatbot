"""
eval/scripts/latency_breakdown.py

Phase B6 — Latency budget breakdown.

Runs a fixed prompt set, measures:
  - vector_search_ms
  - section_intent_ms
  - retrieval_total_ms
  - generation_total_ms (llm + parse)
  - total_ms

Reports P50 / P95 (and mean) overall, by language, and by path
(section-intent triggered vs pure vector).

Usage (from project root):
    python eval/scripts/latency_breakdown.py
    python eval/scripts/latency_breakdown.py --repeats 5
    python eval/scripts/latency_breakdown.py --language hi
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve_with_timings
from pipeline.generation.generate import generate_answer

REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FIXED_PROMPTS = [
    {"id": "en_01", "language": "en", "question": "Who is eligible for PM POSHAN?"},
    {"id": "en_02", "language": "en", "question": "What documents are required for Tablet Assistance scheme?"},
    {"id": "en_03", "language": "en", "question": "What is the benefit under Medical Checkup Assistance scheme?"},
    {"id": "hi_01", "language": "hi", "question": "पीएम पोषण योजना के लिए कौन पात्र है?"},
    {"id": "hi_02", "language": "hi", "question": "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?"},
    {"id": "hi_03", "language": "hi", "question": "चिकित्सा जांच सहायता योजना का लाभ क्या है?"},
    {"id": "mr_01", "language": "mr", "question": "पीएम पोषण योजनेसाठी कोण पात्र आहे?"},
    {"id": "mr_02", "language": "mr", "question": "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?"},
    {"id": "mr_03", "language": "mr", "question": "वैद्यकीय तपासणी सहाय्य योजनेचा लाभ काय आहे?"},
]

DEFAULT_REPEATS = 5


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def matched_via_counts(chunks: list[dict]) -> dict:
    counts = defaultdict(int)
    for c in chunks:
        counts[c.get("matched_via") or "unknown"] += 1
    return dict(counts)


def run_one(prompt: dict, repeat: int) -> dict:
    t_all0 = time.perf_counter()
    chunks, ret_timings = retrieve_with_timings(
        query=prompt["question"],
        language=prompt["language"],
        top_k=5,
    )
    gen = generate_answer(
        query=prompt["question"],
        retrieved_chunks=chunks,
        language=prompt["language"],
        temperature=0.0,  # stable for latency comparison
    )
    total_ms = (time.perf_counter() - t_all0) * 1000

    gen_timings = gen.get("timings_ms") or {}
    return {
        "prompt_id": prompt["id"],
        "language": prompt["language"],
        "question": prompt["question"],
        "repeat": repeat,
        "section_intent_triggered": ret_timings.get("section_intent_triggered"),
        "matched_via": matched_via_counts(chunks),
        "n_retrieved": len(chunks),
        "vector_search_ms": ret_timings.get("vector_search_ms"),
        "section_intent_ms": ret_timings.get("section_intent_ms"),
        "retrieval_total_ms": ret_timings.get("retrieval_total_ms"),
        "llm_ms": gen_timings.get("llm_ms"),
        "parse_validate_ms": gen_timings.get("parse_validate_ms"),
        "generation_total_ms": gen_timings.get("generation_total_ms"),
        "total_ms": round(total_ms, 2),
        "raw_valid": gen.get("raw_valid"),
        "timestamp": datetime.now().isoformat(),
    }


def summarize(runs: list[dict]) -> dict:
    metrics = [
        "vector_search_ms",
        "section_intent_ms",
        "retrieval_total_ms",
        "generation_total_ms",
        "total_ms",
    ]

    def bundle(subset: list[dict]) -> dict:
        out = {"n": len(subset)}
        for m in metrics:
            vals = [r[m] for r in subset if r.get(m) is not None]
            out[m] = stats(vals)
        triggered = sum(1 for r in subset if r.get("section_intent_triggered"))
        out["section_intent_triggered_rate"] = (
            round(triggered / len(subset), 3) if subset else None
        )
        return out

    by_lang = {
        lang: bundle([r for r in runs if r["language"] == lang])
        for lang in sorted({r["language"] for r in runs})
    }
    by_path = {
        "section_intent": bundle([r for r in runs if r.get("section_intent_triggered")]),
        "pure_vector": bundle([r for r in runs if not r.get("section_intent_triggered")]),
    }
    return {
        "overall": bundle(runs),
        "by_language": by_lang,
        "by_path": by_path,
    }


def print_summary(summary: dict):
    def row(label: str, block: dict):
        t = block.get("total_ms") or {}
        r = block.get("retrieval_total_ms") or {}
        g = block.get("generation_total_ms") or {}
        print(
            f"  {label:18s}  n={block.get('n', 0):3d}  "
            f"total p50={t.get('p50')} p95={t.get('p95')}  "
            f"retr p50={r.get('p50')}  gen p50={g.get('p50')}"
        )

    print("\n" + "=" * 72)
    print("LATENCY BREAKDOWN (ms)")
    print("=" * 72)
    print("\nOverall:")
    row("all", summary["overall"])
    print("\nBy language:")
    for lang, block in summary["by_language"].items():
        row(lang, block)
    print("\nBy path:")
    for path, block in summary["by_path"].items():
        row(path, block)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Latency budget breakdown")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--language", choices=["en", "hi", "mr"], default=None)
    args = parser.parse_args()

    prompts = FIXED_PROMPTS
    if args.language:
        prompts = [p for p in prompts if p["language"] == args.language]

    total = len(prompts) * args.repeats
    print(f"Running {len(prompts)} prompts × {args.repeats} repeats = {total} requests")

    runs = []
    done = 0
    for prompt in prompts:
        for rep in range(1, args.repeats + 1):
            done += 1
            print(f"  [{done}/{total}] {prompt['id']} repeat={rep}")
            try:
                rec = run_one(prompt, rep)
                runs.append(rec)
                print(
                    f"         total={rec['total_ms']}ms  "
                    f"retr={rec['retrieval_total_ms']}  "
                    f"gen={rec['generation_total_ms']}  "
                    f"si={rec['section_intent_triggered']}"
                )
            except Exception as e:
                print(f"         ERROR: {e}")
            time.sleep(0.2)

    summary = summarize(runs)
    print_summary(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"latency_breakdown_{ts}.json"
    payload = {
        "config": {"repeats": args.repeats, "prompts": prompts},
        "summary": summary,
        "runs": runs,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()