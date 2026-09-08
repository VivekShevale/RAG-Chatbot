"""
eval/scripts/compare_groq_models.py

Phase C3 — Compare 3 Groq models on a fixed golden subset.

Models (option B):
  - openai/gpt-oss-120b
  - openai/gpt-oss-20b
  - qwen/qwen3.8-27b

Jobs are SHUFFLED (model × question) so language/API session order
does not confound latency (same lesson as Phase B6 latency eval).

Metrics:
  - raw_valid (structured JSON)
  - generation_ms
  - prompt_tokens / completion_tokens (if generate_answer returns usage)
  - n_citations

Usage:
  python eval/scripts/compare_groq_models.py --limit 5
  python eval/scripts/compare_groq_models.py --limit 10 --seed 42
  python eval/scripts/compare_groq_models.py --limit 5 --no-shuffle
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.generation.generate import generate_answer
from pipeline.retrieval.retriever import retrieve

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_dataset" / "golden_questions.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else (data.get("items") or [])


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return round(xs[k], 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Items per language (balanced en/hi/mr subset)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffle (not recommended for latency)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="One cheap warmup generate before timed jobs (default on)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup call",
    )
    args = parser.parse_args()

    items = [i for i in load_items(GOLDEN_PATH) if i.get("question")]
    langs = ["en", "hi", "mr"]

    # Balanced fixed subset: first N per language
    subset = []
    for lang in langs:
        lang_items = [i for i in items if i.get("language") == lang][: args.limit]
        subset.extend(lang_items)

    if not subset:
        print("No golden items found.")
        sys.exit(1)

    # Build all (model, item) jobs, then shuffle
    jobs = []
    for model in MODELS:
        for item in subset:
            jobs.append({"model": model, "item": item})

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(jobs)
        order_mode = f"shuffled(seed={args.seed})"
    else:
        order_mode = "sequential(model-major)"

    print(f"Models: {MODELS}")
    print(f"Subset: {len(subset)} questions ({args.limit} per lang × {len(langs)} langs)")
    print(f"Jobs:   {len(jobs)}  order={order_mode}")

    # Optional warmup (not timed in summary)
    do_warmup = args.warmup and not args.no_warmup
    if do_warmup:
        print("Warmup...")
        w = subset[0]
        chunks = retrieve(query=w["question"], language=w.get("language") or "en", top_k=3)
        generate_answer(
            query=w["question"],
            retrieved_chunks=chunks,
            language=w.get("language") or "en",
            model=MODELS[0],
        )

    results = []
    for i, job in enumerate(jobs, 1):
        model = job["model"]
        item = job["item"]
        q = item["question"]
        lang = item.get("language") or "en"
        print(f"[{i}/{len(jobs)}] {model} | {item.get('id', i)} ({lang})")

        chunks = retrieve(query=q, language=lang, top_k=5)

        t0 = time.perf_counter()
        gen = generate_answer(
            query=q,
            retrieved_chunks=chunks,
            language=lang,
            model=model,
        )
        gen_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        usage = gen.get("usage") or {}
        prompt_tok = usage.get("prompt_tokens")
        completion_tok = usage.get("completion_tokens")
        tok_s = None
        if completion_tok and gen_ms > 0:
            tok_s = round(completion_tok / (gen_ms / 1000.0), 1)

        results.append(
            {
                "job_index": i,
                "model": model,
                "id": item.get("id"),
                "language": lang,
                "question": q,
                "answer_preview": (gen.get("answer") or "")[:300],
                "raw_valid": gen.get("raw_valid"),
                "confidence": gen.get("confidence"),
                "n_citations": len(gen.get("citations") or []),
                "generation_ms": gen_ms,
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "completion_tok_s": tok_s,
            }
        )
        print(
            f"  valid={gen.get('raw_valid')}  gen_ms={gen_ms}  "
            f"cites={len(gen.get('citations') or [])}  tok/s={tok_s}"
        )

    # Aggregates
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    summary = {
        "order_mode": order_mode,
        "seed": args.seed if not args.no_shuffle else None,
        "n_jobs": len(results),
        "n_questions": len(subset),
        "by_model": {},
        "by_model_language": {},
    }

    for model, rows in by_model.items():
        n = len(rows) or 1
        times = [r["generation_ms"] for r in rows]
        valid_n = sum(1 for r in rows if r.get("raw_valid"))
        tok_s_vals = [r["completion_tok_s"] for r in rows if r.get("completion_tok_s") is not None]
        summary["by_model"][model] = {
            "n": len(rows),
            "json_valid_rate": round(valid_n / n, 3),
            "gen_ms_mean": round(sum(times) / len(times), 1) if times else None,
            "gen_ms_p50": percentile(times, 50),
            "gen_ms_p95": percentile(times, 95),
            "tok_s_mean": round(sum(tok_s_vals) / len(tok_s_vals), 1) if tok_s_vals else None,
        }
        for lang in langs:
            lang_rows = [r for r in rows if r["language"] == lang]
            if not lang_rows:
                continue
            ln = len(lang_rows)
            lt = [r["generation_ms"] for r in lang_rows]
            summary["by_model_language"][f"{model}::{lang}"] = {
                "n": ln,
                "json_valid_rate": round(
                    sum(1 for r in lang_rows if r.get("raw_valid")) / ln, 3
                ),
                "gen_ms_mean": round(sum(lt) / len(lt), 1) if lt else None,
                "gen_ms_p50": percentile(lt, 50),
            }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"groq_model_compare_{ts}.json"
    payload = {
        "summary": summary,
        "models": MODELS,
        "results": results,
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()