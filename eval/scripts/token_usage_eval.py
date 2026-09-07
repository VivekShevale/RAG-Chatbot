"""
eval/scripts/token_usage_eval.py

Phase B6 follow-up — Groq token usage by language.

Runs a fixed prompt set and records prompt_tokens / completion_tokens /
total_tokens from generate_answer (via Groq response.usage), plus generation
latency, so you can test whether hi/mr slowdown tracks completion length.

Jobs are SHUFFLED so en/hi/mr are interleaved (avoids sequential-language
confound from API throttling / queueing).

Usage (from project root):
    python eval/scripts/token_usage_eval.py
    python eval/scripts/token_usage_eval.py --repeats 5
    python eval/scripts/token_usage_eval.py --language mr
    python eval/scripts/token_usage_eval.py --seed 7

Output: eval/reports/token_usage_<timestamp>.json
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve
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

DEFAULT_REPEATS = 3


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


def run_one(prompt: dict, repeat: int) -> dict:
    chunks = retrieve(
        query=prompt["question"],
        language=prompt["language"],
        top_k=5,
    )
    gen = generate_answer(
        query=prompt["question"],
        retrieved_chunks=chunks,
        language=prompt["language"],
        temperature=0.0,
    )
    usage = gen.get("token_usage") or {}
    timings = gen.get("timings_ms") or {}
    answer = gen.get("answer") or ""

    return {
        "prompt_id": prompt["id"],
        "language": prompt["language"],
        "question": prompt["question"],
        "repeat": repeat,
        "n_retrieved": len(chunks),
        "answer_chars": len(answer),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "llm_ms": timings.get("llm_ms"),
        "generation_total_ms": timings.get("generation_total_ms"),
        "raw_valid": gen.get("raw_valid"),
        "timestamp": datetime.now().isoformat(),
    }


def summarize(runs: list[dict]) -> dict:
    metrics = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "answer_chars",
        "generation_total_ms",
    ]

    def bundle(subset: list[dict]) -> dict:
        out = {"n": len(subset)}
        for m in metrics:
            vals = [r[m] for r in subset if r.get(m) is not None]
            out[m] = stats(vals)
        pairs = [
            (r["completion_tokens"], r["generation_total_ms"])
            for r in subset
            if r.get("completion_tokens") and r.get("generation_total_ms")
        ]
        if pairs:
            tps = [ct / (ms / 1000.0) for ct, ms in pairs if ms > 0]
            out["completion_tokens_per_sec"] = stats(tps)
        else:
            out["completion_tokens_per_sec"] = stats([])
        return out

    by_lang = {
        lang: bundle([r for r in runs if r["language"] == lang])
        for lang in sorted({r["language"] for r in runs})
    }
    by_prompt = {
        pid: bundle([r for r in runs if r["prompt_id"] == pid])
        for pid in sorted({r["prompt_id"] for r in runs})
    }
    return {
        "overall": bundle(runs),
        "by_language": by_lang,
        "by_prompt": by_prompt,
    }


def print_summary(summary: dict):
    def row(label: str, block: dict):
        pt = block.get("prompt_tokens") or {}
        ct = block.get("completion_tokens") or {}
        gen = block.get("generation_total_ms") or {}
        tps = block.get("completion_tokens_per_sec") or {}
        print(
            f"  {label:10s}  n={block.get('n', 0):3d}  "
            f"prompt_p50={pt.get('p50')}  "
            f"completion_p50={ct.get('p50')}  "
            f"gen_p50={gen.get('p50')}  "
            f"tok/s_p50={tps.get('p50')}"
        )

    print("\n" + "=" * 78)
    print("TOKEN USAGE EVAL (shuffled job order)")
    print("=" * 78)
    print("\nOverall:")
    row("all", summary["overall"])
    print("\nBy language:")
    for lang, block in summary["by_language"].items():
        row(lang, block)
    print("\nBy prompt:")
    for pid, block in summary["by_prompt"].items():
        row(pid, block)
    print("=" * 78)
    print("If completion_tokens track gen_ms across languages → length-driven latency.")
    print("If completion_tokens similar but gen_ms differs → model/API path slower.")
    print("Shuffled order reduces sequential-language API throttling confound.")


def main():
    parser = argparse.ArgumentParser(description="Groq token usage by language (shuffled)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--language", choices=["en", "hi", "mr"], default=None)
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for job order")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup call")
    args = parser.parse_args()

    prompts = FIXED_PROMPTS
    if args.language:
        prompts = [p for p in prompts if p["language"] == args.language]

    jobs = []
    for prompt in prompts:
        for rep in range(1, args.repeats + 1):
            jobs.append({"prompt": prompt, "repeat": rep})

    random.seed(args.seed)
    random.shuffle(jobs)

    print(
        f"Running {len(jobs)} requests "
        f"(shuffled; seed={args.seed}; languages interleaved)"
    )

    if not args.no_warmup and prompts:
        print("Warmup (not recorded)...")
        try:
            run_one(prompts[0], repeat=0)
        except Exception as e:
            print(f"  warmup warning: {e}")
        time.sleep(0.5)

    runs = []
    for i, job in enumerate(jobs, 1):
        prompt = job["prompt"]
        rep = job["repeat"]
        print(
            f"  [{i}/{len(jobs)}] {prompt['id']}  "
            f"repeat={rep}  lang={prompt['language']}"
        )
        try:
            rec = run_one(prompt, rep)
            runs.append(rec)
            print(
                f"         prompt={rec['prompt_tokens']}  "
                f"completion={rec['completion_tokens']}  "
                f"gen={rec['generation_total_ms']}ms  "
                f"chars={rec['answer_chars']}"
            )
        except Exception as e:
            print(f"         ERROR: {e}")
        time.sleep(0.2)

    summary = summarize(runs)
    print_summary(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"token_usage_{ts}.json"
    payload = {
        "config": {
            "repeats": args.repeats,
            "seed": args.seed,
            "shuffled": True,
            "warmup": not args.no_warmup,
            "prompts": prompts,
        },
        "summary": summary,
        "runs": runs,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()