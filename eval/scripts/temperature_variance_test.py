"""
eval/scripts/temperature_variance_test.py

Phase B4 — Temperature variance test.

Runs a fixed prompt set at temperature 0.0 and 0.7, multiple times each,
and logs output variance per language.

Metrics (per language × temperature):
  - JSON validity rate
  - Unique answer count / exact-match consistency across repeats
  - Confidence distribution
  - Average answer length

Usage (from project root):
    python eval/scripts/temperature_variance_test.py
    python eval/scripts/temperature_variance_test.py --repeats 5
    python eval/scripts/temperature_variance_test.py --language hi

Output: eval/reports/temperature_variance_<timestamp>.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer

REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed prompt set — balanced across languages and section types.
# Keep small so the test is cheap on free-tier Groq limits.
FIXED_PROMPTS = [
    # English
    {"id": "en_01", "language": "en", "question": "Who is eligible for PM POSHAN?"},
    {"id": "en_02", "language": "en", "question": "What documents are required for Tablet Assistance scheme?"},
    {"id": "en_03", "language": "en", "question": "What is the benefit under Medical Checkup Assistance scheme?"},
    # Hindi
    {"id": "hi_01", "language": "hi", "question": "पीएम पोषण योजना के लिए कौन पात्र है?"},
    {"id": "hi_02", "language": "hi", "question": "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?"},
    {"id": "hi_03", "language": "hi", "question": "चिकित्सा जांच सहायता योजना का लाभ क्या है?"},
    # Marathi
    {"id": "mr_01", "language": "mr", "question": "पीएम पोषण योजनेसाठी कोण पात्र आहे?"},
    {"id": "mr_02", "language": "mr", "question": "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?"},
    {"id": "mr_03", "language": "mr", "question": "वैद्यकीय तपासणी सहाय्य योजनेचा लाभ काय आहे?"},
]

TEMPERATURES = [0.0, 0.7]
DEFAULT_REPEATS = 3


def normalize_answer(text: str) -> str:
    """Light normalization so minor whitespace differences don't count as variance."""
    return " ".join((text or "").split()).strip().lower()


def run_one(prompt: dict, temperature: float, repeat_idx: int) -> dict:
    chunks = retrieve(
        query=prompt["question"],
        language=prompt["language"],
        top_k=5,
    )
    result = generate_answer(
        query=prompt["question"],
        retrieved_chunks=chunks,
        language=prompt["language"],
        temperature=temperature,
    )
    return {
        "prompt_id": prompt["id"],
        "language": prompt["language"],
        "question": prompt["question"],
        "temperature": temperature,
        "repeat": repeat_idx,
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence"),
        "citations": result.get("citations") or [],
        "raw_valid": result.get("raw_valid"),
        "parse_error": result.get("parse_error"),
        "n_retrieved": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }


def summarize(runs: list[dict]) -> dict:
    """Aggregate variance metrics: overall, by language, by temperature,
    and by language × temperature."""

    def metrics_for(subset: list[dict]) -> dict:
        if not subset:
            return {
                "n_runs": 0,
                "json_validity_rate": None,
                "unique_answers": 0,
                "exact_match_consistency": None,
                "confidence_dist": {},
                "avg_answer_length": None,
            }

        answers_norm = [normalize_answer(r["answer"]) for r in subset]
        unique = set(answers_norm)
        # Pairwise exact-match rate across all runs in this subset
        # (1.0 = every run produced the identical answer)
        if len(answers_norm) <= 1:
            consistency = 1.0
        else:
            matches = 0
            pairs = 0
            for i in range(len(answers_norm)):
                for j in range(i + 1, len(answers_norm)):
                    pairs += 1
                    if answers_norm[i] == answers_norm[j]:
                        matches += 1
            consistency = matches / pairs if pairs else 1.0

        conf_counts = Counter(r.get("confidence") or "unknown" for r in subset)
        lengths = [len(r.get("answer") or "") for r in subset]

        return {
            "n_runs": len(subset),
            "json_validity_rate": round(
                sum(1 for r in subset if r.get("raw_valid")) / len(subset), 3
            ),
            "unique_answers": len(unique),
            "exact_match_consistency": round(consistency, 3),
            "confidence_dist": dict(conf_counts),
            "avg_answer_length": round(sum(lengths) / len(lengths), 1),
        }

    # Per (language, temperature, prompt_id) — most meaningful variance signal
    by_prompt_temp: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        key = f"{r['language']}|{r['temperature']}|{r['prompt_id']}"
        by_prompt_temp[key].append(r)

    per_prompt = {}
    for key, subset in sorted(by_prompt_temp.items()):
        lang, temp, pid = key.split("|", 2)
        per_prompt[key] = {
            "language": lang,
            "temperature": float(temp),
            "prompt_id": pid,
            **metrics_for(subset),
        }

    # Roll up: language × temperature (average consistency across prompts)
    by_lang_temp: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_lang_temp[f"{r['language']}|{r['temperature']}"].append(r)

    lang_temp_summary = {}
    for key, subset in sorted(by_lang_temp.items()):
        lang, temp = key.split("|", 1)
        # Consistency averaged over prompts in this lang×temp bucket
        prompt_consistencies = []
        for pid in {r["prompt_id"] for r in subset}:
            prompt_runs = [r for r in subset if r["prompt_id"] == pid]
            m = metrics_for(prompt_runs)
            if m["exact_match_consistency"] is not None:
                prompt_consistencies.append(m["exact_match_consistency"])

        base = metrics_for(subset)
        base["mean_prompt_consistency"] = (
            round(sum(prompt_consistencies) / len(prompt_consistencies), 3)
            if prompt_consistencies
            else None
        )
        lang_temp_summary[key] = {"language": lang, "temperature": float(temp), **base}

    by_language = {}
    for lang in sorted({r["language"] for r in runs}):
        by_language[lang] = metrics_for([r for r in runs if r["language"] == lang])

    by_temperature = {}
    for temp in sorted({r["temperature"] for r in runs}):
        by_temperature[str(temp)] = metrics_for(
            [r for r in runs if r["temperature"] == temp]
        )

    return {
        "overall": metrics_for(runs),
        "by_language": by_language,
        "by_temperature": by_temperature,
        "by_language_temperature": lang_temp_summary,
        "by_prompt_temperature": per_prompt,
    }


def print_summary(summary: dict):
    def fmt_rate(x):
        return f"{x * 100:.1f}%" if x is not None else "N/A"

    print("\n" + "=" * 72)
    print("TEMPERATURE VARIANCE SUMMARY")
    print("=" * 72)

    print("\nBy language × temperature:")
    print(
        f"  {'lang':4s}  {'temp':6s}  {'n':>4s}  {'json_ok':>8s}  "
        f"{'unique':>6s}  {'consistency':>12s}  {'avg_len':>8s}"
    )
    for key, s in summary["by_language_temperature"].items():
        print(
            f"  {s['language']:4s}  {s['temperature']:6.1f}  {s['n_runs']:4d}  "
            f"{fmt_rate(s['json_validity_rate']):>8s}  "
            f"{s['unique_answers']:6d}  "
            f"{fmt_rate(s.get('mean_prompt_consistency')):>12s}  "
            f"{s['avg_answer_length'] or 0:8.1f}"
        )

    print("\nInterpretation tips:")
    print("  - consistency near 100% at temp=0.0 → deterministic (good)")
    print("  - consistency drop at temp=0.7 → expected variance")
    print("  - compare consistency gap across en / hi / mr")
    print("  - json_ok should stay high at both temps; if not, structured output is fragile")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Temperature 0 vs 0.7 variance test, logged per language"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"How many times to run each prompt at each temperature (default {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["en", "hi", "mr"],
        help="Only run prompts in this language",
    )
    args = parser.parse_args()

    prompts = FIXED_PROMPTS
    if args.language:
        prompts = [p for p in prompts if p["language"] == args.language]

    total = len(prompts) * len(TEMPERATURES) * args.repeats
    print("=" * 72)
    print("Phase B4 — Temperature variance test (0.0 vs 0.7)")
    print(f"Prompts: {len(prompts)} | Temps: {TEMPERATURES} | Repeats: {args.repeats}")
    print(f"Total LLM calls: {total}")
    print("=" * 72)

    runs = []
    done = 0
    for prompt in prompts:
        # Retrieve once per prompt (retrieval is deterministic for this test)
        for temp in TEMPERATURES:
            for rep in range(1, args.repeats + 1):
                done += 1
                print(
                    f"  [{done}/{total}] {prompt['id']}  temp={temp}  repeat={rep}"
                )
                try:
                    record = run_one(prompt, temp, rep)
                    runs.append(record)
                    status = "ok" if record["raw_valid"] else "INVALID_JSON"
                    preview = (record["answer"] or "")[:80].replace("\n", " ")
                    print(f"         → {status} | conf={record['confidence']} | {preview}...")
                except Exception as e:
                    print(f"         → ERROR: {e}")
                    runs.append(
                        {
                            "prompt_id": prompt["id"],
                            "language": prompt["language"],
                            "question": prompt["question"],
                            "temperature": temp,
                            "repeat": rep,
                            "answer": "",
                            "confidence": None,
                            "citations": [],
                            "raw_valid": False,
                            "parse_error": str(e),
                            "n_retrieved": 0,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                time.sleep(0.25)  # light rate-limit courtesy

    summary = summarize(runs)
    print_summary(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"temperature_variance_{ts}.json"
    payload = {
        "config": {
            "temperatures": TEMPERATURES,
            "repeats": args.repeats,
            "prompts": prompts,
            "model_note": "Uses whatever GROQ_MODEL is configured in generate.py",
        },
        "summary": summary,
        "runs": runs,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report saved to: {out}")


if __name__ == "__main__":
    main()
