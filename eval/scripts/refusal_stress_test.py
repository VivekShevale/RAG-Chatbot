"""
eval/scripts/refusal_stress_test.py

Phase B3 — Citation enforcement / refusal stress test.

Deliberately asks out-of-corpus questions in en/hi/mr and records
whether the system refuses instead of inventing an answer.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer


REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Out-of-corpus questions (nothing in your 5 schemes answers these)
STRESS_QUESTIONS = [
    # English
    {"id": "refuse_en_01", "language": "en", "question": "How do I apply for PM Awas Yojana in Mumbai?"},
    {"id": "refuse_en_02", "language": "en", "question": "What is the scholarship amount under National Merit Scholarship 2024?"},
    {"id": "refuse_en_03", "language": "en", "question": "What is the eligibility for Ayushman Bharat in Gujarat?"},
    {"id": "refuse_en_04", "language": "en", "question": "How much subsidy does PM Surya Ghar give for solar panels?"},
    {"id": "refuse_en_05", "language": "en", "question": "What documents are needed for a passport in Pune?"},
    # Hindi
    {"id": "refuse_hi_01", "language": "hi", "question": "मुंबई में पीएम आवास योजना के लिए कैसे आवेदन करें?"},
    {"id": "refuse_hi_02", "language": "hi", "question": "राष्ट्रीय मेरिट छात्रवृत्ति 2024 की राशि कितनी है?"},
    {"id": "refuse_hi_03", "language": "hi", "question": "गुजरात में आयुष्मान भारत की पात्रता क्या है?"},
    {"id": "refuse_hi_04", "language": "hi", "question": "पीएम सूर्य घर योजना में सोलर पैनल पर कितनी सब्सिडी मिलती है?"},
    {"id": "refuse_hi_05", "language": "hi", "question": "पुणे में पासपोर्ट के लिए कौन से दस्तावेज चाहिए?"},
    # Marathi
    {"id": "refuse_mr_01", "language": "mr", "question": "मुंबईत पीएम आवास योजनेसाठी अर्ज कसा करावा?"},
    {"id": "refuse_mr_02", "language": "mr", "question": "राष्ट्रीय मेरीट शिष्यवृत्ती 2024 ची रक्कम किती आहे?"},
    {"id": "refuse_mr_03", "language": "mr", "question": "गुजरातमध्ये आयुष्मान भारतची पात्रता काय आहे?"},
    {"id": "refuse_mr_04", "language": "mr", "question": "पीएम सूर्य घर योजनेत सोलर पॅनेलवर किती सबसिडी मिळते?"},
    {"id": "refuse_mr_05", "language": "mr", "question": "पुण्यात पासपोर्टसाठी कोणती कागदपत्रे लागतात?"},
]

REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "no relevant information",
    "पर्याप्त जानकारी नहीं",
    "पर्याप्त जानकारी नही",
    "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं",
    "पुरेशी माहिती नाही",
    "माझ्याकडे पुरेशी माहिती नाही",
    "उपलब्ध दस्तऐवजांच्या आधारे",
    "उपलब्ध दस्तावेजों के आधार पर",
]


def is_refusal(answer: str) -> bool:
    text = (answer or "").lower()
    return any(m.lower() in text for m in REFUSAL_MARKERS)


def run():
    print("=" * 70)
    print("Phase B3 — Citation enforcement / refusal stress test")
    print(f"Questions: {len(STRESS_QUESTIONS)} (5 per language)")
    print("=" * 70)

    results = []
    for i, item in enumerate(STRESS_QUESTIONS, 1):
        lang = item["language"]
        q = item["question"]
        print(f"\n[{i}/{len(STRESS_QUESTIONS)}] {item['id']} ({lang})")
        print(f"  Q: {q}")

        chunks = retrieve(query=q, language=lang, top_k=5)
        gen = generate_answer(query=q, retrieved_chunks=chunks, language=lang)
        answer = gen.get("answer", "")
        refused = is_refusal(answer)
        citations = gen.get("citations") or []

        # Pass if: refused AND (ideally empty citations)
        passed = refused

        record = {
            "id": item["id"],
            "language": lang,
            "question": q,
            "answer": answer,
            "refused": refused,
            "citations": citations,
            "n_retrieved": len(chunks),
            "retrieved_schemes": sorted({c.get("scheme_id") for c in chunks if c.get("scheme_id")}),
            "raw_valid": gen.get("raw_valid"),
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(record)

        status = "PASS" if passed else "FAIL"
        print(f"  → {status} | refused={refused} | citations={len(citations)}")
        print(f"  Answer: {answer[:180]}...")

    # Summary
    by_lang = {"en": [], "hi": [], "mr": []}
    for r in results:
        by_lang[r["language"]].append(r)

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 3),
        "by_language": {},
    }
    for lang, rows in by_lang.items():
        n = len(rows) or 1
        summary["by_language"][lang] = {
            "n": len(rows),
            "passed": sum(1 for r in rows if r["passed"]),
            "pass_rate": round(sum(1 for r in rows if r["passed"]) / n, 3),
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"refusal_stress_{ts}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nReport saved to: {out}")
    return summary


if __name__ == "__main__":
    run()