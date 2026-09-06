"""
test/manual_eval.py

Interactive manual evaluation for Phase A Step 6.
Works when run from project root OR from inside the test/ folder.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# ---------- Make project root importable ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer


# ---------- Test Questions ----------
TEST_QUESTIONS = [
    # English
    {"id": "en_01", "language": "en", "scheme": "pm_poshan", "question": "Who is eligible for PM Poshan?"},
    {"id": "en_02", "language": "en", "scheme": "tablet_assistance", "question": "What documents are required for Tablet Assistance scheme?"},
    {"id": "en_03", "language": "en", "scheme": "medical_checkup", "question": "What is the benefit under Medical Checkup Assistance scheme?"},
    {"id": "en_04", "language": "en", "scheme": "vahan_aksmat", "question": "Who can get help under Vahan Aksmat Sahay?"},
    {"id": "en_05", "language": "en", "scheme": "doodh_sanjivani", "question": "What is Doodh Sanjivani scheme?"},

    # Hindi
    {"id": "hi_01", "language": "hi", "scheme": "pm_poshan", "question": "पीएम पोषण योजना के लिए कौन पात्र है?"},
    {"id": "hi_02", "language": "hi", "scheme": "tablet_assistance", "question": "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?"},
    {"id": "hi_03", "language": "hi", "scheme": "medical_checkup", "question": "चिकित्सा जांच सहायता योजना का लाभ क्या है?"},
    {"id": "hi_04", "language": "hi", "scheme": "vahan_aksmat", "question": "वाहन अक्षमत सहाय योजना किसे मिलती है?"},
    {"id": "hi_05", "language": "hi", "scheme": "doodh_sanjivani", "question": "दूध संजीवनी योजना क्या है?"},

    # Marathi
    {"id": "mr_01", "language": "mr", "scheme": "pm_poshan", "question": "पीएम पोषण योजनेसाठी कोण पात्र आहे?"},
    {"id": "mr_02", "language": "mr", "scheme": "tablet_assistance", "question": "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?"},
    {"id": "mr_03", "language": "mr", "scheme": "medical_checkup", "question": "वैद्यकीय तपासणी सहाय्य योजनेचा लाभ काय आहे?"},
    {"id": "mr_04", "language": "mr", "scheme": "vahan_aksmat", "question": "वाहन अक्षम सहाय योजना कोणाला मिळते?"},
    {"id": "mr_05", "language": "mr", "scheme": "doodh_sanjivani", "question": "दूध संजीवनी योजना म्हणजे काय?"},
]


RESULTS_DIR = PROJECT_ROOT / "eval" / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_rating() -> str:
    while True:
        print("\nRate this answer:")
        print("  1 → Good")
        print("  2 → Partial")
        print("  3 → Wrong")
        choice = input("Your choice (1/2/3): ").strip()

        if choice == "1":
            return "Good"
        elif choice == "2":
            return "Partial"
        elif choice == "3":
            return "Wrong"
        else:
            print("Invalid input. Please enter 1, 2 or 3.")


def run_evaluation():
    print("=" * 70)
    print("PHASE A – STEP 6: Manual End-to-End Evaluation")
    print("=" * 70)
    print(f"Total questions: {len(TEST_QUESTIONS)}\n")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, item in enumerate(TEST_QUESTIONS, 1):
        print("\n" + "-" * 70)
        print(f"Question {i}/{len(TEST_QUESTIONS)}  |  ID: {item['id']}  |  Lang: {item['language']}")
        print(f"Scheme: {item['scheme']}")
        print(f"Q: {item['question']}")
        print("-" * 70)

        chunks = retrieve(
            query=item["question"],
            language=item["language"],
            top_k=5
        )
        result = generate_answer(
            query=item["question"],
            retrieved_chunks=chunks,
            language=item["language"]
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nTop citations:")
        for j, c in enumerate(result["cited_chunks"][:3], 1):
            print(f"  {j}. {c['scheme_id']} | {c['section']} | {c['language']}")

        rating = get_rating()
        note = input("Any short note? (press Enter to skip): ").strip()

        record = {
            "id": item["id"],
            "language": item["language"],
            "scheme": item["scheme"],
            "question": item["question"],
            "answer": result["answer"],
            "cited_chunks": [
                {
                    "scheme_id": c["scheme_id"],
                    "section": c["section"],
                    "language": c["language"],
                    "score": c.get("score")
                }
                for c in result["cited_chunks"][:5]
            ],
            "rating": rating,
            "note": note,
            "timestamp": datetime.now().isoformat()
        }
        results.append(record)
        print(f"\n→ Recorded as: {rating}")

    output_file = RESULTS_DIR / f"manual_eval_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    from collections import Counter
    counts = Counter(r["rating"] for r in results)
    print(f"Good    : {counts.get('Good', 0)}")
    print(f"Partial : {counts.get('Partial', 0)}")
    print(f"Wrong   : {counts.get('Wrong', 0)}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()