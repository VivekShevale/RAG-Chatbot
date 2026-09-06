"""
scripts/debug_retrieval.py

Diagnostic script: prints FULL retrieval results (all metadata, full text,
scores) for the specific questions that failed in the manual eval, so we
can see exactly what's being retrieved and matched against the expected
content.

Run from project root:
    python scripts/debug_retrieval.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.retrieval.retriever import retrieve

# The exact questions that scored "Wrong" in the manual eval, plus their
# English equivalents for comparison (English scored "Good" on these).
DEBUG_CASES = [
    {
        "label": "EN (baseline - scored Good) - tablet_assistance documents",
        "query": "What documents are required for Tablet Assistance scheme?",
        "language": "en",
    },
    {
        "label": "HI (scored Wrong) - tablet_assistance documents",
        "query": "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?",
        "language": "hi",
    },
    {
        "label": "MR (scored Wrong) - tablet_assistance documents",
        "query": "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?",
        "language": "mr",
    },
    {
        "label": "EN (baseline - scored Good) - vahan_aksmat eligibility",
        "query": "Who can get help under Vahan Aksmat Sahay?",
        "language": "en",
    },
    {
        "label": "HI (scored Wrong) - vahan_aksmat eligibility",
        "query": "वाहन अक्षमत सहाय योजना किसे मिलती है?",
        "language": "hi",
    },
    {
        "label": "MR (scored Wrong) - vahan_aksmat eligibility",
        "query": "वाहन अक्षम सहाय योजना कोणाला मिळते?",
        "language": "mr",
    },
]

for case in DEBUG_CASES:
    print("\n" + "=" * 90)
    print(f"CASE: {case['label']}")
    print(f"Query: {case['query']}")
    print("=" * 90)

    results = retrieve(case["query"], language=case["language"], top_k=5)

    if not results:
        print("  >>> NO RESULTS RETURNED <<<")
        continue

    for rank, r in enumerate(results, 1):
        print(f"\n  Rank {rank} | scheme_id={r['scheme_id']} | section={r['section']} "
              f"| faq_index={r['faq_index']} | score={r['score']:.4f}")
        print(f"  Full text:\n  {r['text']}")

print("\n" + "=" * 90)
print("Compare: does the EXPECTED section (documents_required / eligibility) actually")
print("appear in the Hindi/Marathi top-5, and at what rank/score, versus English?")
print("=" * 90)