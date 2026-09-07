"""
eval/scripts/eval_rerank.py

Compare retrieval quality:
  A) baseline: retrieve top_k=5 (vector + section-intent)
  B) rerank:   retrieve top_k=20 → cross-encoder → top 5

Metrics (on items with expected scheme + section):
  - scheme_hit: expected scheme_id in returned chunks
  - section_hit: expected section present for that scheme

Supports both golden schemas:
  - golden_questions.json: scheme_id + expected_section
  - golden_v1.json: expected_citation: {scheme_id, section}

Usage (from project root):
  python eval/scripts/eval_rerank.py
  python eval/scripts/eval_rerank.py --limit 20
  python eval/scripts/eval_rerank.py --lang hi
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve
from pipeline.retrieval.reranker import Reranker


GOLDEN_CANDIDATES = [
    PROJECT_ROOT / "eval" / "golden_dataset" / "golden_questions.json",
    PROJECT_ROOT / "eval" / "golden_dataset" / "golden_v1.json",
]
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("items") or data.get("questions") or []


def normalize_item(item: dict) -> Optional[dict]:
    """
    Normalize to a common shape:
      {
        id, language, question,
        expected_citation: {scheme_id, section}
      }
    """
    if item.get("should_refuse") is True:
        return None

    scheme_id = None
    section = None

    if item.get("expected_citation"):
        scheme_id = item["expected_citation"].get("scheme_id")
        section = item["expected_citation"].get("section")
    else:
        scheme_id = item.get("scheme_id")
        section = item.get("expected_section")

    question = item.get("question")
    if not scheme_id or not section or not question:
        return None

    return {
        "id": item.get("id"),
        "language": item.get("language") or "en",
        "question": question,
        "expected_citation": {
            "scheme_id": scheme_id,
            "section": section,
        },
    }


def score_chunks(chunks: list[dict], expected_scheme: str, expected_section: str) -> dict:
    schemes = {c.get("scheme_id") for c in chunks}
    section_hit = any(
        c.get("scheme_id") == expected_scheme and c.get("section") == expected_section
        for c in chunks
    )
    return {
        "scheme_hit": expected_scheme in schemes,
        "section_hit": section_hit,
    }


def aggregate(rows: list[dict], key: str) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "scheme_hit_rate": round(sum(1 for r in rows if r[key]["scheme_hit"]) / n, 3),
        "section_hit_rate": round(sum(1 for r in rows if r[key]["section_hit"]) / n, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", choices=["en", "hi", "mr"], default=None)
    parser.add_argument("--candidate-k", type=int, default=20, help="top_k before rerank")
    parser.add_argument("--final-n", type=int, default=5, help="chunks kept after rerank / baseline")
    args = parser.parse_args()

    golden_path = next((p for p in GOLDEN_CANDIDATES if p.exists()), None)
    if not golden_path:
        print("No golden dataset found under eval/golden_dataset/")
        sys.exit(1)

    raw_items = load_raw_items(golden_path)
    items = []
    for it in raw_items:
        norm = normalize_item(it)
        if norm is not None:
            items.append(norm)

    if args.lang:
        items = [i for i in items if i["language"] == args.lang]
    if args.limit:
        items = items[: args.limit]

    print(f"Golden file: {golden_path}")
    print(f"Loaded {len(raw_items)} raw → {len(items)} evaluable")
    print(f"Comparing baseline top-{args.final_n} vs rerank {args.candidate_k}→{args.final_n}")

    if not items:
        print("No evaluable items after filtering. Check golden schema.")
        sys.exit(1)

    print("Loading cross-encoder...")
    reranker = Reranker()

    results = []
    for idx, item in enumerate(items, 1):
        q = item["question"]
        lang = item["language"]
        exp_scheme = item["expected_citation"]["scheme_id"]
        exp_section = item["expected_citation"]["section"]

        print(f"[{idx}/{len(items)}] {item.get('id', idx)} ({lang}) target={exp_scheme}|{exp_section}")

        # A) Baseline production path
        base_chunks = retrieve(query=q, language=lang, top_k=args.final_n)
        base_scores = score_chunks(base_chunks, exp_scheme, exp_section)

        # B) Wider retrieve + cross-encoder rerank
        candidates = retrieve(query=q, language=lang, top_k=args.candidate_k)
        reranked = reranker.rerank(q, candidates, top_n=args.final_n)
        rerank_scores = score_chunks(reranked, exp_scheme, exp_section)

        fixed = (not base_scores["section_hit"]) and rerank_scores["section_hit"]
        regressed = base_scores["section_hit"] and (not rerank_scores["section_hit"])

        results.append({
            "id": item.get("id"),
            "language": lang,
            "question": q,
            "expected": item["expected_citation"],
            "baseline": base_scores,
            "rerank": rerank_scores,
            "fixed_by_rerank": fixed,
            "regressed_by_rerank": regressed,
            "baseline_sections": [
                f"{c.get('scheme_id')}|{c.get('section')}" for c in base_chunks
            ],
            "rerank_sections": [
                f"{c.get('scheme_id')}|{c.get('section')}" for c in reranked
            ],
        })

    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)

    summary = {
        "overall": {
            "baseline": aggregate(results, "baseline"),
            "rerank": aggregate(results, "rerank"),
            "fixed_count": sum(1 for r in results if r["fixed_by_rerank"]),
            "regressed_count": sum(1 for r in results if r["regressed_by_rerank"]),
        },
        "by_language": {},
    }
    for lang, rows in by_lang.items():
        summary["by_language"][lang] = {
            "baseline": aggregate(rows, "baseline"),
            "rerank": aggregate(rows, "rerank"),
            "fixed_count": sum(1 for r in rows if r["fixed_by_rerank"]),
            "regressed_count": sum(1 for r in rows if r["regressed_by_rerank"]),
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"rerank_eval_{ts}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {out}")
    print(
        "\nInterpretation:\n"
        "  section_hit_rate higher under 'rerank' → reranker helps\n"
        "  fixed_count = baseline missed section, rerank got it\n"
        "  regressed_count = baseline had it, rerank lost it (bad)"
    )


if __name__ == "__main__":
    main()