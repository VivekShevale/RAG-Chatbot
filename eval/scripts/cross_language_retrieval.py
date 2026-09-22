"""
eval/scripts/cross_language_retrieval.py

Cross-language retrieval benchmark.

Modes:
  - same_lang: query language = filter language (default production path)
  - all_lang:  search_all_languages=True (no language filter)

For each golden item we record:
  - hit_scheme: expected scheme_id in top-k
  - hit_section: expected scheme_id + section in top-k
  - top1_language: language of rank-1 chunk
  - cross_lingual_hit: hit_scheme and rank-1 (or any hit) language != query language

Usage:
  python eval/scripts/cross_language_retrieval.py
  python eval/scripts/cross_language_retrieval.py --limit 20
  python eval/scripts/cross_language_retrieval.py --top-k 5
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval.retriever import retrieve

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_dataset" / "golden_questions.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else (data.get("items") or data.get("questions") or [])


def score_chunks(item: dict, chunks: list[dict]) -> dict:
    exp_scheme = item.get("scheme_id") or item.get("expected_scheme_id")
    exp_section = item.get("expected_section")
    q_lang = item.get("language")

    scheme_hits = [c for c in chunks if c.get("scheme_id") == exp_scheme]
    section_hits = [
        c for c in chunks
        if c.get("scheme_id") == exp_scheme and c.get("section") == exp_section
    ]

    top1 = chunks[0] if chunks else None
    top1_lang = top1.get("language") if top1 else None

    any_hit_langs = {c.get("language") for c in scheme_hits if c.get("language")}
    cross = bool(scheme_hits) and (q_lang not in any_hit_langs or (
        top1_lang is not None and top1_lang != q_lang
    ))
    # stricter: true cross-lingual if a scheme hit exists in a different language
    cross_strict = any(c.get("language") and c.get("language") != q_lang for c in scheme_hits)

    return {
        "hit_scheme": len(scheme_hits) > 0,
        "hit_section": len(section_hits) > 0,
        "n_retrieved": len(chunks),
        "top1_scheme": top1.get("scheme_id") if top1 else None,
        "top1_section": top1.get("section") if top1 else None,
        "top1_language": top1_lang,
        "hit_languages": sorted(any_hit_langs),
        "cross_lingual_scheme_hit": cross_strict,
        "retrieved": [
            {
                "scheme_id": c.get("scheme_id"),
                "section": c.get("section"),
                "language": c.get("language"),
                "matched_via": c.get("matched_via"),
            }
            for c in chunks
        ],
    }


def rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language", choices=["en", "hi", "mr"], default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    items = [i for i in load_items(GOLDEN_PATH) if i.get("question")]
    if args.language:
        items = [i for i in items if i.get("language") == args.language]
    if args.limit:
        items = items[: args.limit]

    print(f"Running cross-language retrieval on {len(items)} questions (top_k={args.top_k})")

    results = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        lang = item.get("language") or "en"
        print(f"  [{i}/{len(items)}] {item.get('id', i)} lang={lang}")

        same = retrieve(query=q, language=lang, top_k=args.top_k, search_all_languages=False)
        all_ = retrieve(query=q, language=None, top_k=args.top_k, search_all_languages=True)

        same_s = score_chunks(item, same)
        all_s = score_chunks(item, all_)

        results.append({
            "id": item.get("id"),
            "language": lang,
            "scheme_id": item.get("scheme_id") or item.get("expected_scheme_id"),
            "expected_section": item.get("expected_section"),
            "question": q,
            "same_lang": same_s,
            "all_lang": all_s,
        })

    # Summaries
    by_q_lang = defaultdict(list)
    for r in results:
        by_q_lang[r["language"]].append(r)

    summary = {
        "n": len(results),
        "top_k": args.top_k,
        "overall": {
            "same_lang_hit_scheme": rate([r["same_lang"] for r in results], "hit_scheme"),
            "same_lang_hit_section": rate([r["same_lang"] for r in results], "hit_section"),
            "all_lang_hit_scheme": rate([r["all_lang"] for r in results], "hit_scheme"),
            "all_lang_hit_section": rate([r["all_lang"] for r in results], "hit_section"),
            "all_lang_cross_lingual_rate": rate(
                [r["all_lang"] for r in results], "cross_lingual_scheme_hit"
            ),
        },
        "by_query_language": {},
    }

    for lang, rows in sorted(by_q_lang.items()):
        summary["by_query_language"][lang] = {
            "n": len(rows),
            "same_lang_hit_scheme": rate([r["same_lang"] for r in rows], "hit_scheme"),
            "same_lang_hit_section": rate([r["same_lang"] for r in rows], "hit_section"),
            "all_lang_hit_scheme": rate([r["all_lang"] for r in rows], "hit_scheme"),
            "all_lang_hit_section": rate([r["all_lang"] for r in rows], "hit_section"),
            "all_lang_cross_lingual_rate": rate(
                [r["all_lang"] for r in rows], "cross_lingual_scheme_hit"
            ),
        }

    print("\n" + "=" * 72)
    print("CROSS-LANGUAGE RETRIEVAL")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print("=" * 72)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"cross_language_retrieval_{ts}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()