"""
eval/golden_dataset/build_golden_dataset.py

Builds the Phase C golden evaluation dataset from real scheme data —
NOT fabricated. Two sources of questions, kept clearly labeled:

1. "faq" source: pulled verbatim from each scheme's real FAQ section.
   The expected_answer is the actual government-provided answer text —
   zero fabrication risk, since it's copied directly from your own
   verified source documents.

2. "template" source: a fixed set of 5 question patterns per scheme
   (eligibility / documents_required / benefit / application_process /
   details), phrased using the SAME wording style already validated
   against your real 15-question manual eval. These intentionally have
   NO pre-written expected_answer — grading them relies on (a) whether
   retrieval surfaced the correct scheme+section, and (b) an LLM-judged
   faithfulness check against whatever context was actually retrieved
   (see eval/scripts/run_eval.py). This avoids fabricating scheme facts
   I can't independently verify.

Run from project root:
    python eval/golden_dataset/build_golden_dataset.py

Output: eval/golden_dataset/golden_questions.json
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from scripts.chunk_schemes import split_into_sections, parse_faq_section, LANG_FILES

DATA_DIR = Path("data/schemes")
OUTPUT_PATH = Path("eval/golden_dataset/golden_questions.json")

# Max FAQs pulled per scheme per language. Schemes have varying FAQ counts;
# capping keeps the dataset from being dominated by verbatim FAQ matches
# (too easy / not representative of real user phrasing) relative to the
# harder templated questions.
MAX_FAQS_PER_SCHEME_LANG = 3

# Question templates per language. Each maps a target section to a
# question pattern with {scheme} as the placeholder for the scheme's
# title in that language (captured during chunking — see chunk_schemes.py).
# Phrasing deliberately matches patterns already validated against the
# real 15-question manual eval (see docs/RETRIEVAL_DEBUGGING_CASE_STUDY.md)
# so these are known-good triggers for section-intent routing, not novel
# untested phrasings.
QUESTION_TEMPLATES = {
    "en": {
        "eligibility": "Who is eligible for {scheme}?",
        "documents_required": "What documents are required for {scheme}?",
        "benefit": "What is the benefit under {scheme}?",
        "application_process": "How do I apply for {scheme}?",
        "details": "What is {scheme}?",
    },
    "hi": {
        "eligibility": "{scheme} के लिए कौन पात्र है?",
        "documents_required": "{scheme} के लिए कौन से दस्तावेज चाहिए?",
        "benefit": "{scheme} का लाभ क्या है?",
        "application_process": "{scheme} के लिए आवेदन कैसे करें?",
        "details": "{scheme} क्या है?",
    },
    "mr": {
        "eligibility": "{scheme} साठी कोण पात्र आहे?",
        "documents_required": "{scheme} साठी कोणती कागदपत्रे लागतात?",
        "benefit": "{scheme} चा लाभ काय आहे?",
        "application_process": "{scheme} साठी अर्ज कसा करावा?",
        "details": "{scheme} म्हणजे काय?",
    },
}


def build_faq_questions(scheme_id: str, language: str, title: str, sections: dict) -> list[dict]:
    entries = []
    faq_body = sections.get("faq", "")
    if not faq_body:
        return entries

    faqs = parse_faq_section(faq_body)
    for faq_index, question, answer in faqs[:MAX_FAQS_PER_SCHEME_LANG]:
        entries.append({
            "id": f"{language}_faq_{scheme_id}_q{faq_index}",
            "language": language,
            "scheme_id": scheme_id,
            "scheme_title": title,
            "expected_section": "faq",
            "question": question,
            "expected_answer": answer,   # verbatim from source — verified ground truth
            "source": "faq",
            "difficulty": "easy",
        })
    return entries


def build_template_questions(scheme_id: str, language: str, title: str, sections: dict) -> list[dict]:
    entries = []
    templates = QUESTION_TEMPLATES.get(language, {})
    for section, template in templates.items():
        if section not in sections or not sections[section].strip():
            continue  # don't generate a question for a section that doesn't exist
        question = template.format(scheme=title)
        entries.append({
            "id": f"{language}_template_{scheme_id}_{section}",
            "language": language,
            "scheme_id": scheme_id,
            "scheme_title": title,
            "expected_section": section,
            "question": question,
            "expected_answer": None,  # graded via retrieval-accuracy + LLM faithfulness, not string match
            "source": "template",
            "difficulty": "medium",
        })
    return entries


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Expected data directory not found: {DATA_DIR.resolve()}")

    all_entries = []
    scheme_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    for scheme_dir in scheme_dirs:
        scheme_id = scheme_dir.name
        if scheme_id == "scheme_01_example_scheme_name":
            continue  # skip the placeholder template folder

        print(f"Processing {scheme_id} ...")
        scheme_entry_count = 0

        for language, filename in LANG_FILES.items():
            file_path = scheme_dir / filename
            if not file_path.exists():
                print(f"  [warn] missing {filename}, skipping this language")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            title, sections = split_into_sections(text)
            if not title:
                print(f"  [warn] no title found for {language}, skipping this language")
                continue

            faq_entries = build_faq_questions(scheme_id, language, title, sections)
            template_entries = build_template_questions(scheme_id, language, title, sections)

            all_entries.extend(faq_entries)
            all_entries.extend(template_entries)
            scheme_entry_count += len(faq_entries) + len(template_entries)

        print(f"  -> {scheme_entry_count} questions generated")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Summary
    by_lang = {}
    by_source = {}
    by_section = {}
    for e in all_entries:
        by_lang[e["language"]] = by_lang.get(e["language"], 0) + 1
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        by_section[e["expected_section"]] = by_section.get(e["expected_section"], 0) + 1

    print(f"\nDone. {len(all_entries)} total questions written to {OUTPUT_PATH}")
    print(f"By language: {by_lang}")
    print(f"By source:   {by_source}")
    print(f"By section:  {by_section}")


if __name__ == "__main__":
    main()