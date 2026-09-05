"""
chunk_schemes.py

Reads scheme markdown files (en.md / hi.md / mr.md) from /data/schemes/<scheme_id>/
and produces a flat list of chunk objects ready for embedding.

Expected folder structure:
  data/schemes/
    scheme_01_postmatric_scholarship/
      en.md
      hi.md
      mr.md
      meta.json   (optional but recommended)

Expected markdown structure inside each language file:
  # Scheme Title
  ## Details
  ...
  ## Benefit
  ...
  ## Eligibility
  ...
  ## Application Process
  ...
  ## Documents Required
  ...
  ## FAQ
  **Q1: ...**
  A1: ...
  **Q2: ...**
  A2: ...

Output: data/processed/chunks.json
  A list of chunk dicts, each with:
    scheme_id, language, section, faq_index (or null), text, source_url
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(r"D:\RAG-Chatbot\data\schemes")
OUTPUT_PATH = Path(r"D:\RAG-Chatbot\data\processed\chunks.json")

# Section headers we expect. Matching is keyword/substring based (not exact
# equality) and language-aware, because scheme docs use English headers in
# en.md but Devanagari headers in hi.md / mr.md, and real-world copy-pasted
# headers sometimes contain stray extra whitespace or minor spelling variants
# (e.g. दस्तावेज़ vs दस्तावेज़, both with/without nukta).
#
# Each canonical section maps to a list of keywords; if ANY keyword appears
# as a substring of the (whitespace-normalized) header, it matches. Order
# matters — more specific / longer entries should come first to avoid a
# short keyword accidentally matching the wrong section.
SECTION_KEYWORDS = [
    ("details", ["details", "विवरण", "तपशील"]),
    ("benefit", ["benefit", "लाभ", "फायदे"]),
    ("eligibility", ["eligibility", "पात्रता"]),
    ("application_process", ["application process", "आवेदन", "अर्ज"]),
    ("documents_required", ["documents required", "दस्तावेज", "कागदपत्र"]),
    ("faq", ["faq"]),
]


def normalize_header(raw_header: str) -> str:
    """Collapses any run of whitespace (including stray double/triple spaces
    from copy-pasted source content) into a single space, and lowercases
    for case-insensitive matching of English headers."""
    return re.sub(r"\s+", " ", raw_header.strip()).lower()


def classify_section(raw_header: str) -> str:
    """Maps a raw section header (in any of the 3 languages) to a canonical
    section name. Falls back to a normalized/slugified version of the raw
    header if nothing matches, so unrecognized sections are still captured
    (visible in the by-section summary) rather than silently dropped."""
    normalized = normalize_header(raw_header)
    for canonical_name, keywords in SECTION_KEYWORDS:
        for keyword in keywords:
            if keyword.lower() in normalized:
                return canonical_name
    # Unrecognized header — keep it visible instead of losing the content
    fallback = re.sub(r"\s+", "_", raw_header.strip())
    print(f"    [warn] unrecognized section header '{raw_header.strip()}' "
          f"-> keeping as '{fallback}' (add a keyword to SECTION_KEYWORDS if this should map elsewhere)")
    return fallback

LANG_FILES = {
    "en": "en.md",
    "hi": "hi.md",
    "mr": "mr.md",
}

# Matches lines like: **Q1: What is X?**  followed by a line like A1: ...
FAQ_Q_PATTERN = re.compile(r"^\*\*Q(\d+):\s*(.+?)\*\*\s*$")
FAQ_A_PATTERN = re.compile(r"^A(\d+):\s*(.+)$")


def split_into_sections(markdown_text: str) -> dict:
    """
    Splits a markdown file into a dict of {normalized_section_name: section_body_text}.
    Ignores the top-level '# Title' line (title is pulled from meta.json instead,
    but we don't fail if meta.json is missing).
    """
    lines = markdown_text.splitlines()
    sections = {}
    current_section = None
    buffer = []

    def flush():
        if current_section is not None:
            sections[current_section] = "\n".join(buffer).strip()

    for line in lines:
        header_match = re.match(r"^##\s+(.+)$", line.strip())
        if header_match:
            flush()
            raw_name = header_match.group(1)
            current_section = classify_section(raw_name)
            buffer = []
        elif line.strip().startswith("# "):
            # top-level title line, skip — not a section
            continue
        else:
            if current_section is not None:
                buffer.append(line)

    flush()
    return sections


def parse_faq_section(faq_text: str) -> list:
    """
    Parses the FAQ section body into a list of (faq_index, question, answer) tuples.
    Expects alternating **Qn: ...** / An: ... lines, possibly with blank lines between.
    """
    faqs = []
    lines = [l for l in faq_text.splitlines() if l.strip() != ""]

    current_q_index = None
    current_question = None

    for line in lines:
        q_match = FAQ_Q_PATTERN.match(line.strip())
        a_match = FAQ_A_PATTERN.match(line.strip())

        if q_match:
            current_q_index = int(q_match.group(1))
            current_question = q_match.group(2).strip()
        elif a_match and current_question is not None:
            answer = a_match.group(2).strip()
            faqs.append((current_q_index, current_question, answer))
            current_q_index = None
            current_question = None

    return faqs


def build_chunks_for_scheme(scheme_dir: Path) -> list:
    scheme_id = scheme_dir.name

    meta_path = scheme_dir / "meta.json"
    source_urls = {"en": None, "hi": None, "mr": None}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                url = meta.get("source_url")
                source_urls = {"en": url, "hi": url, "mr": url}
            else:
                print(f"  [warn] {scheme_id}: meta.json is not a JSON object (got {type(meta).__name__}) "
                      f"— expected {{...}} not [...]. source_url will be null for this scheme.")
        except json.JSONDecodeError as e:
            print(f"  [warn] {scheme_id}: meta.json is invalid JSON ({e}) — source_url will be null.")
    else:
        print(f"  [warn] no meta.json found for {scheme_id} — source_url will be null")

    chunks = []

    for lang, filename in LANG_FILES.items():
        file_path = scheme_dir / filename
        if not file_path.exists():
            print(f"  [warn] {scheme_id}: missing {filename}, skipping this language")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        sections = split_into_sections(text)

        for section_name, section_body in sections.items():
            if section_name == "faq":
                faqs = parse_faq_section(section_body)
                if not faqs:
                    print(f"  [warn] {scheme_id}/{lang}: FAQ section found but no Q&A pairs parsed — check formatting")
                for faq_index, question, answer in faqs:
                    chunks.append({
                        "scheme_id": scheme_id,
                        "language": lang,
                        "section": "faq",
                        "faq_index": faq_index,
                        "text": f"Q: {question}\nA: {answer}",
                        "source_url": source_urls.get(lang),
                    })
            else:
                if section_body.strip() == "":
                    continue
                chunks.append({
                    "scheme_id": scheme_id,
                    "language": lang,
                    "section": section_name,
                    "faq_index": None,
                    "text": section_body.strip(),
                    "source_url": source_urls.get(lang),
                })

    return chunks


def main():
    all_chunks = []

    if not DATA_DIR.exists():
        raise SystemExit(f"Expected data directory not found: {DATA_DIR.resolve()}")

    scheme_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    if not scheme_dirs:
        raise SystemExit(f"No scheme folders found inside {DATA_DIR.resolve()}")

    for scheme_dir in scheme_dirs:
        print(f"Processing {scheme_dir.name} ...")
        chunks = build_chunks_for_scheme(scheme_dir)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_chunks)} total chunks written to {OUTPUT_PATH}")

    # Quick sanity summary
    by_lang = {}
    by_section = {}
    for c in all_chunks:
        by_lang[c["language"]] = by_lang.get(c["language"], 0) + 1
        by_section[c["section"]] = by_section.get(c["section"], 0) + 1

    print("\nChunks by language:", by_lang)
    print("Chunks by section:", by_section)


if __name__ == "__main__":
    main()
