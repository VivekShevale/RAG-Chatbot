# Case Study: Debugging Multilingual Retrieval Failures

This document records a real debugging process from this project — from
noticing a quality gap in manual evaluation, to root-causing it precisely,
to a targeted architectural fix. Kept here because the process itself is
as valuable a portfolio artifact as the final code.

## Starting Point

15-question manual eval (5 schemes × 3 languages, `eval/manual_eval.py`):

| Language | Result            |
| -------- | ----------------- |
| English  | 5/5 Good          |
| Hindi    | 3/5 Good, 2 Wrong |
| Marathi  | 3/5 Good, 2 Wrong |

English worked perfectly; Hindi and Marathi failed on structurally similar
questions (asking for `documents_required` or `eligibility` details).

## Problem 1: Missing E5 Instruction Prefixes

**Symptom:** Retrieval quality was noticeably weaker for Hindi/Marathi than English.

**Root cause:** `intfloat/multilingual-e5-large` requires a `"passage: "`
prefix on indexed text and a `"query: "` prefix on search queries — this is
part of how the model was trained, not optional formatting. The original
code embedded raw text with no prefix at all.

**Fix:** Embed manually with the correct prefixes, while keeping the
*stored/displayed* chunk text clean (no prefix leaking into citations or
LLM context). Implemented in `pipeline/ingestion/build_index.py` and
`pipeline/retrieval/retriever.py`.

**Result:** Partial improvement — fixed some cases, not all.

## Problem 2: Diagnosing "Still the Same Results" — Not Guessing Twice

After the prefix fix, results looked unchanged. Rather than guessing again,
built two diagnostic scripts (`scripts/debug_retrieval.py`,
`scripts/debug_rank_of_expected_chunk.py`) to answer precisely:

1. Does the expected chunk even exist in the index?
2. What is its **true rank** among all chunks in that language (not just top-5)?

**Finding:** The expected chunks existed and were close-but-outside the
cutoff — ranks 6, 7, 9, and 21 out of 73 chunks (top-5 was being returned).

## Problem 3: Root-Causing *Why* Those Chunks Ranked Low

Tested a hybrid BM25 (keyword) retrieval hypothesis against the real data
before writing any code. Two things emerged:

- **A tokenizer bug of my own making**: naive `\w+` regex tokenization
  shatters Devanagari words at combining marks (matras/virama), e.g.
  "दस्तावेज़" fragmented into meaningless pieces. Fixed with whitespace-based
  tokenization instead.
- **Even with a correct tokenizer, BM25 alone performed *worse*, not
  better** — one case had **zero keyword overlap** between the query and
  the target chunk. Root cause: `documents_required` and `eligibility`
  sections are bare fact-lists (bullet points of document names, criteria)
  that **never restate the scheme name or section type**. A question like
  "what documents are needed for X" shares almost no vocabulary with a
  chunk that's just `- Passport photo / - Aadhaar card / ...`.

This ruled out hybrid BM25 as *the* fix and pointed at the real problem:
missing context, not a retrieval algorithm choice.

## Problem 4: Contextual Chunk Augmentation

**Fix:** Capture each document's title (previously discarded during
chunking — `scripts/chunk_schemes.py`) and prepend
`"<scheme title>. <section label>: "` to the text used for **embedding
only** — never to the stored/cited text. This is the standard "contextual
retrieval" technique: giving the model back the context a human reader
would have had from seeing the page title and section heading.

**Result:** Fixed 1 of 4 remaining cases outright (Hindi eligibility
question moved into top-5). Confirmed real, measurable improvement — but
not a complete fix for the other 3 cases, since it's a *soft* nudge to
embedding similarity, not a guarantee.

## Problem 5: The Actual Exact Fix — Section-Intent Routing

**Key insight, validated against every single eval question (passing and
failing):** vector search reliably identifies the correct **scheme**
every time. It's specifically the **section** (documents_required vs.
eligibility vs. benefit, etc.) that's unreliable for bare fact-list content.

This corpus has exactly 6 known section types per scheme — a structural
fact a generic RAG pipeline doesn't usually get to exploit. Built a
deterministic routing layer on top of vector search
(`pipeline/retrieval/retriever.py`):

1. Vector search runs as before.
2. If the query's keywords unambiguously indicate one of 4 actionable
   sections, **deterministically fetch that exact chunk via metadata
   filter** and guarantee its inclusion — rather than hoping embedding
   similarity ranks it high enough.
3. Ambiguous/general questions ("what is scheme X") fall through to pure
   vector search, unchanged.

**A bug caught before shipping:** the first keyword list used bare
"कौन"/"कोण" ("who/which") for eligibility — but this word is *also* used
generically in "which documents" phrasing ("कौन से दस्तावेज"), causing
false ambiguous double-matches that cancelled each other out. Fixed by
using only unambiguous, section-specific keyword phrases, then validated
the fix against all 15 real eval questions before deploying it.

**Result:** All 4 remaining failing cases were confirmed (via simulation
against the actual pasted vector-search results) to now correctly
force-inject the right chunk. Final eval: **14 Good + 1 Partial / 15**,
up from 11 Good + 4 Wrong.

## Key Lessons

1. **Diagnose before fixing.** Every fix in this log was built on a
   specific, verified measurement (exact rank numbers, keyword overlap
   counts) rather than a plausible-sounding guess. Two "obvious" fixes
   (hybrid BM25, naive tokenization) were tested and found insufficient
   or actively wrong before being discarded.
2. **A domain's structure is an asset.** Generic RAG advice says "improve
   your embeddings." This corpus has a much stronger signal available —
   a small, fixed set of section types — that a deterministic routing
   layer can exploit far more reliably than tuning embeddings alone.
3. **Citation enforcement was already working correctly** throughout this
   process — even when retrieval failed, the system said "I don't have
   enough information" rather than hallucinating, which is exactly the
   intended fallback behavior from Phase B's design.

---

## Temperature variance (Phase B4)

Ran a fixed 9-prompt set (3 en / 3 hi / 3 mr) at temperature **0.0** and
**0.7**, 5 repeats each (90 runs total). Structured JSON validity stayed
**100%** at both temperatures — Pydantic validation + retry is solid.

Exact-match consistency (same prompt, repeated runs):

| Language × temp | Mean prompt consistency | Notes                           |
| --------------- | ----------------------- | ------------------------------- |
| hi @ 0.0        | 0.73                    | Most stable; one prompt was 1.0 |
| en @ 0.0        | 0.40                    | Facts stable, wording drifts    |
| mr @ 0.0        | 0.27                    | Weakest at temp 0               |
| * @ 0.7         | ~0.00–0.03              | Near-total surface variance     |

**Findings:**

- Temperature 0 is *mostly* deterministic, not bit-identical — paraphrases
  still appear (normal on many hosted APIs).
- Temperature 0.7 is unsuitable for factual scheme answers; use only if you
  want stylistic variety.
- Core facts (eligibility, document lists, ₹1,950 / 25%) stayed correct
  across languages and temperatures; variance was form, not hallucination.
- Hindi was the most stable language at temp 0; Marathi the least.

**Decision:** Prefer `temperature=0.0` for production RAG and evals (max
stability). Current default `0.2` is acceptable if slightly more natural
phrasing is desired.

Report: `eval/reports/temperature_variance_20260906_230214.json`.

---

## Latency budget (Phase B6)

Instrumented retrieval (vector + section-intent) and generation on a fixed
9-prompt set. Generation dominates end-to-end latency; retrieval stays
~250–300 ms P50 even when section-intent routing fires. Section-intent’s
extra `collection.get()` is cheap (~2 ms P50) — keep routing for quality.

### Initial sequential run (misleading)

First latency pass ran **all English, then Hindi, then Marathi**. That
design confounded language with API session position:

| Bucket  | total P50 (ms) | retr P50 | gen P50 |
| ------- | -------------- | -------- | ------- |
| Overall | 8524           | 280      | 8237    |
| en      | 1494           | 258      | 1204    |
| hi      | 8524           | 275      | 8240    |
| mr      | 11713          | 298      | 11444   |

It looked like hi/mr were 6–8× slower than English. Report:
`eval/reports/latency_breakdown_20260907_094717.json`.

### Token usage (sequential) — incomplete story

Logged Groq `prompt_tokens` / `completion_tokens` via `generate_answer`.
Completion lengths were similar across languages; sequential numbers still
suggested much lower tok/s for hi/mr. That mixed a real signal with the
same order confound.

Report: `eval/reports/token_usage_20260907_103524.json`.

### Revised finding — shuffled job order

Re-ran token and latency evals with **shuffled** `(prompt × repeat)` jobs
(`--seed 42`) so en/hi/mr are interleaved, plus a warmup call.

**Evidence against pure language slowdown:**

- Early jobs were often **~1–1.5 s across all three languages** (hi, mr, and en).
- Later jobs slowed down for **English as well** (several en runs at ~7 s gen).
- Hindi P50 in the shuffled latency run was **~2.4 s gen** (tok/s P50 ~128) —
  faster than English in that particular shuffle.

**What remains real:**

| Factor | Evidence |
| ------ | -------- |
| API session / throttling | Early-fast / late-slow pattern for every language |
| Large Marathi prompts | **mr_02** always ~15–25 s gen with ~2971 prompt tokens |
| Residual Marathi cost | mr median still higher on average, but not the old 6–8× story |
| Retrieval | Stable ~250–300 ms; not the budget owner |

**Method lesson:** For multilingual latency comparisons, shuffle language
order (or fully randomize jobs) and report P50/P95. Do not run languages
in fixed blocks.

Shuffled reports:

- `eval/reports/token_usage_20260907_105334.json`
- `eval/reports/latency_breakdown_20260907_105856.json`

**Mitigation direction (not implemented yet):**

- Document the tradeoff honestly in the portfolio write-up.
- For demos: reduce context size on long document-list queries (especially mr).
- Optionally use a faster model path for non-English if product latency matters.