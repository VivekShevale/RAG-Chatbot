# Prompt changelog

All language variants (`en`, `hi`, `mr`) are edited together when possible.
Format: one entry per change set, newest first.

---

## v2 — 2026-09-06

**Scope:** `en/answer_prompt.txt`, `hi/answer_prompt.txt`, `mr/answer_prompt.txt`

**What changed:**
- Added structured JSON output instructions at the end of each prompt
- Required shape: `{answer, citations, language, confidence}`
- Explicit rule: return ONLY valid JSON (no markdown fences, no extra text)
- Refusal path: set `answer` to the language-specific refusal sentence and `citations` to `[]`

**Why:**
- Groundwork for Pydantic validation + retry in `pipeline/generation/generate.py`
- Makes answers machine-checkable for eval and later fine-tuning (JSON validity rate)

**Related code:**
- `pipeline/generation/schemas.py` (`StructuredAnswer`, `Citation`)
- `pipeline/generation/generate.py` (parse + validate + retry-once)

---

## v1 — initial

**Scope:** `en/answer_prompt.txt`, `hi/answer_prompt.txt`, `mr/answer_prompt.txt`

**What changed:**
- First version of language-specific answer prompts
- Rules: answer only from context, cite scheme + section, refuse when unsupported
- Reply in the target language (en / hi / mr)

**Why:**
- Phase A3 basic retrieval + answer generation