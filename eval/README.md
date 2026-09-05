# eval/

Phase C evaluation work, wired into CI in Phase D.

- `golden_dataset/` — 50–200 manually verified Q&A pairs, split by language (en/hi/mr), some FAQ-derived and some hand-written/harder
- `scripts/` — evaluation runner (RAGAS faithfulness scoring, JSON validity checks, etc.) — this is what .github/workflows/eval-gate.yml calls
- `reports/` — generated model comparison + fine-tuning before/after reports (gitignored; summarize key results in the top-level README instead)
