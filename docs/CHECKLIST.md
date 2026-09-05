# Multilingual Voice Knowledge Assistant — Full Build Checklist
### (English + Hindi + Marathi | Combines all 5 portfolio project phases)

Work top to bottom. Each phase builds on the last. Check items off as you complete them — when everything is checked, the project covers RAG, local inference, observability, fine-tuning, and real-time multimodal, end to end.

---

## PHASE 0 — Setup & Planning

- [ ] Pick your domain corpus (must be sourceable in English; Hindi/Marathi via existing docs or translation)
- [ ] Confirm corpus size is reasonable (aim for 20–100 documents to start)
- [ ] Set up project repo with folders: `/data`, `/pipeline`, `/eval`, `/finetune`, `/observability`, `/frontend`
- [ ] Set up Python environment (venv/conda) and dependency file (`requirements.txt` or `pyproject.toml`)
- [ ] Install Ollama locally
- [ ] Install Docker (needed later for self-hosted Langfuse)
- [ ] Create a `README.md` skeleton — you'll fill this in as you go (this becomes your portfolio write-up)
- [ ] Set up a GitHub repo with a proper `.gitignore` (exclude models, embeddings, `.env`)

---

## PHASE A — Foundations (End-to-End Pipeline, One Pass, All 3 Languages)

### A1. Document Ingestion & Chunking
- [ ] Write ingestion script for PDF/markdown/web sources
- [ ] Implement chunking: 500–800 tokens per chunk, ~100 token overlap
- [ ] Tag each chunk with source document + language metadata
- [ ] Store raw chunks (pre-embedding) in a simple JSON/SQLite file for inspection

### A2. Embeddings & Vector Store
- [ ] Choose a multilingual embedding model (e.g., `multilingual-e5-large`)
- [ ] Generate embeddings for all chunks
- [ ] Set up ChromaDB and load embeddings
- [ ] Sanity check: run a manual English query, confirm relevant chunks return
- [ ] Sanity check: run a manual Hindi query, confirm relevant chunks return (cross-lingual retrieval test)
- [ ] Sanity check: run a manual Marathi query, confirm relevant chunks return

### A3. Basic Retrieval + Answer Generation
- [ ] Build top-k retrieval function
- [ ] Build prompt template that instructs the LLM to answer only from retrieved chunks
- [ ] Add citation output (which chunk/source supported the answer)
- [ ] Test manually: show a doc, show the generated answer, show the exact cited paragraph

### A4. Local LLM Setup
- [ ] Pull a multilingual-capable model via Ollama (e.g., Qwen 2.5/3 7B)
- [ ] Build a CLI or FastAPI wrapper around the model
- [ ] Benchmark: tokens/sec, time-to-first-token, total latency (English baseline)
- [ ] Repeat benchmark for Hindi and Marathi prompts — log any differences
- [ ] Document baseline numbers in `README.md`

### A5. Voice Pipeline (ASR + TTS)
- [ ] Set up Whisper locally for ASR
- [ ] Confirm language auto-detection works across English/Hindi/Marathi test clips
- [ ] Set up Piper TTS (or ElevenLabs free tier) with voices for each language
- [ ] Set up WebSocket server to orchestrate: audio in → ASR → RAG/LLM → TTS → audio out
- [ ] Build a minimal frontend or CLI client to speak to the assistant and hear a response

### A6. Phase A Milestone
- [ ] Can ask a spoken question in English and get a cited, spoken answer
- [ ] Can ask a spoken question in Hindi and get a cited, spoken answer
- [ ] Can ask a spoken question in Marathi and get a cited, spoken answer
- [ ] Record a short demo video/GIF of this working (for portfolio use later)

---

## PHASE B — Production Quality

### B1. Hybrid Retrieval
- [ ] Implement BM25 keyword search alongside vector search
- [ ] Test per-language tokenization for BM25 (Marathi/Hindi need different tokenizers than English)
- [ ] Combine BM25 + vector scores (e.g., reciprocal rank fusion or weighted sum)
- [ ] Compare hybrid vs. vector-only retrieval quality per language, note findings

### B2. Reranking
- [ ] Add a cross-encoder reranker (Sentence Transformers) on top of retrieved chunks
- [ ] Check if a multilingual reranker is needed vs. English-only model — test and document
- [ ] Measure precision improvement before/after reranking (qualitative or with a small labeled set)

### B3. Citation Enforcement
- [ ] Implement logic: if retrieved chunks don't support an answer, respond with an explicit "I don't have enough information" instead of guessing
- [ ] Test refusal behavior in English
- [ ] Test refusal behavior in Hindi
- [ ] Test refusal behavior in Marathi
- [ ] Note any language where refusal quality is noticeably worse

### B4. Structured Output & Reliability
- [ ] Define a JSON schema for responses: `{answer, citations, language, confidence}`
- [ ] Validate LLM output against schema using Pydantic
- [ ] Implement retry-once-then-fail-gracefully logic for invalid JSON
- [ ] Test temperature 0 vs. 0.7 on a fixed prompt set, log output variance
- [ ] Repeat temperature test for Hindi and Marathi prompts, compare variance across languages

### B5. Prompt Versioning
- [ ] Move all prompts into a version-controlled config file (not hardcoded in scripts)
- [ ] Create separate prompt variants per language if needed (don't assume literal translation works)
- [ ] Add a changelog note format for prompt edits

### B6. Latency Budget Breakdown
- [ ] Instrument timing around each pipeline stage: ASR, retrieval, rerank, LLM TTFT, TTS TTFB
- [ ] Log latency breakdown for every request
- [ ] Build a simple visualization (bar chart) of latency by stage
- [ ] Run the same test query set in all 3 languages, compare latency breakdown across languages
- [ ] Document findings (e.g., "Hindi ASR adds 300ms vs. English on average")

### B7. Phase B Milestone
- [ ] Have a table: language × (retrieval precision, refusal accuracy, latency breakdown, output validity rate)
- [ ] Update `README.md` with hybrid retrieval and reranking design decisions

---

## PHASE C — Evaluation, Model Comparison & Fine-Tuning

### C1. Golden Evaluation Dataset
- [ ] Write 50–200 Q&A pairs manually verified against your corpus
- [ ] Split roughly evenly across English, Hindi, Marathi
- [ ] Store as structured file (JSON/CSV) with question, expected answer, expected citation

### C2. Offline Faithfulness Evaluation
- [ ] Set up RAGAS (or equivalent) evaluation framework
- [ ] Run faithfulness evaluation on English subset, record score
- [ ] Run faithfulness evaluation on Hindi subset, record score
- [ ] Run faithfulness evaluation on Marathi subset, record score
- [ ] Identify weakest-performing language/component — this becomes your fine-tuning target

### C3. Model Comparison Study
- [ ] Select 3 candidate local models (e.g., Qwen 2.5 7B, Llama 3.2, Mistral 7B)
- [ ] Run all 3 on the same hardware using your 30–50 test prompt set
- [ ] Measure memory usage, tokens/sec, and output quality — per language
- [ ] Test GGUF quantized versions (Q4/Q5) of the winning model
- [ ] Document quality-vs-speed trade-off in a short report
- [ ] Pick your final model based on this data (not popularity)

### C4. Fine-Tuning Data Prep
- [ ] Identify the exact weak task (e.g., Hindi JSON extraction accuracy)
- [ ] Collect/generate 2,000–10,000 clean training examples
- [ ] Ensure examples cover English, Hindi, and Marathi proportionally
- [ ] Clean and validate formatting consistency across all examples
- [ ] Split into train / validation / held-out test sets

### C5. Supervised Fine-Tuning (SFT)
- [ ] Set up LoRA/QLoRA fine-tuning environment (Hugging Face TRL or Axolotl)
- [ ] Run SFT on Colab/Kaggle free GPU (or rented GPU if needed)
- [ ] Evaluate on held-out set: JSON validity rate, exact match accuracy, refusal correctness — per language
- [ ] Save training curve plots

### C6. Preference Tuning (DPO)
- [ ] Generate multiple outputs per prompt (good vs. worse) — per language
- [ ] Label preference pairs
- [ ] Run DPO-style training on top of the SFT checkpoint
- [ ] Re-evaluate on the same held-out set, compare against SFT-only baseline
- [ ] Document incremental improvement, per language

### C7. Phase C Milestone
- [ ] Before/after fine-tuning table, broken out by language (e.g., Hindi JSON accuracy 61% → 89%)
- [ ] Write up the model comparison + fine-tuning story in `README.md`

---

## PHASE D — Observability, CI Gating & Resilience

### D1. Tracing
- [ ] Set up Langfuse (self-hosted via Docker, or free cloud tier)
- [ ] Instrument every pipeline stage: transcript, detected language, retrieved chunks, reranked order, prompt sent, response, token counts
- [ ] Confirm traces are viewable per request in the Langfuse dashboard

### D2. Quality Metrics Dashboard
- [ ] Track latency at P50 and P95 (not just average)
- [ ] Track cost per request (even if $0 for local models, compute a "compute-time cost" proxy)
- [ ] Track citation coverage (% of answers grounded in retrieved evidence)
- [ ] Track failure rate (errors, unsupported responses)
- [ ] Segment all of the above by language
- [ ] Build/export a dashboard view you can screen-share

### D3. CI Regression Gating
- [ ] Set up a CI pipeline (GitHub Actions or similar)
- [ ] Wire the Phase C golden eval set into CI
- [ ] Fail the build automatically if faithfulness drops below a defined threshold, for any of the 3 languages
- [ ] Version prompts and configs alongside code so changes are tracked together
- [ ] Test the gate actually works: intentionally break something, confirm CI fails

### D4. Resilience & Failure Handling
- [ ] Implement timeout handling for ASR (fallback if it stalls)
- [ ] Implement timeout handling for LLM (fallback if it stalls)
- [ ] Implement timeout handling for TTS (fallback if it stalls)
- [ ] Implement graceful degradation (e.g., fall back to text-only response, or fall back to English if a language-specific component fails)
- [ ] Build a replay mode: feed a recorded session's audio/text back through the pipeline for debugging
- [ ] Test replay mode on a deliberately failed session

### D5. Phase D Milestone
- [ ] Live dashboard demo works and shows real traces from a live session
- [ ] Can walk through: "quality dropped for Marathi last run, here's the trace, here's why, here's the fix"
- [ ] CI is actively gating merges based on evaluation results

---

## PHASE E — Wrap-Up & Portfolio Packaging

- [ ] Finalize `README.md`: problem statement, architecture diagram, tech stack, key decisions, results
- [ ] Include the language comparison tables (latency, faithfulness, fine-tuning improvement) directly in the README
- [ ] Record a 2–3 minute demo video showing all 3 languages working end-to-end
- [ ] Add an architecture diagram (pipeline stages + where each project's contribution lives)
- [ ] Write a short "what I'd do with more time / known limitations" section (shows maturity)
- [ ] Clean up repo structure, add setup instructions so someone else could run it
- [ ] Push final version, tag a release (e.g., `v1.0`)
- [ ] Prepare a 2-minute verbal walkthrough for interviews, covering all 5 skill areas in one narrative

---

## Quick Reference: Which Original Project Each Phase Covers

| Checklist Phase | Original Projects Covered |
|---|---|
| Phase A | Project 1 (Ph.1), Project 2 (Ph.1), Project 5 (Ph.1) |
| Phase B | Project 1 (Ph.2), Project 2 (Ph.2), Project 5 (Ph.2) |
| Phase C | Project 1 (Ph.3), Project 2 (Ph.3), Project 4 (Ph.1–2) |
| Phase D | Project 3 (all phases), Project 5 (Ph.3) |
| Phase E | Final packaging (not in original md, but required for portfolio impact) |

Once every box above is checked, you'll have one live, demoable, multilingual system that legitimately covers RAG, local/small-model inference, observability, fine-tuning, and real-time multimodal engineering — with real numbers to back up every claim.
