# Repo Structure Reference

This structure is designed to hold the full project end to end — from raw
data collection through fine-tuning and observability — without needing to
reorganize later. Each folder is tagged with the build phase it belongs to
(see your project checklist: Phase A–E).

```
RAG-Chatbot/
│
├── data/                          [Phase A]
│   ├── schemes/                   Raw source docs, one folder per scheme
│   │   └── scheme_01_xxx/
│   │       ├── en.md
│   │       ├── hi.md
│   │       ├── mr.md
│   │       └── meta.json
│   ├── processed/                 Output of chunking script (chunks.json) — gitignored, regenerable
│   └── eval/                      Raw material used to build golden eval questions later
│
├── pipeline/                      Core application code
│   ├── ingestion/                 [Phase A] chunking, embedding, vector store loading
│   ├── retrieval/                 [Phase A/B] top-k retrieval, hybrid (BM25+vector), reranking
│   ├── generation/                [Phase A/B] prompt templates, LLM calls, citation enforcement, JSON validation
│   └── voice/                     [Phase A/B/D] ASR, TTS, WebSocket server, latency instrumentation, resilience/fallback logic
│
├── models/                        Local model wrapper code (NOT model weights — those are gitignored)
│
├── configs/                       [Phase B] Versioned, NOT hardcoded in scripts
│   ├── prompts/
│   │   ├── en/                    English prompt templates
│   │   ├── hi/                    Hindi prompt templates (not literal translations — tuned per language)
│   │   └── mr/                    Marathi prompt templates
│   └── models/                    Model configs (temperature, top-k, chosen model per environment)
│
├── finetune/                      [Phase C]
│   ├── data/                      Training examples (SFT + DPO pairs), per language
│   ├── sft/                       LoRA/QLoRA supervised fine-tuning scripts
│   ├── dpo/                       DPO preference-tuning scripts
│   └── checkpoints/                Trained model checkpoints — gitignored (too large for git)
│
├── eval/                          [Phase C]
│   ├── golden_dataset/            50–200 verified Q&A pairs, split by language
│   ├── scripts/                   Evaluation runner (e.g. RAGAS faithfulness scoring), used by CI
│   └── reports/                   Generated eval reports / model comparison results — gitignored
│
├── observability/                 [Phase D]
│   ├── langfuse/                  Self-hosted Langfuse config (docker-compose, etc.)
│   └── dashboards/                Exported dashboard views / screenshots for the portfolio write-up
│
├── frontend/                      Minimal client (CLI or simple web UI) to interact with the voice assistant
│
├── scripts/                       One-off / utility scripts (e.g. chunk_schemes.py)
│
├── tests/                         Unit tests for pipeline components
│
├── .github/
│   └── workflows/
│       └── eval-gate.yml          [Phase D] CI job: fails build if eval faithfulness drops below threshold
│
├── docs/
│   ├── architecture/              Architecture diagrams
│   ├── demo/                      Demo video/GIF for the portfolio (gitignored if large — link externally instead)
│   ├── CHECKLIST.md               Your full phase-by-phase build checklist
│   └── REPO_STRUCTURE.md          This file
│
├── .env.example                   Documents required environment variables (secrets never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Why This Structure Holds Up End to End
- **`data/` vs `pipeline/` separation**: raw content never mixes with code, so you can regenerate everything in `data/processed/` from `data/schemes/` at any time
- **`configs/prompts/` is language-first, not an afterthought**: this is where Phase B's "prompts are part of system architecture" principle lives — every prompt change is a diffable, reviewable commit
- **`finetune/checkpoints/` and `eval/reports/` are gitignored**: large binary/generated artifacts don't belong in git; document results in `README.md` instead, with the actual files kept locally or on a model registry
- **`.github/workflows/eval-gate.yml` exists from day one**: even though it won't do much until Phase C's golden dataset exists, having it in the repo from the start means you build toward CI gating rather than bolting it on at the end
- **`docs/CHECKLIST.md`**: keep the phase checklist we built earlier committed in the repo itself — it doubles as a visible project-management artifact, which is a nice thing for a reviewer to see
