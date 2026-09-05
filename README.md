# RAG-Chatbot: Multilingual Voice Knowledge Assistant

A production-style RAG system that answers voice questions about Maharashtra
government welfare schemes in **English, Hindi, and Marathi**, with
citation-grounded answers, hybrid retrieval, observability, and a
fine-tuned local model.

> Status: 🚧 In progress — see `docs/CHECKLIST.md` for build phases.

## What This Project Demonstrates
- **RAG**: hybrid (BM25 + vector) retrieval, reranking, citation enforcement, CI-gated evaluation
- **Local inference**: small open models run fully offline via Ollama, benchmarked across 3 models
- **Observability**: full request tracing, P50/P95 latency, cost, and failure-rate dashboards
- **Fine-tuning**: LoRA/QLoRA SFT + DPO preference tuning, with measurable before/after improvement per language
- **Real-time multimodal**: streaming voice pipeline (ASR → RAG → TTS) with a full latency budget breakdown and graceful failure handling

## Repo Structure
See `docs/REPO_STRUCTURE.md` for the full annotated folder layout.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in real values
```

## Quick Start
```bash
# 1. Chunk the raw scheme documents into structured pieces
python scripts/chunk_schemes.py

# 2. Build embeddings + load into the vector store
python pipeline/ingestion/build_index.py

# 3. Run a text query against the pipeline
python pipeline/retrieval/query.py --lang en --q "Am I eligible for the scholarship?"

# 4. Start the voice assistant server
python pipeline/voice/server.py
```

## Results
_(Fill this in as Phase C completes — model comparison table, fine-tuning
before/after numbers, per-language faithfulness scores.)_

## License
_(Add a license once you're ready to make this public.)_
