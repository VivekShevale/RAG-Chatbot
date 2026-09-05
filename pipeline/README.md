# pipeline/

Core application code, split by responsibility:

- `ingestion/` — Phase A: chunking (calls scripts/chunk_schemes.py), embedding generation, vector store loading
- `retrieval/` — Phase A/B: top-k retrieval, hybrid BM25+vector search, cross-encoder reranking
- `generation/` — Phase A/B: prompt assembly (reads from configs/prompts/), LLM calls via Ollama, citation enforcement, JSON schema validation + retry
- `voice/` — Phase A/B/D: ASR (Whisper), TTS (Piper/ElevenLabs), WebSocket orchestration server, per-stage latency instrumentation, timeout/fallback resilience logic
