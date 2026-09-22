"""
api/app.py — FastAPI backend for the scheme RAG chatbot.

Run from project root:
    uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.generation.generate import generate_answer
from pipeline.retrieval.retriever import retrieve

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Scheme RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    language: Literal["en", "hi", "mr"] = "en"
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    language: str
    confidence: str
    raw_valid: bool
    n_retrieved: int
    latency_ms: float
    sources: list[dict[str, Any]]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.message.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty message")

    t0 = time.perf_counter()
    try:
        chunks = retrieve(query=q, language=req.language, top_k=req.top_k)
        result = generate_answer(
            query=q,
            retrieved_chunks=chunks,
            language=req.language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = [
        {
            "scheme_id": c.get("scheme_id"),
            "scheme_title": c.get("scheme_title"),
            "section": c.get("section"),
            "language": c.get("language"),
            "matched_via": c.get("matched_via"),
        }
        for c in chunks
    ]

    return ChatResponse(
        answer=(result.get("answer") or "").strip(),
        citations=result.get("citations") or [],
        language=result.get("language") or req.language,
        confidence=result.get("confidence") or "low",
        raw_valid=bool(result.get("raw_valid")),
        n_retrieved=len(chunks),
        latency_ms=latency_ms,
        sources=sources,
    )


# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
