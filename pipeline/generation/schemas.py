"""
pipeline/generation/schemas.py

Pydantic models for structured RAG responses.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    scheme_id: str
    section: str
    language: Optional[str] = None


class StructuredAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    language: Literal["en", "hi", "mr"]
    confidence: Literal["high", "medium", "low"] = "medium"