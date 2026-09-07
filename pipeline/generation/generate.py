"""
pipeline/generation/generate.py

Generate structured answers using Groq API + language-specific prompts.
Validates JSON with Pydantic; retries once on parse/validation failure.
"""

import json
import os
import re
from pathlib import Path
import time

from dotenv import load_dotenv
from groq import Groq

from pipeline.generation.schemas import StructuredAnswer

load_dotenv()

# ---------- Config ----------
PROMPTS_DIR = Path("configs/prompts")
GROQ_MODEL = "openai/gpt-oss-120b"  # or "llama-3.3-70b-versatile" / "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.2

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REFUSAL_BY_LANG = {
    "en": "I don't have enough information to answer this question based on the available documents.",
    "hi": "उपलब्ध दस्तावेजों के आधार पर मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
    "mr": "उपलब्ध दस्तऐवजांच्या आधारे या प्रश्नाचे उत्तर देण्यासाठी माझ्याकडे पुरेशी माहिती नाही.",
}


def load_prompt(language: str) -> str:
    prompt_path = PROMPTS_DIR / language / "answer_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        scheme_label = chunk.get("scheme_title") or chunk.get("scheme_id")
        header = (
            f"[Source {i}] "
            f"Scheme: {scheme_label} | "
            f"Section: {chunk.get('section')} | "
            f"Language: {chunk.get('language')}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict:
    """Parse JSON from model output; strip markdown fences if present."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # If model prepends junk, try first { ... } block
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    return json.loads(text)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    language: str = "en",
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Generate a structured answer using Groq.

    Args:
        query: User question
        retrieved_chunks: Chunks from retrieve()
        language: "en" | "hi" | "mr"
        temperature: Sampling temperature (0.0 = deterministic, higher = more variance).
                     Default 0.2 for production; pass 0.0 / 0.7 for variance tests.

    Returns:
        {
            "answer": str,
            "citations": list[dict],
            "language": str,
            "confidence": str,
            "cited_chunks": list[dict],
            "raw_valid": bool,
            "parse_error": str | None,
            "temperature": float,
        }
    """
    refusal = REFUSAL_BY_LANG.get(language, REFUSAL_BY_LANG["en"])

    if not retrieved_chunks:
        return {
            "answer": refusal,
            "citations": [],
            "language": language,
            "confidence": "low",
            "cited_chunks": [],
            "raw_valid": True,
            "parse_error": None,
            "temperature": temperature,
            "timings_ms": {
                "llm_ms": 0.0,
                "parse_validate_ms": 0.0,
                "generation_total_ms": 0.0,
            },
        }

    prompt_template = load_prompt(language)
    context = format_context(retrieved_chunks)
    full_prompt = prompt_template.format(context=context, query=query)

    last_error = None
    raw = ""
    llm_ms = 0.0
    parse_ms = 0.0

    for attempt in range(2):
        messages = [{"role": "user", "content": full_prompt}]
        if attempt == 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Previous output was invalid JSON. "
                        "Reply with ONLY valid JSON matching the required schema. "
                        "No markdown, no extra text."
                    ),
                }
            )

        t_llm0 = time.perf_counter()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        llm_ms += (time.perf_counter() - t_llm0) * 1000
        raw = response.choices[0].message.content.strip()

        t_parse0 = time.perf_counter()
        try:
            data = _extract_json(raw)
            parsed = StructuredAnswer.model_validate(data)
            parse_ms += (time.perf_counter() - t_parse0) * 1000
            return {
                "answer": parsed.answer,
                "citations": [c.model_dump() for c in parsed.citations],
                "language": parsed.language,
                "confidence": parsed.confidence,
                "cited_chunks": retrieved_chunks,
                "raw_valid": True,
                "parse_error": None,
                "temperature": temperature,
                "timings_ms": {
                    "llm_ms": round(llm_ms, 2),
                    "parse_validate_ms": round(parse_ms, 2),
                    "generation_total_ms": round(llm_ms + parse_ms, 2),
                },
            }
        except Exception as e:
            parse_ms += (time.perf_counter() - t_parse0) * 1000
            last_error = e
            continue

    # Graceful failure after retry
    return {
        "answer": raw if raw else refusal,
        "citations": [],
        "language": language,
        "confidence": "low",
        "cited_chunks": retrieved_chunks,
        "raw_valid": False,
        "parse_error": str(last_error),
        "temperature": temperature,
        "timings_ms": {
            "llm_ms": round(llm_ms, 2),
            "parse_validate_ms": round(parse_ms, 2),
            "generation_total_ms": round(llm_ms + parse_ms, 2),
        },
    }
