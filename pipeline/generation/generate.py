"""
pipeline/generation/generate.py

Generate structured answers using Groq API + language-specific prompts.
Validates JSON with Pydantic; retries once on parse/validation failure.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from pipeline.generation.schemas import StructuredAnswer

load_dotenv()

# ---------- Config ----------
PROMPTS_DIR = Path("configs/prompts")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
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
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    return json.loads(text)


def _empty_token_usage() -> dict:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    language: str = "en",
    temperature: float = DEFAULT_TEMPERATURE,
    model: str | None = None,
) -> dict:
    """
    Generate a structured answer using Groq.

    Returns dict with answer, citations, language, confidence, cited_chunks,
    raw_valid, parse_error, temperature, timings_ms, token_usage/usage, model.
    """
    refusal = REFUSAL_BY_LANG.get(language, REFUSAL_BY_LANG["en"])
    use_model = model or GROQ_MODEL

    if not retrieved_chunks:
        empty_usage = _empty_token_usage()
        return {
            "answer": refusal,
            "citations": [],
            "language": language,
            "confidence": "low",
            "cited_chunks": [],
            "raw_valid": True,
            "parse_error": None,
            "temperature": temperature,
            "model": use_model,
            "timings_ms": {
                "llm_ms": 0.0,
                "parse_validate_ms": 0.0,
                "generation_total_ms": 0.0,
            },
            "token_usage": empty_usage,
            "usage": empty_usage,  # alias for eval scripts
        }

    prompt_template = load_prompt(language)
    context = format_context(retrieved_chunks)
    full_prompt = prompt_template.format(context=context, query=query)

    last_error = None
    raw = ""
    llm_ms = 0.0
    parse_ms = 0.0

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

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
            model=use_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        llm_ms += (time.perf_counter() - t_llm0) * 1000.0
        raw = response.choices[0].message.content.strip()

        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
            completion_tokens = getattr(usage_obj, "completion_tokens", None)
            total_tokens = getattr(usage_obj, "total_tokens", None)

        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        t_parse0 = time.perf_counter()
        try:
            data = _extract_json(raw)
            parsed = StructuredAnswer.model_validate(data)
            parse_ms += (time.perf_counter() - t_parse0) * 1000.0
            return {
                "answer": parsed.answer,
                "citations": [c.model_dump() for c in parsed.citations],
                "language": parsed.language,
                "confidence": parsed.confidence,
                "cited_chunks": retrieved_chunks,
                "raw_valid": True,
                "parse_error": None,
                "temperature": temperature,
                "model": use_model,
                "timings_ms": {
                    "llm_ms": round(llm_ms, 2),
                    "parse_validate_ms": round(parse_ms, 2),
                    "generation_total_ms": round(llm_ms + parse_ms, 2),
                },
                "token_usage": token_usage,
                "usage": token_usage,  # alias for eval scripts
            }
        except Exception as e:
            parse_ms += (time.perf_counter() - t_parse0) * 1000.0
            last_error = e
            continue

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {
        "answer": raw if raw else refusal,
        "citations": [],
        "language": language,
        "confidence": "low",
        "cited_chunks": retrieved_chunks,
        "raw_valid": False,
        "parse_error": str(last_error),
        "temperature": temperature,
        "model": use_model,
        "timings_ms": {
            "llm_ms": round(llm_ms, 2),
            "parse_validate_ms": round(parse_ms, 2),
            "generation_total_ms": round(llm_ms + parse_ms, 2),
        },
        "token_usage": token_usage,
        "usage": token_usage,
    }