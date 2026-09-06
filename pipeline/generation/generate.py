"""
pipeline/generation/generate.py

Generate answers using Groq API + language-specific prompts.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------- Config ----------
PROMPTS_DIR = Path("configs/prompts")
GROQ_MODEL = "openai/gpt-oss-120b"   # or "llama-3.1-8b-instant" for faster/cheaper

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    language: str = "en"
) -> dict:
    """
    Generate an answer using Groq.

    Returns:
        {
            "answer": str,
            "cited_chunks": list[dict],
            "language": str
        }
    """
    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information to answer this question based on the available documents.",
            "cited_chunks": [],
            "language": language
        }

    prompt_template = load_prompt(language)
    context = format_context(retrieved_chunks)

    full_prompt = prompt_template.format(
        context=context,
        query=query
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2,      # low temperature = more factual
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "cited_chunks": retrieved_chunks,
        "language": language
    }