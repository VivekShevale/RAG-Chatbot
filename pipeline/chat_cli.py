"""
pipeline/chat_cli.py

Simple text-based chatbot CLI for the RAG pipeline.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer


def detect_language(text: str) -> str:
    """Very simple language detection based on script."""
    # Devanagari range (covers Hindi + Marathi)
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        # Rough heuristic: if it contains typical Marathi words, treat as Marathi
        marathi_markers = ["आहे", "काय", "कोण", "योजना", "मुले", "शाळा"]
        if any(m in text for m in marathi_markers):
            return "mr"
        return "hi"
    return "en"


def print_answer(result: dict):
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print(f"\nConfidence: {result.get('confidence', 'n/a')} | valid_json: {result.get('raw_valid', 'n/a')}")
    print("\nCITATIONS:")
    citations = result.get("citations") or []
    if citations:
        for i, c in enumerate(citations, 1):
            print(f"  {i}. {c.get('scheme_id')} | {c.get('section')} | {c.get('language')}")
    else:
        # fallback to retrieved chunks
        for i, chunk in enumerate(result.get("cited_chunks", [])[:3], 1):
            print(f"  {i}. {chunk.get('scheme_id')} | {chunk.get('section')} | {chunk.get('language')}")
    print("=" * 60 + "\n")


def main():
    print("Multilingual RAG Chatbot (English / Hindi / Marathi)")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # Detect language
        language = detect_language(query)
        print(f"[Detected language: {language}]")

        # Retrieve
        chunks = retrieve(query, language=language, top_k=5)

        if not chunks:
            print("No relevant information found.")
            continue

        # Generate
        result = generate_answer(query, chunks, language=language)
        print_answer(result)


if __name__ == "__main__":
    main()