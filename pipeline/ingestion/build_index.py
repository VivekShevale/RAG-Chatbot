"""
pipeline/ingestion/build_index.py

Builds a persistent ChromaDB vector index from data/processed/chunks.json

IMPORTANT — E5 embedding prefixes:
intfloat/multilingual-e5-* models were trained with instruction prefixes:
"passage: " for indexed documents, "query: " for search queries. Skipping
this measurably hurts retrieval quality, especially for non-English text,
because the model relies on the prefix to align cross-lingual embeddings.

We can't let Chroma's built-in embedding_function auto-embed the raw
"documents" text (that would either bake "passage: " into the stored/
returned text, or skip the prefix entirely). Instead we embed manually with
the correct prefix, then hand Chroma pre-computed vectors — the stored
"documents" text stays clean (no prefix) for citations and LLM context.
"""

import argparse
import json
from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = Path("data/processed/chroma_db")   # persistent folder
COLLECTION_NAME = "scheme_chunks"

# Use large for better quality. Change to "intfloat/multilingual-e5-base" if too slow.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(rebuild: bool = False):
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.json not found at {CHUNKS_PATH.resolve()}")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    # Create persistent Chroma client
    client = PersistentClient(path=str(CHROMA_PATH))

    # Load the embedding model directly (not via Chroma's wrapper) so we
    # control exactly what text gets embedded (with the "passage: " prefix).
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, may take a while)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Handle rebuild
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # collection didn't exist

    # Get or create collection. No embedding_function attached here — we
    # supply our own pre-computed embeddings on every add() and query().
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine is standard for e5 models
    )

    # Prepare data for Chroma
    ids = []
    documents = []       # clean text, no prefix — this is what gets displayed/cited
    embedding_inputs = []  # "passage: " + text — this is what gets embedded
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{chunk['scheme_id']}_{chunk['language']}_{chunk['section']}_{chunk.get('faq_index') or 'none'}_{i}"

        ids.append(chunk_id)
        documents.append(chunk["text"])
        embedding_inputs.append(f"passage: {chunk['text']}")

        # Chroma metadata values must be str, int, float or bool
        metadatas.append({
            "scheme_id": str(chunk["scheme_id"]),
            "language": str(chunk["language"]),
            "section": str(chunk["section"]),
            "faq_index": str(chunk["faq_index"]) if chunk["faq_index"] is not None else "none",
            "source_url": str(chunk["source_url"]) if chunk["source_url"] else "",
        })

    # Add in batches (safer for large data)
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        batch_embeddings = model.encode(
            embedding_inputs[start:end],
            normalize_embeddings=True,   # cosine similarity expects normalized vectors
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
        )
        print(f"Added chunks {start} → {end-1}")

    print(f"\nIndex built successfully!")
    print(f"Total documents in collection: {collection.count()}")
    print(f"Persisted at: {CHROMA_PATH.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index from chunks.json")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing collection and rebuild from scratch"
    )
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()