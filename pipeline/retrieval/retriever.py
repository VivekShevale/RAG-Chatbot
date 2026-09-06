"""
pipeline/retrieval/retriever.py

Simple vector retrieval over the ChromaDB index built by build_index.py
"""

from pathlib import Path
from typing import Optional

from chromadb import PersistentClient
from chromadb.utils import embedding_functions


# ---------- Config ----------
CHROMA_PATH = Path("data/processed/chroma_db")
COLLECTION_NAME = "scheme_chunks"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class Retriever:
    def __init__(self):
        self.client = PersistentClient(path=str(CHROMA_PATH))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = 5,
        search_all_languages: bool = False
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks.

        Args:
            query: User question
            language: "en" | "hi" | "mr"  (ignored if search_all_languages=True)
            top_k: Number of chunks to return
            search_all_languages: If True, search across all languages

        Returns:
            List of dicts, each containing:
            {
                "text": ...,
                "scheme_id": ...,
                "language": ...,
                "section": ...,
                "faq_index": ...,
                "source_url": ...,
                "score": ...          # distance (lower is better for cosine)
            }
        """
        where_filter = None
        if language and not search_all_languages:
            where_filter = {"language": language}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Flatten the results into a clean list of dicts
        retrieved = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            retrieved.append({
                "text": results["documents"][0][i],
                "scheme_id": meta.get("scheme_id"),
                "language": meta.get("language"),
                "section": meta.get("section"),
                "faq_index": None if meta.get("faq_index") == "none" else meta.get("faq_index"),
                "source_url": meta.get("source_url") or None,
                "score": results["distances"][0][i],
            })

        return retrieved


# Convenience function so you can also do:
# from pipeline.retrieval.retriever import retrieve
_retriever = None

def retrieve(
    query: str,
    language: Optional[str] = None,
    top_k: int = 5,
    search_all_languages: bool = False
) -> list[dict]:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever.retrieve(query, language, top_k, search_all_languages)