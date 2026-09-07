"""
pipeline/retrieval/retriever.py

Retrieval over the ChromaDB index built by build_index.py, with two layers:

1. VECTOR SEARCH — semantic retrieval via multilingual-e5. Reliable for
   identifying the correct SCHEME (details/faq chunks strongly mention the
   scheme name), but measurably unreliable for certain STRUCTURAL sections
   (documents_required, eligibility) — these are bare fact-lists that never
   restate the scheme name or section type, so they share little vocabulary
   with a natural question, even after contextual chunk augmentation
   (see build_index.py). Confirmed via manual eval + rank analysis.

2. SECTION-INTENT ROUTING — this corpus has exactly 6 known section types
   per scheme. When a query's intent maps unambiguously to one of the 4
   "actionable" sections (documents_required, eligibility, benefit,
   application_process) via keyword detection, we deterministically fetch
   that exact chunk via metadata filter and ensure it's included in the
   returned context — instead of hoping vector search ranks it highly
   enough. This guarantees correctness for domain-typical question
   patterns without relying purely on embedding quality.

   Ambiguous or non-matching queries (e.g. "what is scheme X") fall through
   to pure vector search, which already handles them well.

IMPORTANT — E5 embedding prefixes:
Every query must be embedded with a "query: " prefix to match how
build_index.py embedded documents with "passage: ". Mismatched prefixes
(or no prefix at all) measurably hurt retrieval quality, especially for
Hindi/Marathi — see docs/CHECKLIST.md eval notes for the manual-eval run
that surfaced this bug.
"""

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple
import time

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


# ---------- Config ----------
CHROMA_PATH = Path("data/processed/chroma_db")
COLLECTION_NAME = "scheme_chunks"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Keyword -> section intent mapping, per language. Only covers the 4
# "actionable" structural sections that are reliably keyword-identifiable
# and where vector search has shown measurable weakness. "details" and
# "faq" are deliberately excluded — vector search already handles general
# / open-ended questions about them correctly, and forcing them would risk
# false positives on unrelated queries.
#
# IMPORTANT: avoid bare generic interrogatives like "कौन"/"कोण" ("who/which")
# as standalone keywords — they appear in BOTH "who is eligible" (एलिजिबिलिटी)
# AND "which documents" (दस्तावेज़/कागदपत्रे) phrasing, causing ambiguous
# double-matches that cancel out (see debug notes / manual eval hi_02, mr_02).
# Use only unambiguous, section-specific keywords/phrases.
SECTION_INTENT_KEYWORDS = {
    "en": {
        "documents_required": ["document", "documents", "papers", "certificate"],
        "eligibility": ["eligible", "eligibility", "who can", "who is", "qualify"],
        "benefit": ["benefit", "how much", "amount", "assistance amount"],
        "application_process": ["apply", "application", "how to apply", "register", "registration"],
    },
    "hi": {
        "documents_required": ["दस्तावेज", "कागज", "प्रमाण पत्र"],
        "eligibility": ["पात्र", "पात्रता", "किसे", "किसको"],
        "benefit": ["लाभ", "राशि", "कितनी सहायता", "कितना पैसा"],
        "application_process": ["आवेदन", "अर्जी", "आवेदन प्रक्रिया"],
    },
    "mr": {
        "documents_required": ["कागदपत्र", "दस्तऐवज", "प्रमाणपत्र"],
        "eligibility": ["पात्र", "पात्रता", "कोणाला", "कुणाला"],
        "benefit": ["फायदा", "लाभ", "किती सहाय्य", "किती पैसे"],
        "application_process": ["अर्ज करण्याची", "नोंदणी", "अर्ज प्रक्रिया", "अर्ज कसा", "अर्ज"],
    },
}


def detect_section_intent(query: str, language: str) -> Optional[str]:
    """Returns a canonical section name if the query unambiguously maps to
    exactly one of the actionable sections, else None (ambiguous or no
    match -> caller should fall back to pure vector search)."""
    keywords_by_section = SECTION_INTENT_KEYWORDS.get(language, {})
    query_lower = query.lower()
    matched_sections = set()
    for section, keywords in keywords_by_section.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                matched_sections.add(section)
                break
    if len(matched_sections) == 1:
        return matched_sections.pop()
    return None


class Retriever:
    def __init__(self):
        self.client = PersistentClient(path=str(CHROMA_PATH))
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # No embedding_function passed — this collection was built with
        # manually pre-computed embeddings, and queries also supply their
        # own embedding vector (see retrieve() below).
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def _vector_search(self, query: str, language: Optional[str], top_k: int,
                        search_all_languages: bool) -> list[dict]:
        where_filter = None
        if language and not search_all_languages:
            where_filter = {"language": language}

        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if not results["ids"][0]:
            return retrieved

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            retrieved.append({
                "text": results["documents"][0][i],
                "scheme_id": meta.get("scheme_id"),
                "scheme_title": meta.get("scheme_title") or None,
                "language": meta.get("language"),
                "section": meta.get("section"),
                "faq_index": None if meta.get("faq_index") == "none" else meta.get("faq_index"),
                "source_url": meta.get("source_url") or None,
                "score": results["distances"][0][i],
                "matched_via": "vector",
            })
        return retrieved

    def _fetch_exact_chunk(self, scheme_id: str, section: str, language: str) -> Optional[dict]:
        """Deterministically fetch one chunk by exact metadata match. Used
        by section-intent routing to guarantee inclusion of a known-relevant
        chunk regardless of its vector similarity rank."""
        result = self.collection.get(
            where={
                "$and": [
                    {"scheme_id": scheme_id},
                    {"section": section},
                    {"language": language},
                ]
            },
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None

        meta = result["metadatas"][0]
        return {
            "text": result["documents"][0],
            "scheme_id": meta.get("scheme_id"),
            "scheme_title": meta.get("scheme_title") or None,
            "language": meta.get("language"),
            "section": meta.get("section"),
            "faq_index": None if meta.get("faq_index") == "none" else meta.get("faq_index"),
            "source_url": meta.get("source_url") or None,
            "score": None,  # not ranked — deterministically injected
            "matched_via": "section_intent_routing",
        }

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = 5,
        search_all_languages: bool = False,
    ) -> list[dict]:
        """Same public API as before — returns only chunks."""
        chunks, _ = self.retrieve_with_timings(
            query=query,
            language=language,
            top_k=top_k,
            search_all_languages=search_all_languages,
        )
        return chunks

    def retrieve_with_timings(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = 5,
        search_all_languages: bool = False,
    ) -> Tuple[list[dict], dict]:
        """
        Same retrieval logic as retrieve(), plus stage timings.

        Returns:
            (chunks, timings) where timings has:
              vector_search_ms, section_intent_ms,
              section_intent_triggered, retrieval_total_ms
        """
        t0 = time.perf_counter()

        t_vec0 = time.perf_counter()
        vector_results = self._vector_search(
            query, language, top_k, search_all_languages
        )
        vector_ms = (time.perf_counter() - t_vec0) * 1000

        # Cross-language / no language → vector only
        if search_all_languages or not language:
            total_ms = (time.perf_counter() - t0) * 1000
            timings = {
                "vector_search_ms": round(vector_ms, 2),
                "section_intent_ms": 0.0,
                "section_intent_triggered": False,
                "retrieval_total_ms": round(total_ms, 2),
            }
            return vector_results, timings

        t_si0 = time.perf_counter()
        intent_section = detect_section_intent(query, language)

        if not intent_section or not vector_results:
            section_intent_ms = (time.perf_counter() - t_si0) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            timings = {
                "vector_search_ms": round(vector_ms, 2),
                "section_intent_ms": round(section_intent_ms, 2),
                "section_intent_triggered": False,
                "retrieval_total_ms": round(total_ms, 2),
            }
            return vector_results, timings

        top_for_vote = vector_results[:3]
        scheme_votes = Counter(r["scheme_id"] for r in top_for_vote)
        anchor_scheme_id = scheme_votes.most_common(1)[0][0]

        already_present = any(
            r["scheme_id"] == anchor_scheme_id and r["section"] == intent_section
            for r in vector_results
        )
        if already_present:
            section_intent_ms = (time.perf_counter() - t_si0) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            timings = {
                "vector_search_ms": round(vector_ms, 2),
                "section_intent_ms": round(section_intent_ms, 2),
                "section_intent_triggered": True,
                "retrieval_total_ms": round(total_ms, 2),
            }
            return vector_results, timings

        forced_chunk = self._fetch_exact_chunk(
            anchor_scheme_id, intent_section, language
        )
        section_intent_ms = (time.perf_counter() - t_si0) * 1000
        section_intent_triggered = forced_chunk is not None

        results = (
            vector_results + [forced_chunk] if forced_chunk else vector_results
        )

        total_ms = (time.perf_counter() - t0) * 1000
        timings = {
            "vector_search_ms": round(vector_ms, 2),
            "section_intent_ms": round(section_intent_ms, 2),
            "section_intent_triggered": section_intent_triggered,
            "retrieval_total_ms": round(total_ms, 2),
        }
        return results, timings


# Convenience function so you can also do:
# from pipeline.retrieval.retriever import retrieve
_retriever = None

def retrieve(
    query: str,
    language: Optional[str] = None,
    top_k: int = 5,
    search_all_languages: bool = False,
) -> list[dict]:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever.retrieve(query, language, top_k, search_all_languages)


def retrieve_with_timings(
    query: str,
    language: Optional[str] = None,
    top_k: int = 5,
    search_all_languages: bool = False,
) -> Tuple[list[dict], dict]:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever.retrieve_with_timings(
        query, language, top_k, search_all_languages
    )