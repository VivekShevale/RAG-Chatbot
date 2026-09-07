"""
Cross-encoder rerank on top of vector ( + section-intent ) candidates.
"""

from sentence_transformers import CrossEncoder

# Start with a small English-strong model for experiments.
# For hi/mr, try a multilingual cross-encoder later if quality is weak.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            item["matched_via"] = (
                f"{chunk.get('matched_via', 'vector')}+rerank"
            )
            ranked.append(item)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]