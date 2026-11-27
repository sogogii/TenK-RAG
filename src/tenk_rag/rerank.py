"""Rerank retrieved chunks with a cross-encoder."""

from sentence_transformers import CrossEncoder

from tenk_rag.config import RERANK_MODEL
from tenk_rag.embed import pick_device


class Reranker:
    """Scores each (question, chunk) pair together and reorders the results."""

    def __init__(self, model_name=RERANK_MODEL):
        # Reads the question and chunk together as one input, so it can catch
        # fine-grained matches that separate embedding miss
        self.model = CrossEncoder(model_name, device=pick_device(), max_length=512)

    def rerank(self, query, results):
        """Reorder (chunk, score) pairs by cross-encoder relevance, best first."""
        pairs = [(query, chunk["text"]) for chunk, _ in results]
        # Scores every pair, 32 at a time
        scores = self.model.predict(pairs, batch_size=32)
        order = sorted(range(len(results)), key=lambda i: -scores[i])
        return [(results[i][0], float(scores[i])) for i in order]