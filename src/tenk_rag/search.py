"""Search one company's chunks by semantic similarity."""

import json
import sys 

import numpy as np
from sentence_transformers import SentenceTransformer

from tenk_rag.config import EMBED_META_PATH, EMBED_MODEL, EMBEDDINGS_PATH
from tenk_rag.embed import load_chunks, pick_device


class Searcher:
	"""Loads chunks, vectors and the model once, then answer many queries."""

	def __init__(self):
		self.chunks = load_chunks()
		self.embeddings = np.load(EMBEDDINGS_PATH)

		meta = json.loads(EMBED_META_PATH.read_text())
		# Stops if the chunks changed but the embeddings weren't built
		if meta["model"] != EMBED_MODEL or meta["n_chunks"] != len(self.chunks):
			raise RuntimeError("Embeddings are out of date. Run: python -m tenk_rag.embed")

		self.model = SentenceTransformer(EMBED_MODEL, device=pick_device())
		self.tickers = np.array([c["ticker"] for c in self.chunks])

	def search(self, query, ticker, k=5):
		"""Return the top-k (chunk, score) pairs for one company."""
		idx = np.where(self.tickers == ticker)[0]
		if len(idx) == 0:
			# Raises a clear error for an unknown ticker
			raise ValueError(f"No chunks for ticker {ticker}")

		q = self.model.encode(query, normalize_embeddings=True)
		scores = self.embeddings[idx] @ q
		top = np.argsort(-scores)[:k]
		return [(self.chunks[idx[i]], float(scores[i])) for i in top]


def main():
	if len(sys.argv) < 3:
		print('Usage: python -m tenk_rag.search TICKER "question"')
		return

	ticker, query = sys.argv[1].upper(), " ".join(sys.argv[2:])
	for chunk, score in Searcher().search(query, ticker):
		print(f"{score:.3f}  {chunk['id']:<12} {chunk['type']:<5}  {chunk['text'][:100]!r}")


if __name__=="__main__":
	main()