"""Embed all chunks and save the vectors to disk."""

import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from tenk_rag.config import CHUNKS_PATH, EMBED_META_PATH, EMBED_MODEL, EMBEDDINGS_PATH, PROJECT_ROOT


def load_chunks():
	"""Read chunks from the JSONL file, one chunk per line."""
	with CHUNKS_PATH.open(encoding="utf-8") as f:
		return [json.loads(line) for line in f]


def pick_device():
	"""Use the MAC GPU when available."""
	return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
	chunks = load_chunks()
	model = SentenceTransformer(EMBED_MODEL, device=pick_device())

	start = time.time()
	embeddings = model.encode(
		[c["text"] for c in chunks],
		batch_size=64,
		normalize_embeddings=True,   # unit length, so cosine similarity = dot product
		show_progress_bar=True
	)
	np.save(EMBEDDINGS_PATH, embeddings)

	# Record what these vectors were built from, so stale embeddings can be detected
	meta = {"model": EMBED_MODEL, "n_chunks": len(chunks), "dim": int(embeddings.shape[1])}
	EMBED_META_PATH.write_text(json.dumps(meta, indent=2))

	print(f"{embeddings.shape} saved to {EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)} "
	      f"in {time.time() - start:.0f}s")


if __name__=="__main__":
	main()