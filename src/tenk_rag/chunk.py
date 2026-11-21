"""Split parsed 10-K blocks into retrieval-sized chunks."""

import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# Saves all companies' chunks to a single file, since the embedding step processes them all together
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"

MAX_WORDS = 150   # numbers use extra tokens, so stay well under the model's 256-token limit
OVERLAP_WORDS = 30   # words shared between consecutive text chunks
MAX_CONTEXT_WORDS = 40   # table context kept closest to the table (often holds years and units)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)   # loads the same tokenizer the embedding model uses
MAX_TOKENS = 250   # model reads 256 tokens, including 2 special tokens


def n_words(text):
	"""Count words, ignoring markdown table symbols."""
	return sum(1 for w in text.split() if w not in ("|", "---"))


def n_tokens(text):
	"""Count tokens the way the embedding model does."""
	return len(TOKENIZER.tokenize(text)) + 2   # +2 for the [CLS] and [SEP] tokens


def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
	"""Split text into overlapping windows of at most max_words words."""
	words = text.split()
	chunks = []
	start = 0
	while start < len(words):
		end = start + max_words
		chunks.append(" ".join(words[start:end]))
		if end >= len(words):
			break
		start = end - overlap   # step back so a sentence cut at the boundary appears in both chunks
	return chunks


def chunk_table(block, max_tokens=MAX_TOKENS):
	"""Split a table by rows, repeating its context and header in every chunk.
	
	Sizes are measured in model tokens, since pipes and numbers use many tokens.
	"""
	lines = block["content"].splitlines()
	header, rows = lines[:2], lines[2:]   # header line + separator, then data rows
	context = " ".join(block["context"].split()[-MAX_CONTEXT_WORDS:])
	prefix = [context] + header

	# Fallback: prose laid out as a table
	if n_tokens("\n".join(prefix)) > max_tokens // 2 or any(n_tokens(r) > max_tokens for r in rows):
		return chunk_text(block["context"] + " " + block["content"].replace("|", " "))

	chunks = []
	current = []
	for row in rows:
		candidate = current + [row]
		if current and n_tokens("\n".join(prefix + candidate)) > max_tokens:
			chunks.append("\n".join(prefix + current))
			current = [row]
		else:
			current = candidate

	if current or not rows:   # last chunk, or a header-only table
		chunks.append("\n".join(prefix + current))
	return chunks



def chunk_document(doc):
	"""Turn a parsed 10-K into chunks with metadata for search and citations."""
	chunks = []
	for i, block in enumerate(doc["blocks"]):
		if block["type"] == "text":
			pieces = chunk_text(block["content"])
		else:
			pieces = chunk_table(block)

		for j, text in enumerate(pieces):
			chunks.append({
				"id": f"{doc['ticker']}-{i}-{j}",   # e.g. "AAPL-42-1" = block 42, piece 1
				"ticker": doc["ticker"],
				"filing_date": doc["filing_date"],
				"type": block["type"],
				"text": text
			})
	return chunks


def main():
	"""Chunk every parsed filing and save all chunks to one JSONL file."""
	all_chunks = []
	for path in sorted(PROCESSED_DIR.glob("*.json")):
		doc = json.loads(path.read_text(encoding="utf-8"))
		chunks = chunk_document(doc)
		all_chunks.extend(chunks)

		n_table = sum(c["type"] == "table" for c in chunks)
		print(f"{doc['ticker']}: {len(chunks)} chunks ({n_table} table)")

	# JSONL: one JSON object per line, easy to read back one chunk at a time
	with CHUNKS_PATH.open("w", encoding="utf-8") as f:
		for c in all_chunks:
			f.write(json.dumps(c, ensure_ascii=False) + "\n")

	sizes = [n_words(c["text"]) for c in all_chunks]
	print(f"\n{len(all_chunks)} chunks saved to {CHUNKS_PATH.relative_to(PROJECT_ROOT)}")
	print(f"words: min {min(sizes)}, median {statistics.median(sizes)}, max {max(sizes)}")


if __name__ == "__main__":
	main()