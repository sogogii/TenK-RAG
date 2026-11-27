"""Shared paths and settings used across the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBED_META_PATH = DATA_DIR / "embeddings_meta.json"

# Chunks sizes depends on this model's tokenizer, so changing it means re-chunking
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-haiku-5-5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge": "BAAI/bge-reranker-base",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}