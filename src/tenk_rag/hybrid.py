"""Hybrid search: combine embedding and BM25 results with Reciprocal Rank Fusion."""

from tenk_rag.bm25 import BM25Searcher
from tenk_rag.search import Searcher

RRF_K = 60   # standard constant; dampens the gap between top ranks
CANDIDATES = 50   # how many results to take from each searcher


def fuse(result_lists, k):
    scores, chunks = {}, {}
    # Walks each result list and adds 1 / (60 + rank) per chunk, 
    # ignoring the raw score (_) and using only the rank.
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
            chunks[cid] = chunk
    order = sorted(scores, key=lambda cid: -scores[cid])[:k]
    return [(chunks[cid], scores[cid]) for cid in order]


# Takes 50 from each searcher and fuses them. That's up to 100 candidates, so Recall@50 can go up
class HybridSearcher:
    def __init__(self):
        self.dense = Searcher()
        # If search.py uses a different name, change it to match
        self.bm25 = BM25Searcher(self.dense.chunks)

    def search(self, query, ticker, k=5):
        dense = self.dense.search(query, ticker, k=CANDIDATES)
        sparse = self.bm25.search(query, ticker, k=CANDIDATES)
        return fuse([dense, sparse], k)