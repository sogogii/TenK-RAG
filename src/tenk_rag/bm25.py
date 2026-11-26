"""Keyword search over chunks with BM25."""

import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    # Lowercase words and numbers; punctuation and table pipes are dropped
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Searcher:
    def __init__(self, chunks):
        # One index per company, since every search is limited to one ticker
        self.by_ticker = {}   # groups chunks by company
        for chunk in chunks:
            self.by_ticker.setdefault(chunk["ticker"], []).append(chunk)
        self.index = {
            # Builds a BM25 index per company, including IDF
            ticker: BM25Okapi([tokenize(c["text"]) for c in group])
            for ticker, group in self.by_ticker.items()
        }

    # Same shape as Searcher.search, so it plugs straight into the evaluation
    def search(self, query, ticker, k=5):
        group = self.by_ticker[ticker]
        scores = self.index[ticker].get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(group[i], float(scores[i])) for i in top]