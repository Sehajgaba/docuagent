"""Keyword search (Layer 4, part 1) -- the other half of hybrid retrieval.

Vector search (Day 3) finds MEANING: "sales rose by a tenth" matches "revenue
grew 12%" even with zero shared words. That is also its blind spot -- an exact
token like an acronym, a product code, or "EBITDA" can get diluted by everything
else the embedding also encodes (tone, topic, sentence shape). Keyword search
finds exact terms, nothing else, which is precisely what fills that gap.

BM25 (Best Matching 25) is the standard keyword-ranking algorithm -- what
Elasticsearch/Lucene use by default. The intuition, no formula required:

  1. TERM FREQUENCY: a chunk that says "EBITDA" three times is probably more
     about EBITDA than one that says it once. But BM25 SATURATES this -- the
     10th occurrence barely counts more than the 5th. Plain word-count scoring
     would let a chunk that just repeats "EBITDA" fifty times dominate.
  2. INVERSE DOCUMENT FREQUENCY: a term that appears in every chunk ("the",
     "company", "crore") is useless for telling chunks apart, so it is
     downweighted automatically. A rare term ("nemoretriever") that appears in
     one chunk is a strong signal, so it is upweighted automatically. This is
     WHY we don't need a stopword list here -- IDF already suppresses common
     words as a side effect of the math, no hand-maintained list required.
  3. LENGTH NORMALIZATION: a 50-token chunk matching once is a stronger signal
     than a 2000-token chunk matching once (the short chunk is clearly ABOUT
     the term; the long one might mention it once in passing).

No embedding model, no network call, no API key: BM25 works over the raw chunk
text you already have, purely index lookup + arithmetic. That is also why it
doesn't need Qdrant's ANN trick (Day 3) -- an inverted index (term -> which
documents contain it) already gives exact O(1)-ish lookup per term; there is no
"approximate nearest neighbour in a 768-dim vector space" problem to solve here.

Why `rank-bm25` over alternatives: it's a tiny, dependency-free, pure-Python
reference implementation -- good for learning exactly what BM25 does, and fine
at this project's current scale (tens of chunks). A real production system
would use Elasticsearch/OpenSearch or Qdrant's own sparse-vector support
(letting one server do both dense and sparse search) -- trade-off there is
operational complexity (another service, or another index type to manage) for
speed at millions of documents.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Deliberately simple: lowercase + split on runs of non-alphanumeric characters.
# No stemming ("growing" vs "grow" stay different tokens), no stopword removal
# (BM25's IDF term handles that, see module docstring). A fancier tokenizer
# would catch more matches (plurals, tenses) at the cost of being another
# unexplainable black box -- not worth it at this project's scale.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory keyword index over a fixed list of chunks.

    Unlike Qdrant, this has no persistence layer and no server -- it is cheap
    enough (tens of thousands of chunks, still just Python lists in RAM) to
    rebuild from data/chunks/*.json every time the process starts. That is a
    deliberate simplicity trade-off, not an oversight: adding a server here
    would be solving a scale problem this project doesn't have yet.
    """

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Return the top `limit` chunks by BM25 score (highest first)."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [
            {"score": float(scores[i]), **self.chunks[i]}
            for i in ranked[:limit]
            if scores[i] > 0  # a zero score means no query term matched at all
        ]
