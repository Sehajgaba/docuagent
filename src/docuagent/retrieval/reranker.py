"""Reranking (Layer 5): a second, more expensive pass over a short shortlist.

Day 3's embedder and Day 4's BM25 are both BI-ENCODERS in spirit: they score a
query against a chunk WITHOUT ever looking at the two together. The embedder
encodes the chunk once, in isolation, at indexing time, with no idea what
question will eventually be asked of it. That's what makes it fast enough to
search thousands of chunks -- but it also caps how much it can understand: it
can't reason about "does THIS query actually match THIS passage" because it
never sees them side by side.

A CROSS-ENCODER can. It takes the (query, passage) pair TOGETHER as one input,
runs one transformer forward pass over both at once, and outputs a single
relevance score. Every word of the query can attend to every word of the
passage directly, so it catches things a cosine-similarity-of-two-vectors
comparison misses -- word order, negation, "which of these five mentions of
'margin' is the one this question is actually about."

The catch is cost: a cross-encoder can't be precomputed. There's no such thing
as "the passage's cross-encoder vector" sitting ready in Qdrant, because the
score only exists once you pair it with a specific query -- so it must run
fresh, per query, for every candidate. Run it over all 25 (or 25,000) chunks
and you've lost all of Day 3's speed advantage. Run it only over the top ~20
candidates that hybrid search (Day 4) already shortlisted, and it's cheap AND
accurate: cheap because 20 forward passes is nothing, accurate because it's a
strictly better judge of relevance than rank fusion over two independent
scores.

This is the standard two-stage retrieval architecture:
    stage 1 (recall):    bi-encoder + BM25, cast a wide net over everything
    stage 2 (precision): cross-encoder, re-judge only the top ~20, keep top ~5
Day 4 already demonstrated the gap this closes: a table-of-contents chunk
(keyword-dense, matched BM25 well) rode into the hybrid top-5 on rank-fusion
alone. A cross-encoder reading "how has the company's phone and internet
business grown" against that chunk's actual text should recognise it isn't a
real answer and push it back down.

Why `cross-encoder/ms-marco-MiniLM-L-6-v2` specifically: trained on the MS
MARCO passage-ranking dataset (real search queries + labelled relevant
passages) -- exactly the query-answers-passage task this is. "L-6" = 6
transformer layers, the mid-size point in this model family:
  - L-2/L-4: faster, noticeably lower quality.
  - L-12:    better quality, roughly 2x slower.
  L-6 is the standard default trade-off, small enough for CPU inference
  (no GPU, no API key -- runs entirely local, first call downloads ~80MB once).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Wraps a local cross-encoder to re-score and re-order a candidate list."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Re-score `candidates` (each must have a "text" field) against `query`.

        Returns candidates sorted by cross-encoder score, highest first, with a
        `rerank_score` field added. This score is NOT comparable to cosine
        similarity or BM25 score -- it's the model's own relevance logit, only
        meaningful for ordering within this one call.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        scored = [
            {"rerank_score": float(score), **candidate}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_k] if top_k is not None else scored
