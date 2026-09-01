"""Hybrid search (Layer 4, part 2): fuse vector search + BM25 into one ranking.

Vector search (Day 3) and BM25 (this layer) each catch what the other misses:
  - "how did Jio's subscriber base change?"  -> vector wins (paraphrase, no
    exact word overlap with the chunk that actually answers it)
  - "what was the EBITDA margin?"            -> BM25 wins (exact financial
    term; the embedding may rank a chunk that TALKS ABOUT margins generally
    above the one with the actual EBITDA number)
Running only one of the two means silently losing whichever category it's
blind to. Hybrid search runs both and merges the results.

THE HARD PART: combining two DIFFERENT scales. Cosine similarity is bounded,
roughly 0..1 in practice. BM25 is unbounded -- 4.7 on a short corpus, 23.8 on a
big one, no fixed ceiling. Averaging or summing those two numbers directly is
meaningless: whichever system happens to produce bigger numbers today would
dominate the fused ranking regardless of which result is actually better.

RECIPROCAL RANK FUSION (RRF) sidesteps the whole problem: instead of combining
SCORES, it combines RANKS (1st place, 2nd place, ...), which are already on the
same scale no matter what produced them. For each ranked list, chunk at rank r
contributes 1 / (k + r) to its fused score; a chunk's fused score is the SUM of
that contribution across every list it appears in. A chunk ranked #1 by both
systems beats one ranked #1 by only one of them -- agreement across systems is
rewarded, which is exactly the signal you want.

`k` (=60, the constant from the original RRF paper, TREC 2009) softens the
curve: without it, rank #1 vs rank #2 would swing the score by 2x (1/1 vs 1/2);
with k=60, it's 1/61 vs 1/62, a ~1.6% difference. The intuition: BM25 and
vector search both carry some ranking noise near the top, so #1 and #2 usually
aren't "twice as good," and RRF shouldn't pretend they are.
"""

from __future__ import annotations

from docuagent.embedding.embedder import Embedder
from docuagent.retrieval.bm25_index import BM25Index
from docuagent.vectorstore.qdrant_store import QdrantStore

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse multiple ranked lists of chunk_ids into one fused score per chunk_id.

    Each `ranked_lists[i]` is a list of chunk_ids, best match first. Returns a
    dict of chunk_id -> fused score (higher = better), NOT yet sorted -- the
    caller decides how to use it (sort, take top N, etc).
    """
    fused: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridSearcher:
    """Runs vector search + BM25 in parallel and fuses the results with RRF."""

    def __init__(self, chunks: list[dict], embedder: Embedder, store: QdrantStore) -> None:
        self.bm25 = BM25Index(chunks)
        self.embedder = embedder
        self.store = store
        # chunk_id -> full chunk dict, so fused results can be re-hydrated with
        # text/metadata regardless of which ranker(s) surfaced them.
        self._by_id = {c["chunk_id"]: c for c in chunks}

    def search(
        self,
        query: str,
        limit: int = 5,
        candidate_pool: int = 20,
        company: str | None = None,
        fy: str | None = None,
        section_type: str | None = None,
    ) -> list[dict]:
        """Hybrid search: fuse vector + BM25 rankings, return top `limit`.

        `candidate_pool` is how many results EACH ranker contributes before
        fusion (wider than the final `limit`) -- a chunk ranked #15 by vector
        search but #1 by BM25 should still be able to win after fusion; if we
        only pulled the top 5 from each side it could never surface at all.
        """
        query_vector = self.embedder.embed_query(query)
        vector_hits = self.store.search(
            query_vector,
            limit=candidate_pool,
            company=company,
            fy=fy,
            section_type=section_type,
        )
        bm25_hits = self.bm25.search(query, limit=candidate_pool)

        vector_ids = [h["chunk_id"] for h in vector_hits]
        bm25_ids = [h["chunk_id"] for h in bm25_hits]
        fused_scores = reciprocal_rank_fusion([vector_ids, bm25_ids])

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        results = []
        for chunk_id in ranked_ids[:limit]:
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue  # defensive: a ranker returned an id outside this corpus
            results.append(
                {
                    "fused_score": fused_scores[chunk_id],
                    "in_vector_top": chunk_id in vector_ids,
                    "in_bm25_top": chunk_id in bm25_ids,
                    **chunk,
                }
            )
        return results
