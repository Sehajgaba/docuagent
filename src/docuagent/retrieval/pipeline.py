"""The full retrieval pipeline (Layers 3+4+5 composed): recall wide, judge narrow.

    query
      |
      v
    HybridSearcher (Day 3 vector + Day 4 BM25, fused by RRF)   -- recall: cheap,
      |  wide net, keeps `fusion_pool` candidates                approximate
      v
    Reranker (Day 5 cross-encoder)                              -- precision:
      |  re-judges each candidate against the query directly      expensive,
      v  keeps top `limit`                                        accurate
    final results

Each stage narrows the field and gets more expensive per item -- that's the
point. HNSW (Day 3) and BM25 (Day 4) can each scan every chunk in the corpus
because they're cheap per item. The cross-encoder (Day 5) could never do that
at scale (one transformer forward pass per chunk, per query), so it only ever
sees the shortlist the first two stages already narrowed down.
"""

from __future__ import annotations

from docuagent.embedding.embedder import Embedder
from docuagent.retrieval.hybrid import HybridSearcher
from docuagent.retrieval.reranker import Reranker
from docuagent.vectorstore.qdrant_store import QdrantStore


class RetrievalPipeline:
    def __init__(
        self,
        chunks: list[dict],
        embedder: Embedder,
        store: QdrantStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.hybrid = HybridSearcher(chunks, embedder, store)
        self.reranker = reranker or Reranker()

    def search(
        self,
        query: str,
        limit: int = 5,
        fusion_pool: int = 20,
        rerank_pool: int = 20,
        company: str | None = None,
        fy: str | None = None,
        section_type: str | None = None,
    ) -> list[dict]:
        """Retrieve `limit` best chunks for `query`.

        `fusion_pool`: how many results EACH of vector/BM25 contributes before
        RRF fusion (Day 4's candidate_pool).
        `rerank_pool`: how many fused hybrid results get passed to the cross-
        encoder. Must be >= limit; wider than `limit` so the reranker can pull
        up a good result that hybrid ranked just outside the final cut, not
        just re-order the same top 5 it was already going to return.
        """
        shortlist = self.hybrid.search(
            query,
            limit=rerank_pool,
            candidate_pool=fusion_pool,
            company=company,
            fy=fy,
            section_type=section_type,
        )
        return self.reranker.rerank(query, shortlist, top_k=limit)
