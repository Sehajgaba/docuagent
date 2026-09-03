"""Run Layers 4+5: hybrid search (vector + BM25 fused via RRF), then reranked.

Usage:
    python scripts/run_hybrid_search.py --search "What was the EBITDA margin?"
    python scripts/run_hybrid_search.py --search "how did Jio's subscribers change?" --limit 5

Prints vector-only, BM25-only, hybrid, and hybrid+reranked results side by side
for the same query -- the point is to SEE each stage's effect, not just claim it.

Requires: chunking done, Qdrant already indexed (scripts/run_embedding.py has
been run), GEMINI_API_KEY set in .env. First run downloads the ~80MB
cross-encoder model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docuagent.chunking.chunker import load_chunks  # noqa: E402
from docuagent.config import DOCUMENTS  # noqa: E402
from docuagent.embedding.embedder import Embedder  # noqa: E402
from docuagent.retrieval.bm25_index import BM25Index  # noqa: E402
from docuagent.retrieval.hybrid import HybridSearcher  # noqa: E402
from docuagent.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from docuagent.retrieval.reranker import Reranker  # noqa: E402
from docuagent.vectorstore.qdrant_store import QdrantStore  # noqa: E402


def load_all_chunks() -> list[dict]:
    chunks: list[dict] = []
    for doc in DOCUMENTS:
        chunks.extend(load_chunks(doc.doc_id))
    return chunks


def preview(chunk: dict) -> str:
    return chunk["text"][:100].replace("\n", " ")


def print_block(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"--- {title} ---")
    if not rows:
        print("  (no results)")
    for label, chunk in rows:
        print(f"  {label}  [{chunk['chunk_id']}]")
        print(f"      {preview(chunk)}...")
    print()


def run_search(query: str, limit: int) -> None:
    chunks = load_all_chunks()
    embedder = Embedder()
    store = QdrantStore()
    bm25 = BM25Index(chunks)
    hybrid = HybridSearcher(chunks, embedder, store)
    reranker = Reranker()
    pipeline = RetrievalPipeline(chunks, embedder, store, reranker=reranker)

    print(f'Query: "{query}"\n')

    query_vector = embedder.embed_query(query)
    vector_hits = store.search(query_vector, limit=limit)
    print_block(
        "VECTOR ONLY (semantic)",
        [(f"score={h['score']:.4f}", h) for h in vector_hits],
    )

    bm25_hits = bm25.search(query, limit=limit)
    print_block(
        "BM25 ONLY (keyword)",
        [(f"score={h['score']:.4f}", h) for h in bm25_hits],
    )

    hybrid_hits = hybrid.search(query, limit=limit)
    print_block(
        "HYBRID (RRF fused)",
        [
            (
                f"fused={h['fused_score']:.4f}  "
                f"(vector:{'Y' if h['in_vector_top'] else 'n'} "
                f"bm25:{'Y' if h['in_bm25_top'] else 'n'})",
                h,
            )
            for h in hybrid_hits
        ],
    )

    reranked_hits = pipeline.search(query, limit=limit)
    print_block(
        "HYBRID + RERANKED (cross-encoder)",
        [(f"rerank={h['rerank_score']:.4f}", h) for h in reranked_hits],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=str, required=True, help="query to test")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    if not DOCUMENTS:
        print("No documents registered. Add entries to DOCUMENTS in config.py.")
        return

    run_search(args.search, args.limit)


if __name__ == "__main__":
    main()
