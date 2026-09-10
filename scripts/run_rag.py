"""Run Layer 6: full RAG — retrieve, ground, generate, cite.

Usage:
    python scripts/run_rag.py --ask "What was Jio's revenue growth?"
    python scripts/run_rag.py --ask "What was the CEO's salary?" --limit 5

Requires: chunking + Qdrant indexing done (scripts/run_embedding.py),
GEMINI_API_KEY and NVIDIA_API_KEY set in .env.
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
from docuagent.generation.llm import LLMClient  # noqa: E402
from docuagent.generation.rag import RAGPipeline  # noqa: E402
from docuagent.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from docuagent.vectorstore.qdrant_store import QdrantStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", type=str, required=True, help="question to answer")
    ap.add_argument("--limit", type=int, default=5, help="chunks to retrieve")
    args = ap.parse_args()

    if not DOCUMENTS:
        print("No documents registered. Add entries to DOCUMENTS in config.py.")
        return

    chunks = []
    for doc in DOCUMENTS:
        chunks.extend(load_chunks(doc.doc_id))

    embedder = Embedder()
    store = QdrantStore()
    retrieval = RetrievalPipeline(chunks, embedder, store)
    llm = LLMClient()
    rag = RAGPipeline(retrieval, llm)

    print(f'Question: "{args.ask}"\n')
    result = rag.ask(args.ask, limit=args.limit)

    print(f"[{result.generation.backend} | {result.generation.model} | "
          f"{result.generation.latency_seconds:.2f}s]\n")
    print(result.answer)
    print("\nSources:")
    for s in result.sources:
        print(f"  [{s['chunk_id']}]  page {s['page_numbers']}  ({s['section_type']})")


if __name__ == "__main__":
    main()
