"""Run Layer 3: embed every chunk and index it into Qdrant.

Usage:
    python scripts/run_embedding.py                  # embed + index all registered docs
    python scripts/run_embedding.py --recreate        # wipe + rebuild the collection first
    python scripts/run_embedding.py --search "what was Jio's revenue growth?"

Requires chunking to have run first (data/chunks/<doc_id>.json must exist) and
NVIDIA_API_KEY set in .env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and crash on characters like '₹'.
sys.stdout.reconfigure(encoding="utf-8")

# make `src/` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docuagent.config import CHUNK_DIR, DOCUMENTS  # noqa: E402
from docuagent.embedding.embedder import Embedder  # noqa: E402
from docuagent.vectorstore.qdrant_store import QdrantStore  # noqa: E402


def load_chunks(doc_id: str) -> list[dict]:
    path = CHUNK_DIR / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No chunks at {path}. Run chunking first: python scripts/run_chunking.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_index(recreate: bool) -> None:
    if not DOCUMENTS:
        print("No documents registered. Add entries to DOCUMENTS in config.py.")
        return

    embedder = Embedder()
    store = QdrantStore()

    # Probe the real vector size from the live API rather than trust the config
    # constant — if NVIDIA changes the model's output size, this catches it
    # instead of silently creating a mismatched collection.
    print("Probing embedding dimension from the API...")
    dim = embedder.dim
    print(f"  -> {dim} dims (model: {embedder.model})")

    created = store.ensure_collection(dim=dim, recreate=recreate)
    print(f"Collection '{store.collection}': {'created new' if created else 'reusing existing'}")

    total_indexed = 0
    for doc in DOCUMENTS:
        chunks = load_chunks(doc.doc_id)
        if not chunks:
            print(f"  {doc.doc_id}: 0 chunks, skipping")
            continue

        print(f"  {doc.doc_id}: embedding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_passages(texts)

        n = store.upsert_chunks(chunks, vectors)
        total_indexed += n
        print(f"    -> indexed {n} chunks")

    print(f"\nTotal points in collection: {store.count()} (indexed this run: {total_indexed})")
    print("Dashboard: http://localhost:6333/dashboard")


def run_search(query: str, limit: int = 5) -> None:
    embedder = Embedder()
    store = QdrantStore()

    if store.count() == 0:
        print("Collection is empty. Run without --search first to index.")
        return

    query_vector = embedder.embed_query(query)
    hits = store.search(query_vector, limit=limit)

    print(f'Query: "{query}"\n')
    for i, hit in enumerate(hits, start=1):
        preview = hit["text"][:150].replace("\n", " ")
        print(f"{i}. score={hit['score']:.4f}  [{hit['chunk_id']}]  (p.{hit['page_numbers']})")
        print(f"   {preview}...")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop + recreate the collection first")
    ap.add_argument("--search", type=str, default=None, help="run a test query instead of indexing")
    ap.add_argument("--limit", type=int, default=5, help="number of search results")
    args = ap.parse_args()

    if args.search:
        run_search(args.search, limit=args.limit)
    else:
        run_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
