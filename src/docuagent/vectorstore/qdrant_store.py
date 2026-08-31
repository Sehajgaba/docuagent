"""Vectors -> searchable index (Layer 3, part 2).

A VECTOR DATABASE stores embeddings and answers "which stored vectors are
closest to this one?" fast.

Why not just a Python list + a for-loop? That is BRUTE FORCE: compare the query
against every stored vector, O(n). At 16 chunks it is instant. At 500k chunks
(where this project is headed) it is ~500k x 2048 float multiplications per
query -- hundreds of milliseconds, per user, per question.

Qdrant instead builds an HNSW index (Hierarchical Navigable Small World). The
mental model: a road network with layers. The top layer has a few "motorway"
nodes with long-range links; lower layers get denser until the bottom layer holds
every point. A search enters at the top, greedily hops toward the query, then
drops a layer and refines. You skip most of the dataset instead of scanning it,
giving roughly O(log n) instead of O(n).

The catch, and the interview answer: HNSW is APPROXIMATE (an "ANN" index). It can
miss a true nearest neighbour. In exchange you get ~100x speed for ~99% recall.
For RAG that trade is nearly free, because Day 5's reranker re-scores the
shortlist anyway -- a slightly imperfect candidate set gets cleaned up later.

Why Qdrant over the alternatives:
  - FAISS    -- faster raw index, but a library not a server: no metadata
                filtering, no persistence, no HTTP API. We would build those.
  - Chroma   -- easiest to start, weaker filtering and production story.
  - pgvector -- great if you already run Postgres; slower at high dimensions.
  - Pinecone -- managed and good, but paid with no real free tier.
Qdrant gives filtered search (critical here: "only Reliance FY2024 balance-sheet
chunks"), runs free in Docker locally and free in their cloud for deploy.
Trade-off: one more service to run.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from docuagent.config import settings

# Namespace for turning our string chunk_ids into stable UUIDs (see _point_id).
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Metadata fields we filter on. Qdrant can filter without an index, but it then
# scans payloads; an index makes it a lookup. Cheap to add, so add them up front.
_INDEXED_FIELDS = ("company", "fy", "section_type")


def _point_id(chunk_id: str) -> str:
    """Map our string chunk_id to a UUID.

    Qdrant point ids must be an unsigned int or a UUID -- arbitrary strings like
    "reliance_industries_2024_balance_sheet_001" are rejected. We hash the
    chunk_id with uuid5, which is DETERMINISTIC: the same chunk_id always yields
    the same UUID. That makes re-indexing IDEMPOTENT -- rerunning overwrites the
    existing point instead of inserting a duplicate. (uuid4 would be random and
    would silently double the collection on every run.)
    """
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantStore:
    """Thin wrapper over the Qdrant collection holding our chunks."""

    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or settings.collection_name
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )

    # --- schema ---------------------------------------------------------------
    def ensure_collection(self, dim: int, recreate: bool = False) -> bool:
        """Create the collection if absent. Returns True if it was created.

        COSINE distance measures the ANGLE between two vectors, ignoring their
        length. That is what we want for text: a 3000-token chunk produces a
        longer vector than a 10-token one, but length reflects verbosity, not
        topic. Euclidean distance would penalise the long chunk for being long.
        (Dot product equals cosine only when vectors are already normalised.)

        `dim` is fixed at creation. Changing embedding model means a new
        dimension, which means the collection must be recreated and every chunk
        re-embedded.
        """
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return False
        if exists:
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field in _INDEXED_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        return True

    # --- write ----------------------------------------------------------------
    def upsert_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        """Store chunks + their vectors. UPSERT = insert new, overwrite existing.

        The chunk text goes into the PAYLOAD (arbitrary JSON stored beside the
        vector), not just the vector. Embeddings are one-way: you cannot decode
        [0.02, -0.14, ...] back into "revenue grew 12%". So the readable text has
        to be kept alongside, or a search would return coordinates we cannot show
        the user or hand to an LLM.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        points = [
            qm.PointStruct(
                id=_point_id(chunk["chunk_id"]),
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "company": chunk["company"],
                    "fy": chunk["fy"],
                    "section_type": chunk["section_type"],
                    "page_numbers": chunk["page_numbers"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    # --- read -----------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        company: str | None = None,
        fy: str | None = None,
        section_type: str | None = None,
    ) -> list[dict]:
        """Nearest-neighbour search, optionally narrowed by metadata.

        Filtering happens INSIDE the index walk, not as a post-filter. Post-
        filtering would fetch the global top-5 and then throw away the ones from
        the wrong company, often leaving zero results. Qdrant instead searches
        only the matching subset, so you always get `limit` real hits.
        """
        conditions = [
            qm.FieldCondition(key=key, match=qm.MatchValue(value=value))
            for key, value in (
                ("company", company),
                ("fy", fy),
                ("section_type", section_type),
            )
            if value is not None
        ]

        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            query_filter=qm.Filter(must=conditions) if conditions else None,
            with_payload=True,
        ).points

        return [{"score": hit.score, **(hit.payload or {})} for hit in hits]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def info(self) -> dict:
        collection = self.client.get_collection(self.collection)
        return {
            "points": collection.points_count,
            "status": str(collection.status),
            "dim": collection.config.params.vectors.size,
            "distance": str(collection.config.params.vectors.distance),
        }
