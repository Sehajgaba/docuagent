"""Text -> vector (Layer 3, part 1).

An EMBEDDING is a list of floats that encodes *meaning*. Two texts that mean
similar things land close together in that space, even with zero shared words:

    "revenue grew 12%"      -> [0.021, -0.144, ...]
    "sales rose by a tenth" -> [0.019, -0.139, ...]   <- close by
    "the cat sat on a mat"  -> [0.310,  0.402, ...]   <- far away

That's the whole reason RAG works: keyword search misses paraphrases, vector
search doesn't.

This module is a BI-ENCODER: it encodes each text independently into one vector,
with no knowledge of what it will later be compared against. That independence is
what makes it fast enough to index thousands of chunks (encode once, reuse
forever). Day 5's cross-encoder reranker is the opposite trade: it reads the
query and the chunk TOGETHER, which is far more accurate but must re-run for
every (query, chunk) pair, so it can only be afforded on a short candidate list.

Provider: Gemini (`text-embedding-004`) via the `google-genai` SDK.
  - Why Gemini here: NVIDIA's build.nvidia.com key is entitlement-gated per
    model (a key can auth successfully yet still 404 "not found for account"
    on a specific model you haven't individually enabled) -- confirmed dead
    end across 4 embedding models. Gemini's AI Studio key works the moment you
    create it, no per-model approval step.
  - Alternative considered: local `sentence-transformers` (zero key, but a
    ~130MB first-run download and weaker on long financial prose). Trade-off
    accepted here: a network dependency + daily quota (free tier: 1500
    req/day) in exchange for quality and zero setup friction today.
"""

from __future__ import annotations

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from docuagent.config import settings

# --- Asymmetric embedding -----------------------------------------------------
# Retrieval embedding models are trained ASYMMETRICALLY: a short question and the
# long passage that answers it are worded nothing alike, so the model is given a
# flag saying which side it is encoding and shifts the vector accordingly.
#
#   TASK_PASSAGE -> use when indexing document chunks
#   TASK_QUERY   -> use when embedding a user's question at search time
#
# Getting this backwards is a silent quality bug: no error, just worse hits. It
# is the single most common mistake with this family of models. Gemini calls
# this `task_type`; NVIDIA's NIM models call the same idea `input_type` --
# different name, identical concept.
TASK_PASSAGE = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class Embedder:
    """Wraps the Gemini embeddings endpoint with batching and retries."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set.\n"
                "Create a .env file in the project root containing:\n"
                "    GEMINI_API_KEY=...\n"
                "Get one free at https://aistudio.google.com/app/apikey"
            )
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self._client = genai.Client(api_key=settings.gemini_api_key)

    # A network call can fail transiently (rate limit, blip). Retrying with an
    # EXPONENTIAL BACKOFF (wait 2s, then 4s, then 8s) is the standard fix: an
    # immediate retry usually hits the same rate limit, backing off gives the
    # quota window time to reset. Alternative: a fixed-delay retry loop (simpler,
    # but hammers a rate-limited API). Trade-off: a hard failure takes ~14s to
    # surface instead of failing instantly.
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        # NOTE: the AI Studio API (unlike Vertex AI) does NOT auto-truncate
        # over-length input -- it raises instead. gemini-embedding-2's 8192-
        # token limit covers even the chunker's oversized narrative chunks (up
        # to ~2952 tokens, from Day 2's blank-line paragraph-splitting bug),
        # but if a future chunk ever exceeds it, that should fail loudly here
        # rather than being silently mangled.
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
        # Gemini preserves input order (no `index` field like NVIDIA's API),
        # so we return embeddings in the order the SDK gives them back.
        return [e.values for e in response.embeddings]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for INDEXING."""
        return self._embed_all(texts, TASK_PASSAGE)

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question for SEARCHING."""
        return self._embed_all([text], TASK_QUERY)[0]

    def _embed_all(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Batch through the whole list.

        Why batch at all? One HTTP round-trip per chunk wastes most of the time
        on network latency. Sending a handful at once amortises that. Why not
        hundreds at once? Providers cap request size, and one failed giant batch
        loses far more work than one failed small batch.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, task_type))
        return vectors

    @property
    def dim(self) -> int:
        """Vector length, measured from the live API (never hardcoded/guessed)."""
        return len(self.embed_query("dimension probe"))
