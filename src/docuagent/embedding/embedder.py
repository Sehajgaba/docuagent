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

Provider: NVIDIA NIM via the OpenAI-compatible SDK.
  - Why the `openai` package for an NVIDIA model? NVIDIA exposes an
    OpenAI-shaped API, so one well-known SDK covers both embeddings and (later)
    the DeepSeek chat model. Alternative: raw `requests` calls (no dependency,
    but we'd hand-roll retries/streaming/errors), or a provider-specific client
    (locks us in). Trade-off: we depend on a client named for a different vendor.
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from docuagent.config import settings

# --- Asymmetric embedding -----------------------------------------------------
# `embedqa` retrieval models are trained ASYMMETRICALLY: a short question and the
# long passage that answers it are worded nothing alike, so the model is given a
# flag saying which side it is encoding and shifts the vector accordingly.
#
#   INPUT_PASSAGE -> use when indexing document chunks
#   INPUT_QUERY   -> use when embedding a user's question at search time
#
# Getting this backwards is a silent quality bug: no error, just worse hits. It
# is the single most common mistake with this family of models.
INPUT_PASSAGE = "passage"
INPUT_QUERY = "query"


class Embedder:
    """Wraps the NVIDIA embeddings endpoint with batching and retries."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set.\n"
                "Create a .env file in the project root containing:\n"
                "    NVIDIA_API_KEY=nvapi-...\n"
                "Get one free at https://build.nvidia.com"
            )
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self._client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )

    # A network call can fail transiently (rate limit, blip). Retrying with an
    # EXPONENTIAL BACKOFF (wait 2s, then 4s, then 8s) is the standard fix: an
    # immediate retry usually hits the same rate limit, backing off gives the
    # quota window time to reset. Alternative: a fixed-delay retry loop (simpler,
    # but hammers a rate-limited API). Trade-off: a hard failure takes ~14s to
    # surface instead of failing instantly.
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
            # `input_type` and `truncate` are NVIDIA extensions, not part of the
            # OpenAI schema, so the SDK passes them through via extra_body.
            # truncate=END: if a chunk exceeds the model's context, drop the tail
            # rather than erroring out the whole batch.
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        # The API may return results out of order; `index` is authoritative.
        ordered = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for INDEXING."""
        return self._embed_all(texts, INPUT_PASSAGE)

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question for SEARCHING."""
        return self._embed_all([text], INPUT_QUERY)[0]

    def _embed_all(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Batch through the whole list.

        Why batch at all? One HTTP round-trip per chunk wastes most of the time
        on network latency. Sending 16 at once amortises that. Why not 500 at
        once? Providers cap request size, and one failed giant batch loses far
        more work than one failed small batch.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, input_type))
        return vectors

    @property
    def dim(self) -> int:
        """Vector length, measured from the live API (never hardcoded/guessed)."""
        return len(self.embed_query("dimension probe"))
