"""Answer generation (Layer 6, part 2): the LLM call, with a real timeout+fallback.

Two different failure modes matter here, and they need two different defenses:
  - The model might be WRONG (hallucination) -- prompts.py's job (grounding +
    citations), not this file's.
  - The model might be SLOW -- this file's job. NVIDIA's DeepSeek endpoint was
    measured live (Day 3/5 testing) at ~224 seconds for a 6-token reply on
    shared, queued infra. A RAG answer with no upper bound on latency is not a
    usable feature, regardless of how good the answer eventually is.

PRIMARY + FALLBACK, not primary-only: rather than picking one LLM and hoping it
stays fast, this calls NVIDIA's DeepSeek (chosen originally for quality/cost)
with a hard client-side timeout, and falls back to Gemini's fastest flash-lite
model on ANY failure -- timeout, rate limit, network error, or the account
entitlement issues already hit once with NVIDIA's embedding models (Day 3).
The trade-off: a fallback answer may be from a smaller/cheaper model. Measured
directly (see config.py) that this doesn't cost quality on a short grounded
question -- flash-lite matched two slower Gemini models word-for-word.

Alternative NOT taken: retry the same model a few times. Retries (Day 3's
embedder does this, correctly, for transient network blips) assume the failure
is temporary and unrelated to the request. A 224-second response isn't a blip
to retry past -- it's the model's typical behavior on this infra right now;
retrying would just wait 224s twice. Falling over to a different, independently
fast model is the correct response to "this whole provider is currently slow,"
not "try again."
"""

from __future__ import annotations

import time

from google import genai
from openai import OpenAI

from docuagent.config import settings


class GenerationResult:
    def __init__(self, text: str, backend: str, model: str, latency_seconds: float) -> None:
        self.text = text
        self.backend = backend  # "primary" or "fallback"
        self.model = model
        self.latency_seconds = latency_seconds

    def __repr__(self) -> str:
        return (
            f"GenerationResult(backend={self.backend!r}, model={self.model!r}, "
            f"latency={self.latency_seconds:.2f}s)"
        )


class LLMClient:
    """Chat completion with a bounded worst case: primary model, hard timeout,
    automatic fallback to a fast model on any failure."""

    def __init__(self) -> None:
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set.\n"
                "Create a .env file in the project root containing:\n"
                "    NVIDIA_API_KEY=nvapi-...\n"
            )
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (required as the generation fallback).\n"
            )
        self._primary = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        self._fallback = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        t0 = time.monotonic()
        try:
            response = self._primary.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
            return GenerationResult(
                text=text,
                backend="primary",
                model=settings.llm_model,
                latency_seconds=time.monotonic() - t0,
            )
        except Exception as primary_error:
            # Deliberately broad: a timeout, a 404 entitlement error (Day 3's
            # NVIDIA lesson), a rate limit, a network drop should all trigger
            # the same response -- fail over, don't leave the user waiting on
            # a provider that isn't answering right now.
            fallback_t0 = time.monotonic()
            response = self._fallback.models.generate_content(
                model=settings.fallback_llm_model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return GenerationResult(
                text=response.text,
                backend="fallback",
                model=settings.fallback_llm_model,
                latency_seconds=time.monotonic() - fallback_t0,
            )
