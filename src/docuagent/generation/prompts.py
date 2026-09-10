"""Prompt construction (Layer 6, part 1): turning retrieved chunks into grounded answers.

GROUNDING is the core anti-hallucination technique in RAG: instead of asking the
LLM "what do you know about X" (answering from parameters learned during
training, which can be wrong, outdated, or just made up), we ask "given ONLY
this text, answer X" (answering from evidence handed to it in the prompt, right
now). The model still generates fluent language, but the CONTENT is supposed to
come from the retrieved chunks, not its own memory.

This does not eliminate hallucination -- a model can still ignore its
instructions, or blend a real number from the context with a fabricated one.
It reduces the SURFACE AREA for it: a well-grounded prompt gives the model
strong incentive (via instructions) and easy material (via context) to just
paraphrase what's already true in front of it, rather than invent.

Citations serve the same anti-hallucination goal from a different angle: making
the model attribute every claim to a chunk_id means a human (or an automated
eval, see Day 9) can check the claim against the source in seconds. An
unsupported answer is much easier to catch when it's supposed to have a
citation and doesn't, than when it's confident prose with no way to verify it.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a financial research assistant answering questions about \
company annual reports, using ONLY the context provided below.

Rules:
1. Answer using ONLY information in the CONTEXT. Do not use outside knowledge, \
even if you are confident about it.
2. Every factual claim must cite its source chunk, like this: [reliance_industries_2024_mda_009].
3. If the context does not contain enough information to answer, say so plainly \
-- do NOT guess or fill gaps with your own knowledge. A correct "I don't know, \
the provided context doesn't cover this" beats a fluent but ungrounded guess.
4. Numbers must be copied exactly as they appear in the context (do not round, \
convert units, or recompute unless asked to)."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into the block of evidence the model sees.

    Each chunk is tagged with its own chunk_id so the model has something
    concrete to cite (rule 2 above) and a human can trace a claim straight back
    to `data/chunks/*.json` to verify it.
    """
    blocks = []
    for chunk in chunks:
        pages = ",".join(str(p) for p in chunk.get("page_numbers", []))
        blocks.append(f"[{chunk['chunk_id']}] (page {pages})\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return f"CONTEXT:\n{context}\n\n---\n\nQUESTION: {query}"
