"""RAG orchestration (Layer 6, part 3): retrieval + generation, wired together.

RAG = Retrieval-Augmented Generation. The "augmented" part is literal: instead
of the LLM answering from what it memorised during training, we fetch relevant
text at question time (Days 3-5's whole retrieval pipeline) and hand it to the
LLM as part of the prompt (prompts.py's grounding). This module is just the
wiring between those two halves -- retrieve, then generate, then attach sources.

Why keep retrieval and generation as separate stages instead of one call? Two
independent things can go wrong (bad retrieval vs. bad generation on good
context), and keeping them separate means each can be tested and evaluated on
its own -- Day 9's RAGAS metrics score exactly this way: "context precision"
and "context recall" grade the retrieval step, "faithfulness" and "answer
relevancy" grade the generation step, independently.
"""

from __future__ import annotations

from docuagent.generation.llm import GenerationResult, LLMClient
from docuagent.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from docuagent.retrieval.pipeline import RetrievalPipeline


class RAGAnswer:
    def __init__(self, answer: str, sources: list[dict], generation: GenerationResult) -> None:
        self.answer = answer
        self.sources = sources
        self.generation = generation

    def __repr__(self) -> str:
        return f"RAGAnswer(backend={self.generation.backend}, sources={len(self.sources)})"


class RAGPipeline:
    def __init__(self, retrieval: RetrievalPipeline, llm: LLMClient) -> None:
        self.retrieval = retrieval
        self.llm = llm

    def ask(
        self,
        query: str,
        limit: int = 5,
        company: str | None = None,
        fy: str | None = None,
        section_type: str | None = None,
    ) -> RAGAnswer:
        chunks = self.retrieval.search(
            query, limit=limit, company=company, fy=fy, section_type=section_type
        )
        user_prompt = build_user_prompt(query, chunks)
        generation = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        sources = [
            {
                "chunk_id": c["chunk_id"],
                "page_numbers": c.get("page_numbers", []),
                "section_type": c.get("section_type"),
            }
            for c in chunks
        ]
        return RAGAnswer(answer=generation.text, sources=sources, generation=generation)
