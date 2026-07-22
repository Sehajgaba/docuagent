# DocuAgent — Learning Growth Tracker

Living log of what I'm learning while building DocuAgent. Updated every build day.
Goal: **explain how an AI agent / RAG system works cold in an interview.**

- **Started:** 2026-07-19
- **Owner:** Sehaj (Backend AI Engineer)
- **Build:** 12 layers, one/day · daily git push · deep-teach cadence
- **Target:** project live + every concept explainable without notes

---

## Mastery scale

| Mark | Meaning |
|---|---|
| 🔴 | Not seen yet |
| 🟡 | Built it / saw it once |
| 🟢 | Can explain it cold (passed the day's quiz) |
| ⭐ | Deep — can argue tradeoffs + edge cases |

Rule: a concept only goes 🟢 after passing its quiz. ⭐ after a tradeoff question.

---

## Concept mastery map

| # | Concept | Layer/Day | Status | Quiz | Notes |
|---|---|---|---|---|---|
| 1 | Tokens & token counting | D1–D2 | 🟡 | — | tiktoken cl100k_base as proxy ruler |
| 2 | Embeddings (text → vector) | D1 | 🟡 | — | 768-dim Gemini |
| 3 | Cosine similarity | D1 | 🟡 | — | range −1..1; practical ~0..1; missed unrelated≈0 in re-quiz |
| 4 | PDF parsing (pymupdf vs pdfplumber) | D1 | 🟡 | — | text vs tables |
| 5 | Financial number normalization | D1 | 🟢 | ✓ | got why; missed 1,47,087→147087 |
| 6 | Registry / metadata pattern | D1 | 🟡 | — | single source of truth; missed naming it |
| 7 | Structure-aware chunking | D2 | 🟡 | — | table=1 chunk; paragraph-atomic grouping |
| 8 | Vector DB + HNSW (ANN) | D3 | 🔴 | — | O(log n) vs brute force |
| 9 | BM25 keyword search | D4 | 🔴 | — | |
| 10 | Hybrid search + RRF fusion | D4 | 🔴 | — | why rank not raw score |
| 11 | Reranking / cross-encoder | D5 | 🔴 | — | bi- vs cross-encoder |
| 12 | RAG generation + grounding | D6 | 🔴 | — | anti-hallucination |
| 13 | Prompt engineering | D6 | 🔴 | — | |
| 14 | Structured numeric querying | D7 | 🔴 | — | RAG vs SQL for numbers |
| 15 | Agents: LLM+tools+loop+memory | D8 | 🔴 | — | |
| 16 | ReAct loop | D8 | 🔴 | — | reason→act→observe |
| 17 | Tool / function calling | D8 | 🔴 | — | |
| 18 | RAGAS eval (4 metrics) | D9 | 🔴 | — | which metric → which fix |
| 19 | Observability / tracing | D10 | 🔴 | — | |
| 20 | FastAPI serving | D11 | 🔴 | — | |
| 21 | Docker + deployment | D12 | 🔴 | — | |

**Score:** 7/21 seen · 1/21 explain-cold · 0/21 deep

---

## Daily log

### Day 1 — 2026-07-19 · Layer 1: Ingestion ✅
**Built:** project scaffold, `pdf_parser.py` (PDF → structured JSON), `normalize_number`, document registry, run script. Pushed `03b313b`.
**Learned:** why two PDF libs; accounting negatives `(x)`; Indian digit grouping; registry pattern; repo hygiene (no secrets/binaries in git); Windows cp1252 `₹` crash fix.
**Bugs seen:** old `main.py` `best_score=0` (cosine can be negative); `--max` slicing after full parse.
**Quiz:** 1/6 cold → concept 5 (number-norm) 🟢. Q1 partial (missed pymupdf=text vs pdfplumber=tables split). Q3/Q4/Q5 punted → taught, re-quiz next round.
**Re-quiz round 2 (Q3/Q4/Q5):** all 3 partial, none flipped 🟢 yet — right instinct, missing precision:
  - Q3 cosine: got range(-1..1)+meaning of endpoints; said "angle between nodes" (should be *vectors*); missed practical 0..1 range for real embeddings + why that's the nuance behind the `best_score=0` bug.
  - Q4 embeddings: got text/photo→vector via a model; missed what "768-dim" means (learned meaning-features, not human-labeled); said "define" (should be "convert/learn," model isn't hand-rule-based).
  - Q5 registry: core idea solid (one place, no per-layer repeats); said "routing" (wrong term — it's metadata storage); didn't name the principle: **single source of truth**.
**Weak / revisit:** vector vs node terminology; embedding dimensionality meaning; naming "single source of truth" explicitly.

### Day 2 — 2026-07-19 · Layer 2: Chunking ✅
**Built:** `chunker.py` — tables kept whole (1 table = 1 chunk, never split); narrative split by paragraph, greedily grouped under 512-token cap, paragraph itself never split mid-sentence. Section-type heuristic (keyword match) gated on `has_tables` for balance_sheet/P&L/cash_flow. Committed `4d6e48d`.
**Learned:** why naive fixed-token chunking breaks tables; paragraph-as-atomic-unit tradeoff (oversized chunk > broken sentence); tiktoken as a cross-model token-counting proxy.
**Bug found + fixed live:** heuristic false-positived "our strong balance sheet" (prose in Chairman's letter, page 4) as the `balance_sheet` section — no actual table on that page. Fix: table-type sections now require `page['tables']` non-empty before the keyword counts. Real lesson in why naive keyword rules need a structural guard.
**Quiz:** not yet run for Day 2 concepts.
**Weak / revisit:** cosine unrelated≈0 (missed again in re-quiz), naming "single source of truth" explicitly.

<!-- TEMPLATE for next entries:
### Day N — YYYY-MM-DD · Layer N: <name>
**Built:**
**Learned:**
**Quiz:** X/Y  →  concepts moved to 🟢:
**Weak / revisit:**
-->

---

## Weak spots to patch (outside the build)
- DSA fundamentals
- System design depth
- Retrieval tradeoffs (HNSW vs IVF, chunk-size effects) — deepen after build

## Interview-readiness checklist
- [ ] Explain LLM vs RAG vs Agent in 60s
- [ ] Whiteboard the full DocuAgent pipeline from memory
- [ ] Answer "bi-encoder vs cross-encoder, when each?"
- [ ] Answer "why hybrid > pure semantic?"
- [ ] Answer "how do you stop hallucination?"
- [ ] Answer "how would you make this agentic?"
- [ ] Explain each RAGAS metric → which layer it diagnoses
- [ ] Live demo the deployed URL
