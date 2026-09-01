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
| 7 | Structure-aware chunking | D2–D3 | 🟡 | — | table=1 chunk; paragraph-atomic grouping; D3: switched to layout-block splitting + digit-density noise filter |
| 8 | Vector DB + HNSW (ANN) | D3 | 🟡 | — | O(log n) vs brute force; cosine vs euclidean; Qdrant built + tested live |
| 9 | BM25 keyword search | D4 | 🟡 | — | TF (saturating) + IDF (auto-downweights common words) + length norm; no stopword list needed |
| 10 | Hybrid search + RRF fusion | D4 | 🟡 | — | why rank not raw score (BM25 unbounded vs cosine 0..1); k=60 softens rank-1-vs-2 swing |
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

**Score:** 10/21 seen · 1/21 explain-cold · 0/21 deep

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

### Day 3 — 2026-08-31 · Layer 3: Embeddings + Qdrant ✅
**Built:** `embedding/embedder.py` (Gemini `gemini-embedding-2`, 768-dim via Matryoshka truncation from native 3072, asymmetric passage/query task_type, batching, retry+backoff), `vectorstore/qdrant_store.py` (collection mgmt, deterministic uuid5 point ids for idempotent upsert, payload-filtered cosine search), `run_embedding.py` (`--recreate`, `--search`). Qdrant running in Docker with a bind-mounted volume. Not committed yet.
**Provider swaps forced mid-build (both real findings):**
  - NVIDIA NIM was the original plan (one key for embed+chat) — dead end: account has zero embedding-model entitlement. A valid key auths (confirmed via 403 on a bad key vs 404 on a real one) but every embed model tried (`nv-embedqa-1b`, `mistral-7b-v2`, `embed-qa-4`, `arctic-embed-l`) 404s "not found for account" — NVIDIA gates model access per-model on build.nvidia.com, `/v1/models` lists models regardless of entitlement. NVIDIA kept for chat only (DeepSeek — works, but ~224s for a 6-token reply; flagged as a Day 6 problem, not solved yet).
  - Switched to Gemini for embeddings. `text-embedding-004` (the CLAUDE.md-planned model) is retired. Live-queried `ListModels` instead of guessing → current successors are `gemini-embedding-001` (2048-token cap) and `gemini-embedding-2` (8192-token cap, multimodal). Picked `gemini-embedding-2` — its longer cap happens to cover chunks even before the chunker fix below.
  - `auto_truncate` exists on Vertex AI's embed config but not AI Studio's — real API surface differs from docs/training data; found via a live 400... actually a `ValueError` raised client-side by the SDK, not a guess.
**Chunker bugs, found via real retrieval (not theory) and fixed same day:**
  - Root cause: `_split_paragraphs` split on blank lines (`\n\s*\n`), but pymupdf's `get_text("text")` rarely emits them — most pages collapsed to 1 "paragraph" → whole-page chunks up to 2952 tokens (5.7x the 512 cap). Fixed by extracting via `get_text("blocks")` instead — pymupdf's own layout-geometry paragraph boundaries, not punctuation-guessed ones. Max chunk size dropped to 545 tokens.
  - Side effect of the same root cause: pymupdf's flowing text also contains every table's numbers a second time (already captured properly, with labels, via pdfplumber). Rather than fragile substring-matching between the two extractors, filtered by block *shape*: table-row blocks are >50% digits by character, real prose isn't (verified against page 5: rule dropped 63/111 blocks, all genuine table remnants, kept all 48 real sentences, one borderline table row slipped through at 45% density — acceptable miss).
  - Two junk table chunks (`'Vari'`, `'Connectivity'` — pdfplumber misreading cover-page graphics as grids) — fixed with a `<3 token` floor on rendered tables, same threshold reused for the digit-density filter's minimum.
  - Added `chunk_type` field (`"table"`/`"narrative"`) so filtering doesn't depend on parsing the `chunk_id` string.
  - What I flagged as "mojibake" (`RIL�s`) mid-review turned out to be a false alarm — real U+2019 curly apostrophe in the JSON (verified: `ord(ch) == 0x2019`), just unrenderable in my Windows terminal. Not a data bug, no fix needed.
**Retrieval verified live, before vs after fix**, same 2 queries against real Qdrant search:
  - "Jio revenue growth" — before: the 2952-token whole-page blob ranked #3 (0.71) on an unrelated query (topic blur from cramming 5 sections into one vector). After: top 3 results are all genuinely Jio content from the right page.
  - "balance sheet total assets" — before: results included a stray 4-number table fragment with no header and the table-of-contents page. After: both gone; remaining noise (2 unrelated results in top-5) is a recall-not-precision problem — the right fix is Day 4 (BM25 exact-term match) + Day 5 (cross-encoder rerank), not more chunking work.
**Chunks:** 16 → 25 after fix (8-page dev slice), avg 336 tokens (was 696), max 545 (was 2952), 20 narrative / 5 table.
**Quiz:** skipped by request — concepts 7/8 stay 🟡, owed before Day 4 quiz backlog grows further.
**Weak / revisit:** Day 2 quiz still outstanding on top of Day 3's.

### Day 4 — 2026-09-01 · Layer 4: Hybrid search (BM25 + RRF) ✅
**Built:** `retrieval/bm25_index.py` (`rank-bm25` `BM25Okapi`, simple lowercase-alnum tokenizer, no stopword list — IDF handles that automatically), `retrieval/hybrid.py` (`reciprocal_rank_fusion`, `HybridSearcher` — runs vector + BM25 in parallel, fuses by rank not raw score), `run_hybrid_search.py` (prints vector-only/BM25-only/hybrid side by side for one query). Small refactor: `load_chunks` moved out of `run_embedding.py` into `chunker.py` (was duplicated, now shared by both scripts). Not committed yet.
**Learned:** BM25 = term frequency (saturating, not linear — 10th mention of a word barely counts more than the 5th) × inverse document frequency (rare terms weighted up, common terms down, automatic — no hand-built stopword list) × length normalization (a short chunk matching once is a stronger signal than a long one matching once). Why RRF not raw-score averaging: BM25 is unbounded (can be 3, can be 30), cosine is ~0..1 — summing them directly lets whichever system's numbers are bigger silently dominate. RRF fuses by RANK POSITION instead (same scale regardless of source), `1/(k+rank)`, `k=60` softens the #1-vs-#2 swing since neither ranker's top slot is reliably "twice as good" as its second.
**Real finding, not textbook:** ran "EBITDA margin" — BM25 ranked a chunk with 2 "margin" + 2 "ebitda" mentions ABOVE one with 1 "margin" + 6 "ebitda" mentions. Looked like a bug, verified via `bm25.idf` directly: "margin" appears in only 4/25 chunks (idf 1.56) vs "ebitda" in 8/25 (idf 0.72) — correct BM25 behavior, just exposed by a tiny single-document corpus where per-term document-frequency stats are noisy. Will settle down once more companies join the DOCUMENTS registry.
**Real finding, hybrid winning clearly:** ran "how has the company's phone and internet business grown" (deliberate paraphrase, no exact term overlap with the right answer). BM25-only top result was the Chairman's shareholder letter (matched on generic words like "company"/"grown") — completely wrong. Vector-only got it right (Jio content, top 3). Hybrid's RRF fusion pulled both real Jio chunks to #1/#2 — proof hybrid recovers from a single ranker's blind spot, not just a claim.
**Known residual noise:** hybrid top-5 still let a table-of-contents chunk in at #4 (it scored well on BOTH BM25 — keyword-dense TOC — and sat somewhere in vector's wider candidate pool). Expected — that's precision cleanup, Day 5's cross-encoder reranker's job, not this layer's.
**Quiz:** not yet run.
**Weak / revisit:** Day 2 + Day 3 quizzes still outstanding, now 3 days deep in backlog.

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
