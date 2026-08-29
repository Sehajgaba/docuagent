# Interview Prep — master tracker

**Goal:** by the end of this build, two things must both be true:
1. **Nothing about this project is unknown to me** — I can explain every
   line, every choice, every tradeoff, cold, no notes.
2. **I can hold my own on the broader AI-engineer interview surface** —
   topics this project touches but doesn't fully teach (DSA, system design
   at scale, general ML knowledge).

This file is the question bank + gap list. `PROGRESS.md` tracks day-by-day
build progress and per-concept mastery (🔴🟡🟢⭐). This file is what an
interviewer might actually ask, organized by topic, plus what's missing
outside the project entirely.

---

## Part A — Questions about THIS project (must answer cold)

*Model answers are intentionally short — the point is to internalize the
idea, not memorize a script. Full depth is in `docs/day1-ingestion.md` and
`docs/day2-chunking.md`, expanding as we build.*

### A1. Fundamentals

**Q: What's the difference between an LLM, RAG, and an agent?**
> LLM answers from what it memorized during training. RAG answers from
> documents you retrieve and hand it at query time. An agent decides
> *which actions to take* (including whether to retrieve) via a loop — RAG
> is one tool an agent might use, not a separate category.

**Q: What is an embedding?**
> A fixed-length vector representing a text's meaning, produced by a model
> trained so that semantically similar texts land close together in that
> vector space.

**Q: Cosine similarity — range, and what do real embeddings actually look like?**
> Mathematically −1 to 1. In practice, sentence embeddings for unrelated
> text cluster near 0, not near −1 — true opposites rarely appear in
> real training data. This matters for any "keep the best score" loop:
> initialize below the true floor (−1), not at 0.

**Q: Bi-encoder vs cross-encoder — when do you use each?**
> Bi-encoder embeds query and document *separately*, compares by cosine —
> fast, scales to millions, less precise. Cross-encoder feeds query+doc
> *together* through one model — slow, but sees their interaction
> directly, so it's precise. Use bi-encoder to retrieve many candidates
> fast, cross-encoder to rerank the top few precisely. *(Rerank = Day 5,
> not built yet — flag if asked for our specific numbers.)*

### A2. Ingestion & data

**Q: Why two PDF libraries instead of one?**
> `pymupdf` is fast at general text/layout extraction; `pdfplumber`
> specifically reconstructs table grids from ruling lines and whitespace
> alignment. Neither is strictly "better" — they solve different
> sub-problems of PDF parsing.

**Q: How do you handle Indian financial number formats?**
> Three rules: parentheses mean negative (accounting convention), commas
> are pure visual grouping so they're stripped (regardless of Western vs
> Indian grouping pattern — both just get commas removed), currency
> symbols/words (`₹`, `Crores`) are stripped before parsing.

**Q: What's the risk of scanned (image) PDFs, and how would you handle them?**
> `pymupdf`/`pdfplumber` extract *embedded* text — a scanned page is just
> an image, no embedded text exists, extraction returns empty. Fix: OCR
> (`pytesseract`/`paddleocr`) as a fallback when a page's extracted text is
> suspiciously short relative to its visual content. *(Not yet built —
> flag as a known gap if pushed on scanned-PDF handling.)*

### A3. Chunking

**Q: Why not just split every 512 tokens?**
> Fixed-size splitting ignores document structure — it can cut a table
> mid-row, separating a number from its row label, or cut a sentence in
> half. We chunk by structure instead: whole tables stay whole, narrative
> splits by paragraph and groups under a token cap without ever breaking
> a paragraph mid-sentence.

**Q: What algorithm groups paragraphs into chunks?**
> Greedy first-fit bin-packing — single pass, O(n): keep adding paragraphs
> to the current chunk until the next one would overflow the cap, then
> start a new chunk. Not globally optimal (true bin-packing is NP-hard),
> but simple, fast, and good enough — we're sizing chunks, not solving a
> logistics problem.

**Q: Tell me about a real bug you hit and fixed.**
> Keyword-based section tagging mis-classified a page as `balance_sheet`
> because the Chairman's letter said "our strong balance sheet" in prose —
> a false positive from lexical matching alone. Fixed by requiring an
> actual extracted table on the page before trusting that keyword — a
> structural signal instead of more keyword patches. General lesson:
> when a lexical heuristic false-positives, look for a structural signal
> the bad case lacks, rather than trying to out-list every exception.

### A4. Retrieval (Day 3-5, build in progress)

*(Fill in once built — Qdrant/HNSW, BM25, RRF fusion, reranking. Track
here as each lands so this file always mirrors the real system.)*

### A5. Generation & agents (Day 6-8, not yet built)

*(ReAct loop, tool calling, grounding/anti-hallucination, agentic RAG.
See the "AI Agents & RAG Field Guide" artifact for the target explanations
— convert to project-specific answers once built.)*

### A6. Evaluation & ops (Day 9-12, not yet built)

*(RAGAS's 4 metrics and which layer each one diagnoses, observability/
tracing, why Docker, why this deploy target over alternatives.)*

---

## Part B — Beyond this project (the real gap)

The project will NOT teach these on its own. These need separate study.
Interviewers for an "AI engineer" role probe here even if the role is
mostly RAG-flavored.

### B1. DSA fundamentals (self-identified weak area)
- Big-O analysis — you'll need this to explain *why* HNSW (O(log n)) beats
  brute-force (O(n)), and why greedy bin-packing is O(n). Practice stating
  complexity for anything you build.
- Core structures: hash maps, trees, graphs (HNSW *is* a graph — a
  layered navigable small-world graph. Understanding graph traversal
  concepts directly helps explain it).
- Sorting/searching basics — reranking is fundamentally a sort-by-score
  problem.

### B2. System design depth (self-identified weak area)
- How do you scale vector search to 100M+ vectors? (sharding, replication,
  approximate indexes, quantization to shrink vector size in memory)
- Caching strategy for a RAG API — what's cacheable (embeddings of common
  queries) vs not (fresh generation)?
- Rate limiting / cost control when calling a paid LLM API at scale.
- Latency budget: where does time go in a RAG request (embed query →
  vector search → rerank → LLM generation) and how would you shave it?

### B3. Retrieval tradeoffs (deepen after the build)
- HNSW vs IVF (inverted file index) — HNSW = better recall, more memory;
  IVF = less memory, needs a training/clustering step.
- Chunk size effects — smaller chunks = more precise retrieval, more
  chunks to search, more embedding calls, less context per chunk for the
  LLM to reason over. Bigger chunks = opposite tradeoffs.
- Why RRF over just averaging BM25 and vector scores? (Scores from
  different methods aren't on comparable scales — rank position is;
  raw-score fusion needs careful normalization RRF sidesteps.)

### B4. General ML/AI knowledge (interviewers may probe outside RAG specifics)
- Overfitting vs underfitting, and why it's less relevant when you're
  *using* pretrained models rather than training your own — but you should
  still be able to explain it.
- Fine-tuning vs RAG vs prompt engineering — when would you reach for each?
  (Fine-tuning: change model *behavior* durably, needs data + compute.
  RAG: inject *fresh/private knowledge* without retraining. Prompting:
  cheapest, works when the model already "knows" what you need, just needs
  steering.)
- Temperature, top-p — what they control in generation, why RAG prompts
  usually use low/zero temperature (determinism, less creative drift from
  grounded facts).

### B5. Behavioral / "tell me about a time" bank
Build this list as real moments happen during the build — they're the
most memorable interview answers because they're true and specific:
- The `balance_sheet` false-positive bug (A3 above) — heuristic fragility,
  fixing with structural signal.
- The `best_score = 0` cosine bug — off-by-assumption on a metric's true
  range.
- Choosing free-tier tools (Gemini/Qdrant/local reranker) over the brief's
  paid defaults (Cohere) — cost-driven engineering tradeoff reasoning.

---

## Part C — Gap tracker

Running list. Add an entry the moment a question exposes a hole — don't
wait for a "review day."

| Date | Gap found | How exposed | Status |
|---|---|---|---|
| 2026-07-19 | Cosine: unrelated-text score ≈ 0, not −1 (practical vs theoretical range) | quiz re-take, missed twice | open — re-quiz again next round |
| 2026-07-19 | Didn't name "single source of truth" unprompted | quiz | open — re-quiz |

*(Keep appending. This table is the honest record of what's actually
sticking vs what needs another pass — don't prune it to look better.)*
