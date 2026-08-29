# Chapter 2: Chunking — from raw pages to retrieval units

*Textbook-style notes for Day 2 of DocuAgent. Every concept below is followed
by a worked example using our actual Reliance Industries data, not an
abstract toy case.*

---

## 2.1 The problem this chapter solves

At the end of Day 1 we had, for the Reliance FY2024 report, page 4's raw text:

```
Dear Shareholders,
Nearly fifty years ago, our visionary founder, Shri Dhirubhai Ambani,
embarked on a bold mission to prove that India could build a
world-class enterprise founded on innovation, integrity, and ambition...

At the heart of Reliance's success is a foundation built on values, trust,
and talent. Our strong balance sheet, relentless focus on productivity...
```

This whole page is **one long string.** If we hand this entire string to an
embedding model, it produces *one* vector that tries to represent everything
on the page at once — the founder's history, the balance-sheet comment,
whatever else is on that page — all blurred into a single point in vector
space. When a user later asks "what did the Chairman say about strategy?",
that one blurry vector competes against equally-blurry vectors from 145
other pages. Retrieval quality collapses.

**The fix:** break documents into smaller, focused pieces — *chunks* — each
one about a single coherent idea, small enough that its embedding stays
sharp, but not so small it loses context. This chapter is about *how* to
decide where the cuts go.

---

## 2.2 Tokens and BPE — measuring "how much text is this?"

### The idea

Before we can say "keep chunks under 512 tokens," we need a way to *count*
tokens. A token is not a word and not a character — it's whatever unit the
language model's tokenizer decided to use, and that unit is learned from
data using an algorithm called **Byte Pair Encoding (BPE)**.

BPE works like this, conceptually:

1. Start by treating every individual character as its own token.
2. Scan a huge pile of text, find the *most frequent adjacent pair* of
   tokens, and merge it into a single new token.
3. Repeat step 2 tens of thousands of times.

The result: extremely common sequences ("the", "ing", "tion") become single
tokens early, because they were frequent pairs early and often. Rare or
unusual sequences (like a jargon term) never get merged all the way, so they
stay split into multiple pieces.

### Worked example

Using `tiktoken`'s `cl100k_base` encoding (the ruler our chunker uses):

| Text | Token count | Why |
|---|---|---|
| `"the"` | 1 | extremely common word, merged into 1 token long ago |
| `"Reliance"` | 2 | company name, not common enough to be 1 token |
| `"EBITDA"` | 3 | financial jargon, rare in general English, splits into pieces like `EB` + `IT` + `DA` |
| `"₹28,500 crores"` | ~6 | currency symbol, digits, and "crores" all cost separate tokens |

**Lesson:** financial text is *token-expensive* compared to plain English.
"EBITDA" costs 3x what "the" costs, even though a human reads both as "one
word." This is exactly why we can't estimate chunk size by word-count or
character-count — we have to actually run the tokenizer.

### Why `cl100k_base` and not Gemini's own tokenizer

We're using Gemini for embeddings and generation, but we count tokens with
`cl100k_base` — the encoding OpenAI's older models used. Google doesn't
publish a convenient pip-installable tokenizer for Gemini, so `cl100k_base`
is the free, offline, "close enough" ruler the whole open-source RAG
community defaults to for chunk-sizing. It won't match Gemini's real count
exactly (maybe ±10-15%), but for the *purpose* — "is this chunk roughly the
right size?" — exact precision doesn't matter. We're sizing a box, not
billing an invoice.

---

## 2.3 Splitting into paragraphs — regex worked example

### The idea

Financial narrative text (the Chairman's letter, MD&A) should split at
*paragraph* boundaries, not arbitrary character counts, because a paragraph
is usually one coherent thought. When a PDF is extracted to plain text,
paragraph breaks survive as **blank lines** — one or more empty lines
between blocks of text.

### The pattern

```python
re.split(r"\n\s*\n", text)
```

Read it piece by piece:
- `\n` — a newline character (end of a line)
- `\s*` — zero or more whitespace characters (covers stray spaces/tabs
  sitting alone on an otherwise-blank line)
- `\n` — another newline

Put together: *"split the text everywhere you see a newline, then any
whitespace, then another newline"* — in plain terms, split on blank lines.

### Worked example

Raw page text (simplified):
```
Dear Shareholders,
Nearly fifty years ago...

At the heart of Reliance's success is a foundation built on values...
```

After `re.split(r"\n\s*\n", text)`:
```python
[
  "Dear Shareholders,\nNearly fifty years ago...",
  "At the heart of Reliance's success is a foundation built on values...",
]
```
Two paragraphs, cleanly separated. Each one is now a candidate chunk (or
part of one, after the grouping step below).

---

## 2.4 Greedy grouping — a bin-packing algorithm

### The problem, formally

We have a list of paragraphs, each with a token count. We want to group
consecutive paragraphs into chunks, where **no chunk exceeds 512 tokens**,
and we want as few chunks as possible (fewer chunks = less redundant
overhead per chunk).

This is a version of the classic **bin-packing problem**: pack items into
fixed-capacity bins, minimize the number of bins. True optimal bin-packing
is *NP-hard* — for large inputs, no algorithm can guarantee the best answer
in reasonable time. We don't need the mathematically optimal packing (we're
not shipping boxes) — we need something fast and good enough.

### The algorithm: greedy first-fit

```
current_bin = []
current_tokens = 0

for paragraph in paragraphs:
    if current_bin is not empty AND current_tokens + paragraph.tokens > 512:
        seal current_bin as a finished chunk
        start a new empty current_bin
    add paragraph to current_bin
    current_tokens += paragraph.tokens

seal the last current_bin if non-empty
```

This is a **single pass through the list — O(n) time.** No backtracking, no
trying different combinations. Just: keep adding to the current bin until
the next item won't fit, then start fresh.

### Worked example

Say a page has 4 paragraphs with these token counts: `300, 150, 100, 400`.
Cap = 512.

| Step | Paragraph | Tokens | Running total | Action |
|---|---|---|---|---|
| 1 | P1 | 300 | 300 | fits, add to bin 1 |
| 2 | P2 | 150 | 450 | fits (450 ≤ 512), add to bin 1 |
| 3 | P3 | 100 | 550 | **doesn't fit** (550 > 512) → seal bin 1 (P1+P2, 450 tokens), start bin 2 with P3 (100) |
| 4 | P4 | 400 | 500 | fits (500 ≤ 512), add to bin 2 |

Result: **2 chunks** — `[P1, P2]` at 450 tokens, `[P3, P4]` at 500 tokens.
Notice bin 1 stopped at 450, leaving 62 tokens "wasted" — greedy isn't
perfectly efficient, but it never has to backtrack or compare combinations,
which is the whole point.

### The one rule that overrides the cap

What if a *single* paragraph is 700 tokens — bigger than the 512 cap by
itself? Our algorithm still adds it as its own chunk, oversized:

```
if current_bin is not empty AND ... : seal and restart   ← only fires if bin is non-empty
add paragraph to current_bin                              ← always happens
```

A lone 700-token paragraph never gets split mid-sentence — it just becomes
a 700-token chunk on its own. **Why this is the right tradeoff:** cutting a
paragraph in half might separate a claim from its number, or a sentence
from its conclusion — that damage to meaning is worse than one chunk being
40% over budget. Financial writing especially: "Revenue grew 12%, driven by
strength in retail and telecom" — cut after "12%" and you've lost *why*.

---

## 2.5 Section detection — a case study in heuristic fragility

### The idea

Tag each chunk with what kind of content it is (`balance_sheet`,
`risk_factors`, `narrative`, ...) so later we can filter — "only search
balance sheet chunks" — and cite sources properly. The simplest possible
approach: look for keywords that would appear near a real heading.

```python
if "balance sheet" in page_text.lower():
    return "balance_sheet"
```

### What went wrong — the real bug

Page 4 of the Reliance report is the **Chairman's letter**, not a financial
statement. It contains this sentence:

> "...our strong **balance sheet**, relentless focus on productivity..."

Our keyword rule saw the phrase "balance sheet" and confidently tagged the
*entire page* as the `balance_sheet` section — even though this is a
CEO's letter using the phrase colloquially, not an actual accounting
statement.

### The fix — add a structural signal, not more keywords

The insight: a **real** balance sheet page always has an actual table on
it (rows of assets, liabilities, numbers) because `pdfplumber` extracts
that grid in Layer 1. A page that merely *mentions* "balance sheet" in a
sentence has no such table.

```python
def detect_section_type(page_text, has_tables):
    for section_type, keywords, requires_table in _SECTION_KEYWORDS:
        if requires_table and not has_tables:
            continue          # skip this rule — no table, can't be the real statement
        if any(kw in page_text.lower() for kw in keywords):
            return section_type
    return "narrative"
```

Now: keyword match + `requires_table=True` + `has_tables=False` → the rule
is skipped entirely, page falls through to `narrative`. Correct.

### The lesson, generalized

This is a pattern worth remembering for *any* heuristic system: when a
lexical rule (matching words) produces a false positive, the fix usually
isn't "add more words to the blocklist" — it's **adding a structural
signal** the false-positive case doesn't have. Here: real financial
statements have tables; prose mentions don't. One boolean check replaced
what would have been an endless whack-a-mole of exception phrases.

---

## 2.6 Chapter summary

| Concept | One-line takeaway |
|---|---|
| Tokens (BPE) | Not words, not chars — learned sub-word units; financial jargon costs more tokens than plain English |
| `cl100k_base` | A free stand-in ruler for token-counting since Gemini has no pip-installable tokenizer |
| Paragraph regex | Blank lines (`\n\s*\n`) mark paragraph boundaries in extracted PDF text |
| Greedy bin-packing | O(n) single-pass grouping; good-enough, not optimal — optimal bin-packing is NP-hard |
| Paragraph-atomic rule | Never split a paragraph mid-sentence, even if it blows the token cap |
| Section detection bug | Keyword-only rules false-positive on prose mentions; fix with a structural guard (table presence), not more keywords |

## 2.7 Check your understanding

Try answering before checking `PROGRESS.md` / asking:

1. Why does "EBITDA" cost more tokens than "the," even though both are one
   word to a human reader?
2. You have paragraphs of size `200, 200, 200` tokens and a cap of 512.
   Walk through the greedy algorithm — how many chunks, and what's in each?
3. A single paragraph is 900 tokens, over the 512 cap. What does our
   chunker do with it, and why is that the right call?
4. Why did "our strong balance sheet" in a CEO's letter get mis-tagged, and
   what single piece of information fixed it?
