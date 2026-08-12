# Phase 2 — Dedup + contamination strip

**Goal:** remove duplicate and near-duplicate documents, and remove anything
resembling the evaluation benchmarks, so held-out perplexity stays honest.

Input `$SLM_ROOT/clean/` (79 shards, 1.42M docs, 5.25B tokens) →
output `$SLM_ROOT/dedup/`. **`clean/` is never modified** — it is the durable
asset, and a bad dedup run must be repeatable without re-streaming.

## What runs

```bash
python -m slm.dedup            # all sources
python -m slm.dedup --report   # analyse only, write nothing
```

| Stage | Applies to | Method |
|---|---|---|
| 1. Exact dedup | all sources | blake2b of whitespace-normalised, lowercased text |
| 2. Near-dedup | case-law | MinHash-LSH, 5-word shingles, 64 perms, Jaccard 0.8 |
| 3. Decontamination | all sources | 13-gram overlap against CaseHOLD / LexGLUE |

## Why MinHash only on case-law

Court opinions are massively republished — the same opinion appears across
reporters, with headnotes and citation formats differing just enough that exact
hashing misses it. SEC filings are already unique per accession number, and
fineweb-edu was deduplicated upstream by HuggingFace. Both still get exact
dedup, which is nearly free.

## Implementation notes

- **Bottom-k shingle sampling.** Documents average ~4,400 words → ~4,400
  shingles. We keep the 4,096 smallest shingle hashes per document. Selecting by
  hash value is *consistent across documents*, so the Jaccard estimate stays
  valid — unlike random sampling, which would break it.
- **Vectorised MinHash.** Signatures are computed in numpy
  (`min over shingles of (a·h + b) mod p`), not with per-shingle Python loops,
  and fanned out over `Pool(44)`. 511k signatures at 64×4 bytes is 131 MB — with
  251 GB of RAM the whole index is trivially in-memory.
- **Banding: 8 bands × 8 rows.** Gives an LSH S-curve threshold of
  `(1/8)^(1/8) ≈ 0.76`, the closest practical fit to the target Jaccard 0.8.
- **Bucket-representative clustering,** not all-pairs. Within an LSH bucket each
  member is verified against the bucket's first document and unioned if the
  signature agreement is ≥ 0.8. Legal boilerplate creates very large buckets, and
  all-pairs inside them would be quadratic.
- **Decontamination threshold: ≥ 5 distinct matching 13-grams.** A single shared
  13-gram is too strict for legal text, where citation formulae and statutory
  language legitimately recur across unrelated documents.

## Success criteria

1. `dedup/` written, `clean/` untouched.
2. Kept/dropped counts per source, split by exact / near / contaminated.
3. Duplicate rate in case-law materially above the other two sources — if it
   isn't, the MinHash stage is not working.

---

# RESULTS

## The bug that mattered: hash width

The first run flagged **423,146 of 511,339 case-law docs (83 %) as contaminated**.
That was not a finding, it was a collision bug.

N-gram hashes were masked to 32 bits. A membership test against an index of
M entries has false-match probability `M / 2**bits` per gram:

| Hash width | P(false match / gram) | Expected FALSE hits per 4,400-gram doc |
|---|---|---|
| 32-bit | 6.4 × 10⁻³ | **28.15** |
| 64-bit | 1.5 × 10⁻¹² | ~0 |

With `CONTAM_MIN_HITS = 5`, the average document accumulated ~28 *false* hits
before seeing a single real one — so essentially the entire corpus tripped the
threshold. Widened decontamination hashing to 64 bits.

MinHash shingling deliberately **stays at 32 bits**: it compares ~4,400-element
sets against each other rather than against a 27.5M-entry index, so the expected
false intersection is ~0.005 grams. That path was validated against ground-truth
Jaccard and left alone.

**Verification after the fix**

| Check | Result |
|---|---|
| True positive (a real SCOTUS test doc) | **10,745 hits** — unambiguous |
| False-positive floor | median **0** hits on all three sources (was ~28) |
| case-law flagged | 21.4 % |
| sec flagged | 6.5 % |
| fineweb-edu flagged | 0.5 % |

That last ordering is the strongest evidence the detector measures real shared
text: flag rate tracks domain proximity to the legal benchmarks, and general web
text barely registers.

**Lesson worth keeping:** a hash-collision bug does not crash, it silently
returns plausible-looking numbers. The only reason it was caught is that 83 %
was *implausibly* high. A subtler threshold would have quietly deleted good data.

## Distribution of contamination hits (case-law, 2,500-doc sample)

| Hits | Share of flagged | Interpretation |
|---|---|---|
| 5–9 | 21 % | boilerplate: reporter headers, citation formulae |
| 10–49 | 50 % | substantial genuine overlap |
| 50–199 | 24 % | near-certain |
| ≥ 200 | 5 % | the document *is* a benchmark case |

Median 26 hits. Lowest scorers trip on strings like
`*** FOR PUBLICATION IN WEST'S HAWAI REPORTS AND PACIFIC REPORTER ***`; the
highest (543, 746 hits) are Mississippi Supreme Court cases plainly present in
the benchmark sets.

## Final counts

| Source | In | Kept | % | Exact | Near | Contaminated |
|---|---|---|---|---|---|---|
| case-law | 511,339 | 398,882 | 78.0 % | 154 | **11,756** | 100,547 |
| sec | 72,496 | 66,863 | 92.2 % | 3,200 | 0 | 2,433 |
| fineweb-edu | 837,755 | 833,600 | 99.5 % | 2,561 | 0 | 1,594 |
| **Total** | **1,421,590** | **1,299,345** | **91.4 %** | 5,915 | 11,756 | 104,574 |

Runtime 15 min total. `clean/` 6.0 GB untouched, `dedup/` 4.8 GB.

## Tokens

| Source | Chars | Tokens |
|---|---|---|
| case-law | 5.49B | 1.372B |
| sec | 7.05B | 1.761B |
| fineweb-edu | 3.96B | 0.991B |
| **Total** | **16.5B** | **4.124B** |

Note the mix inverted: SEC is now the largest source (43 %), because case-law
absorbed nearly all the near-duplicate and contamination losses.

## Near-dedup validated the design

case-law: **154 exact vs 11,756 near-duplicates** — a 76× gap. Those are
opinions republished across reporters with differing headnotes and citation
formats, invisible to blake2b. sec and fineweb-edu produced zero near-duplicates
worth chasing, confirming MinHash was correctly scoped to case-law alone.

## Decision: keep decontamination at threshold 5

Decontamination cost ~1.0B tokens, more than 4× what real duplicates cost
(17,671 docs). Kept anyway, to preserve the option of honest LexGLUE/CaseHOLD
evaluation. Consequence for training:

```
corpus        5.25B  -> 4.124B
steps/epoch  10,013  -> 7,865
max steps    50,065  -> 39,325   (still 5 epochs)
tokens seen   26.2B  -> 20.6B
tok/param       209  -> 164      (still ~8x Chinchilla)
Phase 5 time   ~56 h -> ~44 h    (1.8 days)
electricity  90 kWh  -> 70 kWh
token files   10.5GB -> 8.2 GB
```

## Downstream requirement

`config.CORPUS_DIR` now points at `dedup/`. Phases 3 and 4 must read from it —
reading `clean/` would silently reintroduce all 122,245 removed documents.
