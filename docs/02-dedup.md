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

_(filled in after the run)_
