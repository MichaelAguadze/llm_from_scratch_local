# Phase 4 — Tokenize + pack + split

**Goal:** turn the 1.3M-document deduplicated corpus into fixed 1024-token
windows of `uint16`, split 99/1 train/val, plus an `index.json` that becomes the
**authoritative token count** for the training schedule.

```bash
python -m slm.pack              # tokenize, pack, split, index
python -m slm.pack --verify     # re-check an existing token set
```

## The on-disk format

- Each document is batch-encoded to token ids, then an **`<|eos|>` (id 1) is
  appended** as a document separator.
- The flat id stream is packed into fixed **1024-token windows**. No padding
  inside a window; EOS marks the boundaries.
- Windows are stored as **`uint16` `.bin` shards** — a 16,384 vocab fits in 16
  bits, halving the disk and the page-cache footprint versus `int32`.
- `index.json` per split records shard names, window counts, total tokens,
  dtype, and `seq_len`.

## Deviation from the guide: split by **document**, not by window

The guide specifies "every 100th packed window goes to val ... so no validation
window can leak into training." **By-window splitting does not achieve that
goal**, and this build splits by document instead.

Documents here average ~2,700 tokens (case-law) to ~19,000 (SEC), so a single
document spans 3–19 consecutive windows. Taking every 100th *window* therefore
puts a val window in the middle of a document whose surrounding windows are all
in train — the model reads the paragraphs immediately before and after the held-
out text. Validation perplexity would come out optimistically low, and the
early-stopping signal that governs a 5-epoch run would be measuring partly
memorised context.

Splitting by document (every 100th document → val, deterministic per shard)
costs nothing and makes the held-out set genuinely held out. Same 99/1 ratio,
same reproducibility, same proportional coverage of all three sources.

## Implementation notes

- **One worker per input shard, fully independent.** Each worker tokenizes its
  shard, packs its own train and val streams, and writes its own `.bin`. No
  cross-shard state, so no merge pass.
- **Per-shard remainders are dropped** — the final partial window of each
  stream, under 1024 tokens. Worst case 79 shards × 2 streams × 1023 tokens
  ≈ 162k tokens, or 0.005 % of the corpus.
- **`TOKENIZERS_PARALLELISM=false` in workers.** The Rust tokenizer's internal
  threading fights a 44-process pool and deadlock-warns on fork.
- **Memory-mapped at training time.** 7 GB against 251 GB of RAM means the whole
  token set lands in page cache after the first pass; no prefetch pipeline is
  warranted.

## Success criteria

1. `tokens/train/` and `tokens/val/` written with `index.json` each.
2. Val fraction ≈ 1 %, and **no document contributes to both splits**.
3. Every token id < 16,384; no id exceeds the `uint16` range.
4. A window decodes back to fluent text, with EOS at document boundaries.
5. `config.corpus_tokens()` picks up the exact count automatically.

---

# RESULTS — **PASS** (2026-08-12)

Tokenized 1,299,345 documents in **5.9 minutes** across 44 workers.

| Split | Tokens | Windows | Shards | On disk |
|---|---|---|---|---|
| train | **3,535,785,984** | 3,452,916 | 63 | 6.6 GB |
| val | 35,796,992 | 34,958 | 63 | 69 MB |

Val fraction **1.00 %**. Remainder dropped: 59,123 tokens (**0.0017 %**).

| Source | train | val | docs |
|---|---|---|---|
| case-law | 1.229B | 12.2M | 398,882 |
| sec | 1.386B | 14.5M | 66,863 |
| fineweb-edu | 0.921B | 9.1M | 833,600 |

## Verification

| Check | Result |
|---|---|
| byte count vs index | OK both splits (`tokens × 2` exactly) |
| token id range | min 1, max 16,383 — inside vocab 16,384 |
| EOS density | ~2,276 tokens/doc |
| window decodes to fluent text | yes (Alabama probate opinion, verbatim) |
| `config.corpus_tokens()` | picks up 3,535,785,984 automatically |

Schedule is now exact, not estimated: **6,743 steps/epoch, 33,715 max steps,
17.7B tokens seen, 140 tokens/parameter.**

## The leakage audit — and what it found

Because the by-document split was a deliberate deviation, it was verified
empirically rather than assumed. Searching 40-token val n-grams against train
found matches — but **not** from the split: the split is by document, so one
document's tokens cannot reach both sides. The matches were *different*
documents sharing text, all of them in SEC:

- Palo Verde / APS — same company, different filing years (`APS'` vs `the Company's`)
- Trust Unit distribution clause — identical across 100 tokens of context
- Casino licensure statute — identical recitation by unrelated filers
- Gas agreement exhibit index — same company, different year (`10.1.22` vs `10.2.20`)

Quantified over ~36,000 decorrelated 40-gram samples against 500M train tokens
per source:

| Source | val n-grams also in train |
|---|---|
| case-law | 1.2 % |
| **sec** | **19.3 %** |
| fineweb-edu | 0.8 % |

**This contradicts a Phase 2 assumption.** MinHash was scoped to case-law on the
reasoning that "SEC filings are already unique per accession number" — true of
documents, false of content. Re-running MinHash on SEC catches only 2,709 docs
(3.7 %), because the redundancy is **sub-document**: annual filings share whole
sections while differing enough overall to sit under any sane Jaccard threshold.
No document-level dedup fixes this.

## Decision: per-source validation, early-stop on the clean sources

SEC remains in training — its content is wanted. But it is excluded from the
early-stopping signal:

```python
EARLY_STOP_SOURCES = ("case-law", "fineweb-edu")
EVAL_PER_SOURCE = True
```

Rationale: with 5 epochs over repeated data, early stopping exists to catch
memorisation. Running that detector on a val set that is 19 % pre-seen would
make it blind to the exact failure it guards against. Reporting perplexity
per source keeps SEC's number visible and interpretable rather than silently
folded into a flattering average.

**Generalisable lesson:** a held-out split protects against *split* leakage, not
against redundancy already present in the corpus. Those are different problems,
and only the second one shows up when you actually measure it.
