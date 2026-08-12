# Phase 3 — Train the tokenizer

**Goal:** a fresh **16,384-token byte-level BPE**, trained from scratch on our own
corpus, saved as a HuggingFace `PreTrainedTokenizerFast` at `$SLM_ROOT/tokenizer/`.

```bash
python -m slm.tokenizer            # train + save + validate
python -m slm.tokenizer --validate # re-run checks on an existing tokenizer
```

## Why these choices

- **Byte-level → no out-of-vocabulary token, ever.** Every byte 0–255 is in the
  initial alphabet, so any input encodes. This matters here specifically: Phase 1
  showed the case-law corpus carries real OCR damage (`supe COURT OE ALABAMA`,
  `HAWAT'T`), and a byte-level vocabulary degrades gracefully on it instead of
  emitting `<unk>`.
- **16K is deliberately small.** At 125M params the embedding table is
  16,384 × 768 = 12.6M params — exactly **10.0 %** of the model. A larger vocab
  would buy shorter sequences at the cost of transformer capacity, which is the
  wrong trade at this size.
- **Trained on the deduplicated corpus** (`config.CORPUS_DIR` → `dedup/`), not
  the raw one. A tokenizer fitted on duplicate-heavy text over-weights whatever
  was duplicated.
- **~3 GB proportional sample, not the whole 16.5 GB.** BPE merge tables
  converge well before the full corpus; sampling proportionally by source keeps
  the merge statistics representative while keeping this phase under an hour.
  The sample is streamed through a generator and never lands on disk.

## Special tokens

Reserved at train time so they occupy low, stable ids:
`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, plus `<|user|>`, `<|assistant|>`,
`<|system|>` for later alignment work. `<|eos|>` is the one Phase 4 depends on —
it separates documents in the packed token stream.

## Validation performed

1. **Round-trip** `decode(encode(x)) == x` on samples from all three sources.
2. **No-UNK guarantee** on deliberately hostile input: OCR garble, emoji, CJK,
   control bytes, raw bytes 0–255.
3. **Compression** measured per source in chars/token. This replaces the
   `CHARS_PER_TOKEN_ESTIMATE = 4.0` placeholder used for budget accounting since
   Phase 1, and therefore **corrects the true corpus token count** — which sets
   `STEPS_PER_EPOCH` for training.
4. **Vocab size and special-token ids** are what we asked for.

## Known downstream gotcha (recorded now, bites later)

`llama.cpp`'s `convert_hf_to_gguf.py` rejects a freshly trained byte-level BPE
with `BPE pre-tokenizer was not recognized`, and the tokenizer hash changes on
every retrain. If we export GGUF in Phase 6, patch the converter to map our
unknown byte-level BPE onto the `gpt-2` pre-tokenizer, and keep the patch
idempotent. Not needed for the local FastAPI server.

---

# RESULTS — **PASS** (2026-08-12)

Trained in **0.9 minutes** on a 3.22 GB proportional sample (1.06 GB case-law,
1.39 GB sec, 0.77 GB fineweb-edu). The `tokenizers` trainer is Rust-threaded and
saturated the box on its own.

## Artifact

```
$SLM_ROOT/tokenizer/
├── tokenizer.json          1.14 MB   vocab + merges
└── tokenizer_config.json
```

| Property | Value |
|---|---|
| vocab_size | 16,384 |
| model_max_length | 1024 |
| `<\|bos\|>` / `<\|eos\|>` / `<\|pad\|>` / `<\|unk\|>` | ids 0 / 1 / 2 / 3 |

Recent `transformers` folds the special-token map into `tokenizer_config.json`,
so no separate `special_tokens_map.json` is written. Loading via
`AutoTokenizer.from_pretrained` resolves all four correctly.

## No-UNK guarantee holds on hostile input

Every case round-tripped exactly with zero UNK tokens:

| Input | chars → tokens |
|---|---|
| OCR garble (`supe COURT OE ALABAMA … HAWAT'T ggg`) | 58 → 28 |
| emoji (incl. ZWJ sequence `👨‍⚖️`) | 53 → 37 |
| CJK | 19 → 57 |
| control bytes (`\x00 \x01 \x1f \x7f`) | 9 → 9 |
| **all 256 byte values** | 256 → 358 |
| math/typographic unicode | 33 → 31 |

This is the property that justifies byte-level for this corpus specifically:
Phase 1 confirmed real OCR damage in case-law, and the tokenizer degrades
gracefully on it rather than collapsing to `<unk>`.

## Compression — measured, and it moved the training schedule

Measured with the trained tokenizer across **every shard** (250 docs each),
weighted by each source's true char count:

| Source | chars/token | corpus chars | → tokens |
|---|---|---|---|
| case-law | 4.479 | 5.49B | 1.226B |
| sec | 5.037 | 7.05B | 1.400B |
| fineweb-edu | 4.291 | 3.96B | 0.923B |
| **corpus-weighted** | **4.65** | **16.5B** | **3.548B** |

The 4.0 chars/token placeholder used since Phase 1 was pessimistic. The corpus is
**3.548B tokens, not 4.124B** — that is a *measurement correction*, not data loss.
SEC compresses best (5.04), which is expected: filings are formulaic and
repetitive, so BPE finds long reusable merges.

**Caution worth recording:** the naive "overall" ratio printed by the validation
run was 4.875, which is wrong. It pooled a fixed 300 docs per source, and SEC
documents are ~40× longer than fineweb-edu ones, so SEC dominated the pooled
average. Any per-source statistic aggregated over documents of wildly different
lengths has to be weighted by size, not document count.

## Consequence for training

```
corpus       4.124B -> 3.548B
steps/epoch   7,865 -> 6,767
max steps    39,325 -> 33,835   (still 5 epochs)
tokens seen   20.6B -> 17.7B
tok/param       164 -> 141      (still ~7x Chinchilla)
Phase 5 time  ~44 h -> ~38 h    (1.6 days)
electricity  70 kWh -> 61 kWh
token files  8.2 GB -> 7.1 GB
```

`config.corpus_tokens()` now reads the **exact** total from Phase 4's
`index.json` once it exists, falling back to this estimate until then. The
schedule stops depending on a hand-maintained constant.
