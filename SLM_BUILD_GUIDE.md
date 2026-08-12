# SLM Build Guide — Local Edition

Train and serve a **125M-parameter Llama-style decoder-only LM, from scratch**
(fresh weights, fresh tokenizer) on a domain corpus — **entirely on this machine**.

This replaces the Modal/cloud plan in
[docs/SLM_BUILD_GUIDE.modal-original.md](docs/SLM_BUILD_GUIDE.modal-original.md).
Same architecture, same data recipe, same phase discipline. Different substrate:
no Modal App, no Volume, no per-hour GPU billing — just this box, `tmux`, and
local disk.

Scope, in order:

1. Stream the training data from HuggingFace
2. Clean it (rule-based pipeline)
3. Dedup + strip eval contamination
4. Train the tokenizer (fresh 16K byte-level BPE)
5. Tokenize + pack (99/1 train/val split)
6. Pretrain the 125M model on 4× A6000
7. Serve it locally for inference (FastAPI + a small web UI)

No SFT / alignment / RAG in this build. Those are later.

---

## This machine

Measured, not assumed:

| Component | What's here | What it means |
|---|---|---|
| GPU | **4× NVIDIA RTX A6000**, 48 GB each (**192 GB VRAM**) | sm_86 Ampere → native **bf16**, TF32, FlashAttention-2. No FP8. |
| GPU interconnect | PCIe 4.0 ×16, **no NVLink** (NODE/PHB topology) | DDP all-reduce goes over PCIe. Fine at 125M — see Phase 5. |
| GPU power | 300 W cap each | 4 GPUs at full tilt ≈ **1.2 kW** just for compute. |
| CPU | AMD Threadripper 3960X, **24C / 48T** | Cleaning, dedup, and tokenization are all embarrassingly parallel. |
| RAM | **251 GB** (234 GB free) | The entire token corpus fits in page cache. Data loading is free. |
| Driver / CUDA | 570.211.01 / CUDA 12.8 | Runs cu124 and cu128 PyTorch wheels. |
| Disk | **120 GB free** on `/` (492 G, 75 % full) | Enough for the 10B-token corpus + 5-epoch run (~52 G) with ~68 G to spare. Resolved in Phase 0a. |
| `/dev/nvme0n1p4` | 1.4 TB ext4 — **a live Ubuntu 18.04 install, off-limits** | Not scratch space. See Phase 0a. |
| Python | system 3.14.5 (too new for torch), `uv` 0.11.21, miniforge3 | Pin **3.12** in a `uv` venv. |

**The headline:** this box is *stronger* than the single H100 the original guide
budgeted \$15–25 for. 4× A6000 delivers roughly 120 TFLOP/s of usable bf16 —
comparable to one H100 for a model this small, and you own it. The build cost
becomes electricity: ~1.6 kW × ~44 h of GPU time (5 epochs) ≈ **70 kWh, call it
\$9–14**.

**Storage is settled** (see Phase 0a for how): everything durable lives under
`SLM_ROOT=/home/michael/slm-125m` on the root disk, which now has 120 GB free.

---

## Credentials

Create `.env.local` in this directory (never commit it):

```bash
HUGGINGFACE_TOKEN=hf_...                        # only to *push* the finished model
SLM_ROOT=/home/michael/slm-125m                 # all durable artifacts live here
HF_HOME=/home/michael/slm-125m/.hf_cache        # keep streaming scratch off ~/.cache
```

This file already exists with `SLM_ROOT` and `HF_HOME` filled in; add your token
when you reach Phase 6. It is gitignored.

All three source datasets are **ungated**, so streaming works without a token —
but **set one before Phase 1 anyway.** The Hub applies much tighter rate limits
to anonymous requests, and Phase 1 streams continuously at ~1.1M tokens/s.
Phase 0 confirmed this warning fires on every unauthenticated request.

Every phase script starts with `source .env.local`.

---

## How we work (rules for the agent)

Unchanged from the original, and still firm — they exist because the last build
turned into a black box.

- **Lean and transparent, phase by phase.** Do one phase, show the result, stop.
  No multi-hour fire-and-forget runs *without a written plan first*.
- **A short markdown doc before each phase.** Write `docs/NN-<phase>.md`
  explaining what will run and why, before executing it.
- **`config.py` is the single source of truth.** Model geometry, tokenizer vocab,
  data mix, paths — all there. Every other module imports from it.
- **Immutable, small, well-named code.** Pure functions for the data filters, one
  file per concern, type annotations, no in-place mutation.
- **Local addendum: every long phase runs under `tmux` and logs to a file.** An
  SSH drop or a closed laptop lid must not kill a 4-day run. No exceptions.

---

## The compute layout (local)

What each cloud concept maps to:

| Original (Modal) | Here (local) |
|---|---|
| Modal App | a `uv` venv + `Makefile` targets |
| Modal Volume at `/data` | `$SLM_ROOT` = `/home/michael/slm-125m` |
| CPU containers | `multiprocessing.Pool(48)` on the Threadripper |
| 1× / 8× H100 | `torchrun --nproc_per_node=4` over the A6000s |
| Modal secrets | `.env.local` |
| `--max-usd` cost cap | wall-clock + step cap, plus `nvidia-smi -pl` power cap |
| HF Space inference | local FastAPI on `127.0.0.1:8000` |

Directory of record (`$SLM_ROOT`, same shape as the old Volume):

```
$SLM_ROOT/clean/            cleaned .txt.gz shards, per source        (Phase 1–2)
$SLM_ROOT/tokenizer/        the trained 16K byte-level BPE            (Phase 3)
$SLM_ROOT/tokens/train/     99% packed token windows (.bin + index)   (Phase 4)
$SLM_ROOT/tokens/val/        1% packed token windows                  (Phase 4)
$SLM_ROOT/checkpoints/base/ final model (HF safetensors)              (Phase 5)
$SLM_ROOT/checkpoints/last.pt   resume point, rewritten every 500 steps (Phase 5)
$SLM_ROOT/checkpoints/best.pt   best-val-perplexity weights            (Phase 5)
$SLM_ROOT/logs/             training logs, loss curves                (Phase 5)
```

---

## Phase 0 — Storage, environment, smoke test

### 0a. Storage — **done**, recorded here so nobody re-litigates it

The build needs roughly this much durable space:

| Artifact | Size (10B-token build) |
|---|---|
| Cleaned corpus, gzipped `.txt.gz` | ~14 GB (40 GB raw text, ~3× compression) |
| Packed tokens, `uint16` | ~20 GB (10B × 2 bytes) |
| Checkpoints (rolling 3 + `last` + `best`, ~1.6 GB each) | ~8 GB |
| HF cache / scratch during streaming | ~10 GB |
| **Total** | **~52 GB** |

**What we ruled out.** `/dev/nvme0n1p4` (1.4 TB ext4, unmounted, absent from
`/etc/fstab`) looked like free space. It is not: a read-only inspection via
[scripts/setup_storage.sh](scripts/setup_storage.sh) found a **complete Ubuntu
18.04.6 root filesystem** — 3.76 M files, 1005 GB of it under `/home`
(users `exx`/`safwat`, `exxact`, `user1`, `userli`), last booted **2026-01-28**,
and only 144 GB free anyway. It is the workstation's original vendor image with
live user data. **Do not mount it read-write. Do not write to it.** The script
hard-refuses on ≥3 OS markers, so re-running it with `--commit` is safe — it
will abort.

**What we did instead.** `/` was 95 % full, but 98 GB of that was regenerable
cache. Cleared:

| Cache | Freed |
|---|---|
| `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 59 GB |
| `~/.cache/pip` | 17 GB |
| `~/.cache/packman` (Omniverse/IsaacSim) | 17 GB |
| `~/.cache/rattler` (conda/pixi) | 11 GB |
| `~/.cache/uv` | 2.2 GB |

**Result: 25 GB → 120 GB free on `/`.** The HF `token` and `stored_tokens` files
were deliberately preserved — only the model blobs under `hub/` were removed.

> Two consequences to expect: IsaacSim will re-fetch its packman dependencies on
> next launch, and the 59 GB Nemotron-3-Nano-30B checkpoint is gone from cache
> (re-downloadable; the `nemotron` conda env is untouched).

`SLM_ROOT=/home/michael/slm-125m` — created, on `/`, outside the git repo.
With ~52 GB used by this build, ~68 GB of headroom remains, which is enough to
train a follow-up 350M model on the same corpus without another cleanup.

**Keep checking:** run `df -h /` between every phase. 120 GB is comfortable, not
infinite, and a runaway stream can still fill it.

**If you ever need to shrink the build**, set `TOKEN_BUDGET_B = 2.5` in
`config.py` and scale the per-source budgets proportionally: 2.5B tokens is
~19 GB total and still **20 tokens/parameter**, exactly Chinchilla-optimal for
125M. You'd lose the extra-data regime, not correctness. Not needed today.

### 0b. Environment

System Python is 3.14, which has no PyTorch wheels. Pin 3.12:

```bash
cd /home/michael/Desktop/kasa/llm/llm_from_scratch_local
uv venv --python 3.12 .venv
source .venv/bin/activate

# CUDA 12.8 driver → cu128 wheels
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers tokenizers datasets safetensors accelerate \
               numpy tqdm langdetect datasketch zstandard python-dotenv \
               fastapi "uvicorn[standard]"
```

Verify the GPUs are actually visible to torch before doing anything else:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), \
  torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# expect: 2.x.y  4  NVIDIA RTX A6000  True
```

`nvcc` is absent — that's fine. PyTorch wheels ship their own CUDA runtime. You
only need the toolkit if you compile `flash-attn` from source, and you don't
have to: **PyTorch's built-in `F.scaled_dot_product_attention` dispatches to the
FlashAttention-2 kernel on sm_86**. Use `attn_implementation="sdpa"`.

### 0c. `config.py`

Single source of truth. Sketch:

```python
from pathlib import Path
import os

SLM_ROOT = Path(os.environ.get("SLM_ROOT", "/home/michael/slm-125m"))
CLEAN_DIR, TOKENIZER_DIR = SLM_ROOT / "clean", SLM_ROOT / "tokenizer"
TOKENS_DIR, CKPT_DIR = SLM_ROOT / "tokens", SLM_ROOT / "checkpoints"

# --- data mix -------------------------------------------------------------
TOKEN_BUDGET_B = 10.0            # drop to 2.5 only if disk gets tight
SOURCES = {
    "case-law":    dict(hf_id="HFforLegal/case-law",       split="us",
                        field="document", weight=0.70),
    "sec":         dict(hf_id="PleIAs/SEC",                split="train",
                        field="text",     weight=0.20),
    "fineweb-edu": dict(hf_id="HuggingFaceFW/fineweb-edu", split="sample-10BT",
                        field="text",     weight=0.10),
}

# --- tokenizer ------------------------------------------------------------
VOCAB_SIZE = 16_384
SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
            "<|user|>", "<|assistant|>", "<|system|>"]

# --- model (maps 1:1 to transformers.LlamaConfig) -------------------------
SEQ_LEN, N_LAYER, D_MODEL, N_HEAD = 1024, 12, 768, 12
D_FFN, TIE_EMBEDDINGS = 3072, True

# --- training (tuned for 4× A6000) ----------------------------------------
MICRO_BATCH, GRAD_ACCUM, N_GPU = 32, 4, 4      # 32*4*4*1024 = 524_288 tok/step
TOKENS_PER_STEP = MICRO_BATCH * GRAD_ACCUM * N_GPU * SEQ_LEN

EPOCHS = 5                                      # 5 passes over the 10B corpus
STEPS_PER_EPOCH = int(TOKEN_BUDGET_B * 1e9) // TOKENS_PER_STEP   # ~19_073
MAX_STEPS = STEPS_PER_EPOCH * EPOCHS                             # ~95_365

# The cosine schedule MUST span all 5 epochs, not one. See Phase 5.
LR_PEAK, LR_MIN, WARMUP_STEPS = 6e-4, 6e-5, 500
LR_DECAY_STEPS = MAX_STEPS                      # decay to LR_MIN at the very end
WEIGHT_DECAY, BETAS, GRAD_CLIP = 0.1, (0.9, 0.95), 1.0

# --- checkpointing / resumption -------------------------------------------
CKPT_EVERY_STEPS = 500                          # ~30 min of wall clock
EVAL_EVERY_STEPS = 500
KEEP_LAST_N_CKPTS = 3
EARLY_STOP_PATIENCE = 5                         # evals of rising val ppl → stop
RESHUFFLE_EACH_EPOCH = True                     # new window order per epoch
DATA_SEED = 1337                                # epoch seed = DATA_SEED + epoch

NPROC_CPU = 44                                  # of 48 — leave the desktop alive
```

### 0d. Smoke test

Stream **10 documents** from each source, run them through the cleaner, print
before/after. Confirms network, field extraction, and the filter chain before
any real run.

**Deliverable:** `docs/00-setup.md` + `df -h $SLM_ROOT` showing real headroom +
a passing 10-doc smoke test.

---

## Phase 1 — Stream + clean  (CPU, **33 min actual**, network-bound)

**Stream, don't hoard.** HuggingFace `datasets` with `streaming=True`: pull
documents one at a time, clean on the fly, **stop each source at its token
budget.** The multi-TB raw datasets are never materialized.

| Source | HF id | Weight | Field | Budget (10B) |
|---|---|---|---|---|
| case-law | `HFforLegal/case-law` (split `us`) | 0.70 | `document` | 7B |
| sec | `PleIAs/SEC` | 0.20 | `text` | 2B |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 0.10 | `text` | 1B |

> **Superseded by the actual run.** case-law holds only 541,371 docs (9.1 GB)
> and exhausted at **2.251B**, not 7B. Final corpus is **5.25B** at 43 % legal /
> 38 % financial / 19 % web; `config.TOKEN_BUDGET_B = 5.25`. Streaming took
> **33 min**, not 8-24 h. See [docs/01-data.md](docs/01-data.md).

At ~4 chars/token the achieved 5.25B tokens is ~21 GB of clean text, 6.0 GB
gzipped on disk. Phase 2 then removes duplicates and benchmark contamination,
leaving **4.124B** tokens = 164 tokens/parameter across 5 epochs (well past
Chinchilla's ~20).

**The cleaning pipeline** (fixed, rule-based, deterministic). Per document,
cheapest check first; a drop ends the chain, and every drop is counted by reason:

1. `filter_lines` — drop lines <40 chars or >30 % non-alphanumeric; collapse whitespace.
2. `strip_boilerplate` — delete whole lines matching known regexes (FORM 10-K,
   `Page N of M`, SEC headers, `/s/` signatures, Table of Contents, All rights reserved).
3. length gate — drop the doc if <600 chars survive (`too_short`).
4. `is_repetitive` — drop if the top-10 4-grams cover >50 % of all 4-grams (`repetitive`).
5. `is_english` — `langdetect` on the first 5k chars; ASCII-ratio fallback (`non_english`).
6. optional strict OCR pass — drop docs where >3 % of words look like OCR errors.

**Local specifics:**

- **Run one process per source** (3 total), each with an internal
  `multiprocessing.Pool` over the cleaning function. The filters are pure
  functions of a string, so this parallelizes with zero coordination. The
  network, not the 48 threads, is the bottleneck.
- **Write `.txt.gz`, one document per line**, ~256 MB uncompressed per shard.
  Gzip level 6 costs almost nothing at these speeds and turns 40 GB into ~14 GB.
  Every downstream reader opens with `gzip.open`.
- **Point the HF cache at the big disk**, or streaming scratch will silently
  fill `/`:
  ```bash
  export HF_HOME=$SLM_ROOT/.hf_cache
  ```
- **Run it in tmux**, because this is the phase most likely to take a full night:
  ```bash
  tmux new -s clean
  python -m slm.clean 2>&1 | tee $SLM_ROOT/logs/01-clean.log
  ```
- **Checkpoint progress per source** (documents consumed, tokens kept) to a JSON
  file, so an interrupted stream resumes instead of restarting.

**Reusability note (important):** this cleaned corpus is the durable asset.
Future bigger models **reuse** it; we only ever stream *additional* data for a
model that needs more, appending new cleaned shards. We never re-download or
re-clean what's already here — which is why it lives at `$SLM_ROOT` and not in
a temp directory.

**Deliverable:** `docs/01-data.md` + a drop-count report per source (kept,
dropped-by-reason, tokens estimated).

---

## Phase 2 — Dedup + contamination strip  (CPU, **15 min actual**)

1. **Near-dedup** the dominant source (case-law) with **MinHash-LSH** — 5-word
   shingles → 64-num signature → LSH buckets, Jaccard 0.8. Drop near-duplicates;
   small sources pass through. Exact blake2b hashing is the fallback.
2. **Strip eval contamination** — drop docs resembling the benchmark sets
   (LexGLUE / CaseHOLD) so held-out evaluation stays honest.

> **Actual result:** 1,421,590 → 1,299,345 docs (91.4 %), 5.25B → **4.124B**
> tokens, in 15 min. case-law showed 154 exact vs **11,756 near-duplicates** — a
> 76× gap that vindicates scoping MinHash to it, and that exact hashing alone
> would have missed entirely.
>
> **Decontamination n-gram hashing must be 64-bit.** At 32 bits against a
> 27.5M-gram benchmark index, every document collides ~28 times by chance and
> 83 % of the corpus is falsely flagged — silently, with no crash. See
> [docs/02-dedup.md](docs/02-dedup.md).

**Local specific:** with 251 GB of RAM, the entire MinHash signature table for
~10M documents (64 × uint32 = 256 B/doc ≈ 2.5 GB) lives comfortably in memory.
No sharded/on-disk LSH needed — build the index in one process, fan the
signature *computation* out over `Pool(44)`.

**Deliverable:** `docs/02-dedup.md` + kept/dropped counts.

---

## Phase 3 — Train the tokenizer  (CPU, 20–40 min)

A **fresh 16,384-token byte-level BPE** trained on the cleaned corpus.

- Byte-level → **no out-of-vocabulary token, ever**; robust to OCR garble and
  stray unicode.
- 16K is deliberately small: at 125M the embedding table is ~10 % of the model,
  so a small vocab leaves more budget for the transformer layers.
- Reserve the special tokens from `config.SPECIALS` at train time.
- Save as a HuggingFace `PreTrainedTokenizerFast` at `$SLM_ROOT/tokenizer/`.

**Local specifics:** the `tokenizers` trainer is Rust-threaded — it will use all
48 threads on its own. Feed it a **sample**, not the whole corpus: ~2–5 GB of
text drawn proportionally from the three sources is enough to fit a stable 16K
merge table, and keeps this phase under an hour. Use a Python generator so the
sample never lands on disk.

**Gotcha (write it down now):** `llama.cpp`'s `convert_hf_to_gguf.py` rejects a
fresh byte-level BPE (`BPE pre-tokenizer was not recognized`) and the hash
changes on every retrain. If you later export GGUF for CPU inference, patch the
converter to map our unknown byte-level BPE → the `gpt-2` pre-tokenizer. Keep
that patch idempotent.

**Deliverable:** `docs/03-tokenizer.md` + a round-trip encode/decode sanity
check + measured compression (target: ~4 chars/token on legal text).

---

## Phase 4 — Tokenize + split 99/1  (CPU, 1–2 h)

**The on-disk training format** — the concrete answer to "is there a specific
data format?":

- Each cleaned document is batch-encoded to token ids.
- An **`<|eos|>` id is appended after every document** as a separator.
- The flat id stream is **packed into fixed 1024-token windows** (the model's
  context length). No padding inside a window; EOS marks document boundaries.
- Windows are stored as **`uint16` binary shards** (`*.bin`) plus an
  `index.json` (shard names, token counts, window counts, dtype, seq_len).
  `uint16` because a 16K vocab fits in 16 bits — half the disk of `int32`.
- Tokenization is **parallel: one worker per input shard** (`Pool(44)`), then
  merge + re-index. It reads only the local cleaned corpus, never HuggingFace.

**The split:** deterministic **99 % train / 1 % val** — every 100th packed
window goes to `$SLM_ROOT/tokens/val/`, the rest to `tokens/train/`. By-window
and reproducible, so no validation window can leak into training. Result:
~9.9B train + ~100M val tokens.

**Local specific:** the training loader `np.memmap`s these shards. At 20 GB
against 251 GB of RAM, the whole corpus ends up in page cache after the first
epoch pass — **data loading effectively disappears as a cost**. Do not build a
complicated prefetch pipeline; a memmap plus a shuffled window-index array is
both simpler and faster here. Optionally pre-warm with
`cat $SLM_ROOT/tokens/train/*.bin > /dev/null` before a run.

**Deliverable:** `docs/04-tokenize.md` + final token/window counts for train and val.

---

## Phase 5 — Pretrain the 125M model  (4× A6000, 5 epochs, ~1.8 days)

### Architecture (in `config.py`, maps 1:1 to `transformers.LlamaConfig`)

| Field | Value |
|---|---|
| parameters | ~125M |
| layers | 12 |
| hidden size | 768 |
| attention heads | 12 (head dim 64), MHA (kv heads = heads) |
| MLP | SwiGLU, inner 3072 |
| normalization | RMSNorm, pre-norm |
| position embeddings | RoPE (rotary, 0 params) |
| context length | 1024 |
| vocab | 16,384 |
| embeddings | tied (input = output projection) |

### Recipe

Next-token cross-entropy, AdamW (β 0.9/0.95, wd 0.1), cosine LR with warmup,
grad-clip 1.0, **bf16 autocast** (native on Ampere — do *not* use fp16 + a loss
scaler, you don't need it), gradient checkpointing **off** (the model is small
and 48 GB is plenty). Evaluate **perplexity** on `tokens/val/` every 500 steps.
**5 epochs** over the corpus, with full resumable checkpointing (below).

### The batch math for this box

```
micro_batch 32 × seq 1024              =  32,768 tokens / GPU / fwd-bwd
× 4 GPUs                               = 131,072 tokens / step-slice
× grad_accum 4                         = 524,288 tokens / optimizer step  (~0.5M ✓)
4.124B tokens ÷ 524,288                ≈   7,865 steps / epoch
× 5 epochs                             ≈  39,325 total optimizer steps
                                       =  20.6B tokens seen, 164 tokens/parameter
```

Micro-batch 32 uses roughly 12–16 GB of the 48 GB per card. There is headroom to
push to 64 — measure it, but 0.5M tokens/step is the target *global* batch, so
raising micro-batch means lowering `grad_accum`, not changing the recipe.

### What 5 epochs means (read before launching a 4-day run)

Going from 1 → 5 epochs costs **5× the wall clock: ~44 h, about 1.8 days** on the
final 4.124B corpus.
Two things about it are worth understanding, because both change how you run it:

- **It is over-training on purpose, and that's defensible.** 20.6B tokens on 125M
  params is 164 tokens/parameter — ~8× Chinchilla-optimal. Chinchilla answers
  "best model for a fixed training budget"; you're optimizing for a *small model
  that's good at inference*, which is the same reason modern small models are
  trained far past Chinchilla. Expect real gains over the 1-epoch run.
- **Repeated data is worth less than fresh data, and epoch 5 is near the edge.**
  Published results on data-constrained training find repeated epochs stay
  close to fresh-token value up to roughly 4 passes, then decay. Epoch 5 will
  still help; epochs 8–10 mostly would not. If you later want more compute,
  streaming another 10B tokens (Phase 1 appends cleanly) beats a 6th epoch.

The practical consequence: **watch val perplexity, and let it decide the ending.**
With repeated data, the run can start memorizing — train loss keeps falling while
val perplexity flattens then rises. That divergence, not the step counter, is the
real stopping signal. `EARLY_STOP_PATIENCE = 5` stops after 5 consecutive evals
(~2,500 steps) of no improvement, and `best.pt` preserves the best-val weights
regardless of where the run ends up.

### Three things that must be right for multi-epoch

1. **The cosine schedule spans all 95k steps, not 19k.** This is the single most
   common multi-epoch bug: an LR that decays to `LR_MIN` at the end of epoch 1
   means epochs 2–5 train at a floor LR and waste four days. Set
   `LR_DECAY_STEPS = MAX_STEPS`.
2. **Reshuffle window order every epoch.** Seed the permutation with
   `DATA_SEED + epoch`. Re-feeding an identical order produces periodic loss
   artifacts and makes the model memorize sequence position as well as content.
3. **The val split stays fixed across all epochs.** Never reshuffle train and val
   together — the 99/1 split from Phase 4 is by-window and must remain frozen, or
   validation stops being held-out and the early-stopping signal becomes noise.

### Checkpointing and resumption

The run must survive a power blip on day 3, and must be resumable if you later
decide epoch 6 is worth it. Write to `$SLM_ROOT/checkpoints/`:

| File | When | Purpose |
|---|---|---|
| `last.pt` | every 500 steps, overwritten | **the resume point** — always the newest state |
| `step_NNNNNN.pt` | every 500 steps, rolling last 3 | rollback if loss diverges |
| `best.pt` | whenever val ppl improves | the weights you actually ship |
| `base/` (safetensors) | end of each epoch + final | HF-format artifact for Phase 6 |

`last.pt` must contain **everything needed to continue**, not just weights:

```python
torch.save({
    "model":       raw_model.state_dict(),      # unwrap DDP: model.module
    "optimizer":   optimizer.state_dict(),      # AdamW moments — omit and you restart cold
    "scheduler":   scheduler.state_dict(),
    "step":        global_step,
    "epoch":       epoch,
    "window_pos":  position_within_epoch,       # so a resume doesn't re-see data
    "best_val":    best_val_ppl,
    "torch_rng":   torch.get_rng_state(),
    "cuda_rng":    torch.cuda.get_rng_state_all(),
    "numpy_rng":   np.random.get_state(),
    "config":      asdict(cfg),                 # geometry + LR schedule, to extend later
}, tmp_path)
os.replace(tmp_path, ckpt_path)                 # atomic — a crash mid-save can't corrupt it
```

Notes that matter:

- **Save from rank 0 only**, after a `dist.barrier()`, and unwrap DDP
  (`model.module`) so the checkpoint loads without DDP too.
- **Write to `.tmp` then `os.replace`.** A 1.6 GB save takes seconds; a crash
  during a direct overwrite leaves you with a corrupt `last.pt` and no resume.
- **`--resume` restores everything**, including `window_pos`, so a resumed run
  continues the epoch where it stopped rather than re-reading data or skipping it.
- **Saving `config` in the checkpoint is what makes epoch 6+ possible.** To
  extend training later, load `last.pt`, raise `EPOCHS`, and rebuild the
  scheduler against the *new* `MAX_STEPS`. Without the stored schedule config you
  can't tell what LR the run was on and any extension is guesswork.
- **Size:** ~1.6 GB each (fp32 params + two AdamW moments), so rolling-3 +
  `last` + `best` ≈ 8 GB. Budgeted in Phase 0a.

```bash
# resume after a crash / to extend training
torchrun --standalone --nproc_per_node=4 -m slm.train --resume $SLM_ROOT/checkpoints/last.pt
```

### Launch

```bash
tmux new -s train
source .venv/bin/activate && source .env.local
torchrun --standalone --nproc_per_node=4 -m slm.train \
  2>&1 | tee $SLM_ROOT/logs/05-train.log
```

### Local specifics that matter

- **GPU 0 drives your display** (it was showing 753 MiB used, 23 % util at
  idle). In DDP every GPU waits at each all-reduce barrier, so a busy desktop on
  GPU 0 **slows all four cards**. Either train headless on all 4, or keep using
  the desktop and train on three:
  ```bash
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 -m slm.train
  ```
  Three GPUs costs ~33 % wall clock (≈59 h instead of ≈44 h for 5 epochs). Adjust
  `GRAD_ACCUM` to 5–6 to hold the global batch near 0.5M tokens.
- **No NVLink is fine at this scale.** Gradient all-reduce moves ~250 MB (125M
  params, bf16) per optimizer step over PCIe 4.0 ×16 — tens of milliseconds
  against a ~3.5 s step. Roughly 1 % overhead. If NCCL *hangs* on startup (a
  known P2P-over-PCIe failure mode on some boards), fall back with
  `NCCL_P2P_DISABLE=1` and re-measure; expect a small, survivable slowdown.
- **Use `attn_implementation="sdpa"`** so attention hits the FlashAttention-2
  kernel. Enable TF32 for the non-attention matmuls:
  `torch.set_float32_matmul_precision("high")`.
- **`torch.compile` the model.** On Ampere it is typically worth 20–35 % on a
  model this shape. Compile once, outside the step loop; expect a 1–3 minute
  first-step stall and don't mistake it for a hang.
- **Power and heat.** 4 × 300 W GPUs plus a 280 W Threadripper is ~1.6 kW
  sustained for a day. Confirm your PSU and room cooling can hold that. If
  temps climb past ~83 °C or the breaker is marginal, cap power — it costs far
  less throughput than it sounds:
  ```bash
  sudo nvidia-smi -pl 250   # ~17% less power, ~5% less throughput
  ```
- **Checkpoint every ~500 steps** (≈30 min at ~3.5 s/step) — see the
  checkpointing section above. This is a desktop, not a datacenter, and the run
  is now 4.5 days long: a power blip should cost you half an hour, not the run.
- **The cost cap becomes a step cap.** Set `MAX_STEPS` and a wall-clock
  guard in the trainer, log tokens/s and estimated time-to-finish every 50
  steps, and hold to the same rule as the cloud plan: **report at real
  milestones only.**

**Throughput sanity check:** at ~0.86 GFLOP/token (6N + attention) and ~120
TFLOP/s usable across 4 cards, expect **~120–150k tokens/s** aggregate and
**~3.5 s per optimizer step**. If you see less than half that, something is
wrong — check that bf16, SDPA, and `torch.compile` are all actually active
before you let a 24-hour run proceed.

**Honest framing:** at 125M the headline metric is **held-out validation
perplexity**, not MMLU (near-random at this size). The base model is a
**completer**, not a chat model — give it a passage prefix and it continues it.

**Headroom note:** 192 GB of VRAM comfortably trains 350M–1B parameters with
this exact pipeline. Ship the 125M first; treat the bigger run as a follow-up
that reuses the Phase 1–4 artifacts unchanged.

**Deliverable:** `docs/05-pretrain.md` + a loss/perplexity curve + real sample
generations from a legal/financial prefix.

---

## Phase 6 — Local inference

You asked to do inference on this computer, so the primary target is a local
server. Pushing to HuggingFace becomes optional publication, not deployment.

### The artifact

A standard HuggingFace model directory — the same shape as any Llama checkpoint:

```
$SLM_ROOT/checkpoints/base/
├── config.json              # LlamaConfig: 12L/768d/12h, vocab 16384, ctx 1024, RoPE
├── model.safetensors        # the weights, bf16, ~250 MB
├── generation_config.json   # eos/bos/pad ids, default sampling
├── tokenizer.json           # the fast tokenizer (merges + vocab)
├── tokenizer_config.json
└── special_tokens_map.json  # <|bos|> <|eos|> <|pad|> <|unk|>
```

Directly loadable with `AutoModelForCausalLM.from_pretrained(...)` /
`AutoTokenizer.from_pretrained(...)`. It is a **base completion model**: prompt
= `<|bos|>` + your text, and it continues.

### Serving it here

At ~250 MB the model is trivial to serve — it fits in a corner of one A6000 with
room for a very large KV cache, and it runs acceptably on **CPU** too, which
means you can serve it without tying up a training GPU.

```bash
# pin inference to the display GPU, leaving 1-3 free for the next training run
CUDA_VISIBLE_DEVICES=0 uvicorn slm.serve:app --host 127.0.0.1 --port 8000
```

`slm/serve.py` — FastAPI, model loaded once at startup, exposing:

- `POST /generate {prompt, max_new_tokens, temperature, top_p, top_k}` → `{generated}`
- `GET  /health` → model path, device, dtype, step count it was trained to
- a streaming variant via `TextIteratorStreamer` + SSE, so the UI fills in
  token by token instead of blocking

Serve a single static `index.html` from the same FastAPI app — prompt box,
sampling controls, streamed completion, and a short "what this is" panel (it
speaks the domain register, it is a base completer, the honest metric is
perplexity). One origin, so **no CORS and no cold starts** — the two problems
the HF Space plan had to design around simply don't exist locally.

**Gotcha to bake in (unchanged, and it will bite):** a base model under greedy
decoding can emit EOS after ~1 token. Enforce a `min_new_tokens`, suppress rare
non-ASCII vocab tokens, and set a `top_k` to avoid unicode garbage.

Optional, if you want it always-on: a `systemd --user` unit so the server
survives logout.

```ini
# ~/.config/systemd/user/slm.service
[Service]
WorkingDirectory=/home/michael/Desktop/kasa/llm/llm_from_scratch_local
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/michael/Desktop/kasa/llm/llm_from_scratch_local/.venv/bin/uvicorn \
          slm.serve:app --host 127.0.0.1 --port 8000
Restart=on-failure
[Install]
WantedBy=default.target
```

### Optional extras

- **Publish the weights** to `your-user/slm-125m-base` with the write token —
  the canonical home of the artifact, and a backup of a run you can't cheaply repeat.
- **GGUF export** (Q4/Q8) via `llama.cpp` for laptop inference — needs the
  tokenizer patch from Phase 3.
- **Expose it beyond localhost** only deliberately: bind `0.0.0.0` for the LAN,
  or front it with a Cloudflare tunnel. There is no auth on this server, so do
  not put it on the open internet as written.

**Deliverable:** `docs/06-deploy.md` + a working `http://127.0.0.1:8000` +
measured tokens/s and latency for a 256-token completion.

---

## Quick reference: what format is the data at each step?

| Step | On-disk format |
|---|---|
| Raw (HF) | streamed records (parquet rows), never fully stored |
| Cleaned corpus | UTF-8 **`.txt.gz`**, one document per line, per-source folders |
| Tokenizer | HF `PreTrainedTokenizerFast` (`tokenizer.json` + configs) |
| Tokenized data | **`uint16` `.bin` shards**, EOS-separated, packed into 1024-token windows, `index.json` |
| Train/val split | same `.bin` format under `tokens/train` (99 %) and `tokens/val` (1 %) |
| Pretrained model | HF **`safetensors`** + `config.json` (LlamaConfig) + tokenizer files |
| Optional export | quantized **GGUF** for llama.cpp / CPU inference |

---

## Budget

| Phase | Hardware | Wall clock |
|---|---|---|
| 0 Setup + smoke test | — | ~1 h |
| 1 Stream + clean | 48T CPU, network-bound | **33 min actual** |
| 2 Dedup + decontaminate | 48T CPU | **15 min actual** |
| 3 Tokenizer | 48T CPU | 20–40 min |
| 4 Tokenize + pack | 48T CPU | 1–2 h |
| 5 Pretrain (5 epochs, 20.6B tokens) | 4× A6000 | ~44 h (~1.8 days) |
| 6 Local serving | 1× A6000 or CPU | ~1 h |

**Total ≈ 2.5–3 days wall clock**, most of it unattended in `tmux`. Marginal cost
is electricity — roughly **70 kWh, \$9–14** — against well over \$40 for the
equivalent 20.6B-token run on rented H100s. The
real difference isn't the money, it's that the corpus and the checkpoints stay
on your disk and the next run costs nothing to start.

---

## Known gotchas on this machine

1. **Disk was the near-miss.** Now 120 GB free after a cache purge, but this
   build eats ~52 GB of it. `df -h /` between every phase. **Never** write to
   `/dev/nvme0n1p4` — it's a live Ubuntu 18.04 install (Phase 0a).
2. **System Python is 3.14** — no torch wheels. Use the 3.12 `uv` venv, always.
3. **`HF_HOME` defaults to `~/.cache/huggingface`.** It's already redirected to
   `$SLM_ROOT/.hf_cache` in `.env.local` — keep it that way, or streaming
   scratch silently refills the space we just cleared.
4. **GPU 0 has the display attached.** It is a DDP straggler whenever you use
   the desktop. Train headless, or drop to `CUDA_VISIBLE_DEVICES=1,2,3`.
5. **No NVLink.** Expected and fine here; keep `NCCL_P2P_DISABLE=1` in your back
   pocket if collectives hang at init.
6. **No `nvcc`.** Don't try to build `flash-attn` from source — use torch SDPA.
7. **1.6 kW sustained.** Verify PSU headroom and cooling before a 24-hour run;
   `nvidia-smi -pl 250` is cheap insurance.
8. **Ampere = bf16, not fp8.** Any recipe you copy from an H100 build that
   mentions fp8 or transformer-engine does not apply.

---

## One-line summary for the agent

Mount real disk → stream three HF datasets → clean with a fixed rule chain →
dedup + decontaminate → train a 16K byte-level BPE → pack into `uint16`
1024-token windows split 99/1 → pretrain a 125M Llama-style model
(RoPE/SwiGLU/RMSNorm, tied embeddings) with `torchrun` DDP across 4× A6000 in
bf16 → serve it from a local FastAPI endpoint on `127.0.0.1:8000`. Lean,
per-phase docs, tmux + logs for every long run, `df -h` between phases.
