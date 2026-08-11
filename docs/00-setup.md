# Phase 0 — Setup + smoke test

**Goal:** prove the network, the dataset field extraction, and the cleaning chain
all work *before* any long run. Nothing here is expensive or irreversible.

## What runs

| Step | What | Cost |
|---|---|---|
| 0a | Storage — resolved, see [SLM_BUILD_GUIDE.md](../SLM_BUILD_GUIDE.md) Phase 0a | done |
| 0b | `uv` venv on Python 3.12 + torch cu128 + deps | ~5 min, ~3 GB |
| 0c | `slm/config.py` — the single source of truth | — |
| 0d | Smoke test: 10 docs per source through the cleaner | ~1 min |

## Why these choices

- **Python 3.12, not the system 3.14.** No PyTorch wheels exist for 3.14.
- **cu128 wheels** to match driver 570.211.01 / CUDA 12.8.
- **No `flash-attn` package.** Torch's `scaled_dot_product_attention` already
  dispatches to the FlashAttention-2 kernel on sm_86, and building `flash-attn`
  from source needs an `nvcc` this box doesn't have.

## The cleaning chain being smoke-tested

Cheapest check first; the first failure ends the chain and is counted by reason.

| # | Filter | Drop reason |
|---|---|---|
| 1 | `filter_lines` — drop lines <40 chars or >30 % non-alphanumeric | `no_lines_left` |
| 2 | `strip_boilerplate` — FORM 10-K, `Page N of M`, `/s/` sigs, TOC, ©️ lines | — |
| 3 | length gate — <600 surviving chars | `too_short` |
| 4 | `is_repetitive` — top-10 4-grams cover >50 % of all 4-grams | `repetitive` |
| 5 | `is_english` — `langdetect` on first 5k chars, ASCII-ratio fallback | `non_english` |
| 6 | `looks_like_ocr_garbage` — >3 % of words look like OCR errors | `ocr_garbage` |

Every filter is a **pure function** `str -> bool` or `str -> str`. No I/O, no
mutation, so they are trivially testable and safe to run in a `Pool(44)`.

## Success criteria

1. `torch.cuda.device_count() == 4`, `is_bf16_supported() == True`.
2. All three HF sources stream without auth and yield their expected text field.
3. The cleaner produces a before/after for each source and a drop-reason tally.
4. `df -h /` still shows >100 GB free (the smoke test writes nothing durable).

## What Phase 0 deliberately does *not* do

No writes to `$SLM_ROOT/clean/`. The smoke test streams into memory, prints, and
exits. Phase 1 is the first phase that produces durable artifacts.

---

# RESULTS — **PASS** (2026-08-11)

## Environment

```
python  3.12.13   (uv venv, system 3.14 bypassed as planned)
torch   2.11.0+cu128
GPUs    4× NVIDIA RTX A6000, 50.9 GB each, sm_86
bf16    supported
```

## Derived config (verified against the guide's targets)

```
params (exact)  126M          guide target ~125M          ✓
embed share     10.0%         guide claim "~10%"          ✓
tokens/step     524,288       target ~0.5M                ✓
steps/epoch     19,073
max steps       95,365        (5 epochs)
tokens seen     50.0B         397 tok/param, guide says ~400 ✓
```

The first `summary()` reported 98M because the estimate undercounted SwiGLU
(3 matrices of `d_model × d_ffn`, not 12·d²). `config.n_parameters()` now
computes it exactly: `vocab·d + n_layer·(4d² + 3·d·d_ffn)`.

## Cleaning chain — synthetic cases

| Input | Result |
|---|---|
| varied legal prose (5 distinct sentences) | `kept`, 845 chars |
| one sentence repeated 12× | `repetitive` |
| 59-char doc | `too_short` |
| only boilerplate (FORM 10-K / TOC / `/s/`) | `no_lines_left` |
| empty string | `empty` |

## Cleaning chain — live data, 10 docs per source

| Source | Streamed | Kept | Drops | Sample reduction |
|---|---|---|---|---|
| case-law | 10 | 9 | 1 `too_short` | 23,769 → 20,667 chars (−13 %) |
| sec | 10 | 10 | — | 120,931 → 91,524 chars (−24 %) |
| fineweb-edu | 10 | 10 | — | 3,665 → 3,644 chars (−0.6 %) |

**29/30 kept.** The reductions are the right shape per source: SEC sheds the most
(tilde-rule ITEM headers, signature blocks), case-law sheds court letterhead
(`STATE OF ALABAMA -- JUDICIAL DEPARTMENT`, docket numbers — all short lines),
and fineweb-edu is already clean so only its title line goes. Nothing suggests
the chain is over-firing.

## Findings that changed the plan

1. **An HF token helps *reading*, not just pushing.** The Hub warned:
   *"You are sending unauthenticated requests… set a HF_TOKEN to enable higher
   rate limits and faster downloads."* Phase 1 streams for 8–24 h straight, so
   anonymous rate limits are a real throttle risk. Set `HUGGINGFACE_TOKEN` in
   `.env.local` **before Phase 1**, not at Phase 6 as originally written.
2. **`HF_HOME` redirect confirmed working** — the dataset cache materialised at
   `$SLM_ROOT/.hf_cache/hub/`, leaving `~/.cache` alone.
3. **`fineweb-edu` uses `config_name`, not a split name.** It loads as
   `load_dataset(id, "sample-10BT", split="train")`. `Source.config_name` exists
   for exactly this; `case-law` genuinely does use `split="us"`.

## Disk

```
before venv   120 GB free
after  venv   113 GB free   (~7 GB: torch + deps)
$SLM_ROOT     212 KB        (dirs + dataset metadata only)
```

## Ready for Phase 1

Set the HF token first, then stream + clean under `tmux`.
