# Phase 5 — Pretrain the 125M model

**Goal:** train a 126M-parameter Llama-style decoder on 3.536B tokens for
5 epochs (17.7B tokens seen, 140 tokens/parameter) across 4× RTX A6000.

```bash
# throughput check first - 60 steps, verifies bf16/SDPA/compile before 38 hours
torchrun --standalone --nproc_per_node=4 -m slm.train --bench 60

# the real run, detached
setsid nohup bash -c '... torchrun --standalone --nproc_per_node=4 -m slm.train' &

# resume after a crash, or to extend past epoch 5
torchrun --standalone --nproc_per_node=4 -m slm.train --resume $SLM_ROOT/checkpoints/last.pt
```

## Architecture (`transformers.LlamaConfig`)

| Field | Value |
|---|---|
| parameters | 126M (embeddings 12.6M = 10.0 %) |
| layers / hidden / heads | 12 / 768 / 12 (head dim 64, MHA) |
| MLP | SwiGLU, inner 3072 |
| norm | RMSNorm, pre-norm, eps 1e-5 |
| positions | RoPE, theta 10,000 |
| context | 1024 |
| vocab | 16,384 |
| embeddings | tied |
| attention | `sdpa` → FlashAttention-2 kernel on sm_86 |

## Schedule (exact, from `tokens/train/index.json`)

```
3,535,785,984 tokens / 524,288 per step = 6,743 steps/epoch
x 5 epochs                              = 33,715 steps
micro_batch 32 x 4 GPUs x accum 4 x 1024 = 524,288 tokens/step
```

AdamW β(0.9, 0.95), wd 0.1 (2-D params only), grad-clip 1.0, bf16 autocast,
LR 6e-4 → 6e-5 cosine over **all 33,715 steps** after 500 warmup steps.

## What this run does differently from the guide

1. **Cosine spans all 5 epochs**, not one. Decaying to `LR_MIN` at the end of
   epoch 1 would leave four days of training at the LR floor.
2. **Per-epoch reshuffle** with seed `DATA_SEED + epoch`. Re-feeding an identical
   order makes the model learn sequence position alongside content.
3. **Per-source validation perplexity**, with early stopping driven by
   **case-law + fineweb-edu only**. Phase 4 measured 19.3 % of SEC val n-grams
   already present in SEC train (corpus redundancy, not split leakage), so SEC's
   perplexity is optimistic and unfit to detect memorisation.
4. **Throughput is verified before the long run.** A 2× slowdown discovered at
   hour 20 costs a day; `--bench` costs two minutes.

## Checkpointing

| File | Cadence | Purpose |
|---|---|---|
| `last.pt` | every 500 steps | the resume point |
| `step_NNNNNN.pt` | every 500 steps, keep 3 | rollback |
| `best.pt` | on clean-val improvement | the weights we ship |
| `base/` safetensors | end of each epoch + final | Phase 6 artifact |

`last.pt` carries model + optimizer + scheduler + step + epoch + RNG states +
the resolved config, written to `.tmp` then `os.replace` so a crash mid-save
cannot corrupt the only resume point. Saving `config` is what makes extending
to epoch 6+ possible later.

## Expected

~130–150k tokens/s aggregate, ~3.5 s/step, **~38 h**, ~61 kWh. If throughput is
below half that, something is off — check bf16, SDPA and compile are live before
letting it run.

---

# RESULTS

## Throughput check (`--bench 60`, 2026-08-12)

```
BENCH: 60 steps in 132.0s (2.20 s/step, 238k tok/s)
       full run projection: 20.6 h (0.9 days), 33 kWh
       peak VRAM/GPU: 20.9 GB of 48
```

238k tok/s aggregate against the 130–150k the guide expected — bf16 + SDPA +
`torch.compile` are all live, and PCIe all-reduce is not the bottleneck at 126M
params. Loss fell 9.85 → 7.44 over the 60 steps. Peak VRAM leaves headroom;
micro-batch 32 was not the binding constraint.

Exact schedule read from `tokens/train/index.json`: 3,452,916 train windows,
34,958 val windows, **6,743 steps/epoch × 5 = 33,715 steps**, 524,288 tok/step.

## Full run

Launched 2026-08-12 14:43, detached (`setsid nohup`), logging to
`$SLM_ROOT/logs/05-pretrain.log`. Finished 2026-08-13 10:44.

```
done: 33,715 steps in 20.0 h, best clean val ppl 12.750
```

All 5 epochs ran to completion — **early stopping never fired**, clean val
improved monotonically to the last eval. 17.7B tokens seen, 140 tokens/param.
Steady state ~2.0 s/step at ~261k tok/s; the extra 1.5 h over the bench
projection is the 67 eval + checkpoint pauses (~55 s each).

### Validation perplexity, per source

| Step | epoch | case-law | fineweb-edu | sec | clean |
|---|---|---|---|---|---|
| 500 | 0.07 | 34.07 | 91.35 | 17.99 | 55.79 |
| 7,000 | 1.04 | 10.00 | 24.47 | 5.83 | 15.64 |
| 13,500 | 2.00 | 9.19 | 22.11 | 5.31 | 14.25 |
| 20,000 | 2.97 | 8.73 | 20.86 | 5.00 | 13.50 |
| 27,000 | 4.00 | 8.41 | 19.98 | 4.74 | 12.96 |
| 33,500 | 4.97 | **8.28** | **19.63** | **4.62** | **12.750** |

Train loss 9.85 → ~2.05. The shape is the expected one: epoch 1 buys most of
it (55.8 → 15.6), and epochs 2–5 return a further 18 % for four times the
compute. SEC's 4.62 stays the most optimistic number of the three for the
corpus-redundancy reason in Phase 4 — it is reported, not trusted.

Gains were still real but small at the end (12.757 → 12.750 over the last 500
steps), so the LR floor, not data exhaustion, is what the run ended on. A 6th
epoch is not the move; streaming more tokens is (Phase 1 appends cleanly).

### Artifacts

| Path | Size | Notes |
|---|---|---|
| `checkpoints/base/` | 503 MB | HF safetensors + tokenizer, final step — the Phase 6 input |
| `checkpoints/best.pt` | 1.5 GB | step 33,500, clean val 12.750 |
| `checkpoints/last.pt` | 1.5 GB | step 33,715, resume point |
| `checkpoints/step_0335*.pt` | 1.5 GB × 3 | rollback, last 3 kept |

`best.pt` is step 33,500 because the final step (33,715) is not a multiple of
`EVAL_EVERY_STEPS`, so no eval ran on it. `base/` holds the step-33,715 weights,
215 steps newer than `best.pt` and 0.3 % of one epoch apart — untested against
val but on a monotone curve. Ship `base/`.

Disk after the run: 86 G free on `/`.

### Sanity generations (bf16, temp 0.8, top-p 0.95)

```
The Court held that the evidence of the plaintiff's "intent" to deceive and
defraud his wife was insufficient to support a finding of fraud against the
plaintiff and in favor of the defendant on the issue of punitive damages.

Item 1A. Risk Factors - Risks Associated with the Market for the Company's
Common Stock and Related Stockholder Matters - Nasdaq Delisting and Potential
Dilution. The Company's Common Stock trades on the Nasdaq National Market...

Photosynthesis is the process that leads to photosynthesis. The process is
described by Photosynthesis. Photosynthesis is used to make energy, which is
then converted into chemical energy.
```

Legal and filing register are both convincing — unsurprising at 76 % of the
mix. The fineweb-edu completion degenerates into definitional circling, which
is what a 126M base model with no instruction tuning does on open-domain
prompts, and matches fineweb-edu's 19.63 being 2.4× case-law's perplexity.
