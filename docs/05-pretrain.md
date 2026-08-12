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

Launched 2026-08-12, detached (`setsid nohup`), logging to
`$SLM_ROOT/logs/05-pretrain.log`. Settled at ~2.05 s/step → **ETA ~19.2 h**.

_(final loss, per-source val perplexity and wall clock filled in after the run)_
