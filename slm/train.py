"""Phase 5: pretrain the 125M Llama-style model on 4x A6000 with DDP.

    torchrun --standalone --nproc_per_node=4 -m slm.train --bench 60
    torchrun --standalone --nproc_per_node=4 -m slm.train
    torchrun --standalone --nproc_per_node=4 -m slm.train --resume <ckpt>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import config as C
from .data import EpochSampler, WindowStore


# --------------------------------------------------------------------- setup --


@dataclass
class Ctx:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_master: bool
    ddp: bool


def setup() -> Ctx:
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world = dist.get_world_size()
    else:
        rank = local_rank = 0
        world = 1
    torch.cuda.set_device(local_rank)
    torch.manual_seed(C.SEED + rank)
    np.random.seed(C.SEED + rank)
    torch.set_float32_matmul_precision("high")  # TF32 for non-attention matmuls
    return Ctx(rank, local_rank, world, torch.device(f"cuda:{local_rank}"),
               rank == 0, ddp)


def log(ctx: Ctx, msg: str) -> None:
    if ctx.is_master:
        print(msg, flush=True)


def build_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=C.VOCAB_SIZE,
        hidden_size=C.D_MODEL,
        intermediate_size=C.D_FFN,
        num_hidden_layers=C.N_LAYER,
        num_attention_heads=C.N_HEAD,
        num_key_value_heads=C.N_KV_HEAD,
        max_position_embeddings=C.SEQ_LEN,
        rms_norm_eps=C.RMS_NORM_EPS,
        rope_theta=C.ROPE_THETA,
        tie_word_embeddings=C.TIE_EMBEDDINGS,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        attn_implementation="sdpa",  # FlashAttention-2 kernel on sm_86
        use_cache=False,
    )
    return LlamaForCausalLM(cfg), cfg


def unwrap(model):
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def make_optimizer(model):
    """Weight decay on matrices only; never on norms, biases, or embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": C.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=C.LR_PEAK, betas=C.BETAS, fused=True)


def lr_at(step: int) -> float:
    """Linear warmup, then cosine over ALL epochs down to LR_MIN."""
    if step < C.WARMUP_STEPS:
        return C.LR_PEAK * (step + 1) / C.WARMUP_STEPS
    prog = (step - C.WARMUP_STEPS) / max(1, C.LR_DECAY_STEPS - C.WARMUP_STEPS)
    prog = min(1.0, prog)
    return C.LR_MIN + 0.5 * (C.LR_PEAK - C.LR_MIN) * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------- evaluation --


@torch.no_grad()
def evaluate(model, store: WindowStore, ctx: Ctx, max_batches: int = 0) -> dict:
    """Per-source validation loss. Rank 0 computes it; cheap at 35M tokens."""
    model.eval()
    results = {}
    n = max_batches or C.EVAL_BATCHES
    for src in store.source_names():
        idx = store.indices_for_sources((src,))
        if idx.size == 0:
            continue
        rng = np.random.default_rng(C.SEED)
        pick = rng.permutation(idx.size)[: n * C.MICRO_BATCH]
        total, count = 0.0, 0
        for i in range(0, pick.size, C.MICRO_BATCH):
            batch = store.gather(idx[pick[i : i + C.MICRO_BATCH]]).to(
                ctx.device, non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=batch, labels=batch)
            total += out.loss.float().item()
            count += 1
        results[src] = total / max(1, count)
    model.train()
    return results


def clean_val_loss(per_source: dict) -> float:
    """Early-stopping signal: clean sources only (see config.EARLY_STOP_SOURCES)."""
    vals = [v for k, v in per_source.items() if k in C.EARLY_STOP_SOURCES]
    return sum(vals) / max(1, len(vals))


# --------------------------------------------------------------- checkpoints --


def save_ckpt(path: Path, model, opt, step: int, epoch: int, best: float,
              extra: dict) -> None:
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val": best,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "config": {k: v for k, v in vars(C).items()
                   if k.isupper() and isinstance(v, (int, float, str, bool, tuple))},
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save cannot corrupt the resume point


def save_hf(model, out_dir: Path) -> None:
    m = unwrap(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir, safe_serialization=True)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(str(C.TOKENIZER_DIR)).save_pretrained(out_dir)


def prune_ckpts(ckpt_dir: Path, keep: int) -> None:
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    for old in steps[:-keep]:
        old.unlink()


# --------------------------------------------------------------------- train --


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="run N steps and exit")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    ctx = setup()
    C.ensure_dirs()

    train_store = WindowStore.load(C.TRAIN_TOKENS_DIR)
    val_store = WindowStore.load(C.VAL_TOKENS_DIR)
    windows_per_step = C.MICRO_BATCH * C.GRAD_ACCUM * ctx.world_size
    sampler = EpochSampler(train_store.n_windows, ctx.rank, ctx.world_size,
                           windows_per_step)

    log(ctx, "=" * 78)
    log(ctx, f"train windows {train_store.n_windows:,}  "
             f"val windows {val_store.n_windows:,}")
    log(ctx, f"steps/epoch {sampler.steps_per_epoch:,}  epochs {C.EPOCHS}  "
             f"max steps {sampler.steps_per_epoch * C.EPOCHS:,}")
    log(ctx, f"tokens/step {windows_per_step * C.SEQ_LEN:,}  world {ctx.world_size}")
    log(ctx, "=" * 78)

    model, hf_cfg = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    log(ctx, f"parameters {n_params/1e6:.1f}M  bf16={torch.cuda.is_bf16_supported()}")
    model.to(ctx.device)
    model.gradient_checkpointing_disable()

    if not args.no_compile:
        log(ctx, "compiling (first step will stall 1-3 min) ...")
        model = torch.compile(model)
    if ctx.ddp:
        model = DDP(model, device_ids=[ctx.local_rank])

    opt = make_optimizer(model)

    start_step, start_epoch, best_val = 0, 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=ctx.device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step, start_epoch = ck["step"], ck["epoch"]
        best_val = ck.get("best_val", float("inf"))
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["numpy_rng"])
        log(ctx, f"resumed at step {start_step:,} epoch {start_epoch} "
                 f"best_val {best_val:.4f}")

    max_steps = sampler.steps_per_epoch * C.EPOCHS
    if args.bench:
        max_steps = args.bench

    global_step = start_step
    tokens_per_step = windows_per_step * C.SEQ_LEN
    t_start = time.time()
    t_win = time.time()
    stale_evals = 0
    stop = False

    for epoch in range(start_epoch, C.EPOCHS):
        step_in_epoch = global_step % sampler.steps_per_epoch if epoch == start_epoch else 0
        for step, _ in sampler.rank_batches(epoch, step_in_epoch):
            lr = lr_at(global_step)
            for g in opt.param_groups:
                g["lr"] = lr

            order = sampler.epoch_order(epoch)
            per_rank = windows_per_step // ctx.world_size
            chunk = order[step * windows_per_step : (step + 1) * windows_per_step]
            mine = chunk[ctx.rank * per_rank : (ctx.rank + 1) * per_rank]

            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for micro in range(C.GRAD_ACCUM):
                sl = mine[micro * C.MICRO_BATCH : (micro + 1) * C.MICRO_BATCH]
                batch = train_store.gather(sl).to(ctx.device, non_blocking=True)
                # Only sync gradients on the final micro-batch of the step.
                if ctx.ddp:
                    model.require_backward_grad_sync = micro == C.GRAD_ACCUM - 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=batch, labels=batch)
                    loss = out.loss / C.GRAD_ACCUM
                loss.backward()
                loss_acc += loss.detach().float().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            global_step += 1

            log_every = 1 if args.bench else 10
            if global_step % log_every == 0:
                dt = time.time() - t_win
                t_win = time.time()
                # Divide by the steps actually elapsed since the last log, not a
                # hardcoded 10 - otherwise every 10th line reads 10x too fast.
                per_step = dt / log_every
                tps = tokens_per_step / per_step
                eta = (max_steps - global_step) * per_step / 3600
                log(ctx, f"step {global_step:>6}/{max_steps}  loss {loss_acc:.4f}  "
                         f"lr {lr:.2e}  {tps/1e3:.0f}k tok/s  "
                         f"{per_step:.2f}s/step  eta {eta:.1f}h")

            if args.bench and global_step - start_step >= args.bench:
                total = time.time() - t_start
                n = global_step - start_step
                log(ctx, "=" * 78)
                log(ctx, f"BENCH: {n} steps in {total:.1f}s "
                         f"({total/n:.2f} s/step, "
                         f"{n*tokens_per_step/total/1e3:.0f}k tok/s)")
                full_h = sampler.steps_per_epoch * C.EPOCHS * (total/n) / 3600
                log(ctx, f"       full run projection: {full_h:.1f} h "
                         f"({full_h/24:.1f} days), {1.6*full_h:.0f} kWh")
                if ctx.is_master:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    log(ctx, f"       peak VRAM/GPU: {mem:.1f} GB of 48")
                if ctx.ddp:
                    dist.destroy_process_group()
                return 0

            if global_step % C.EVAL_EVERY_STEPS == 0:
                per_src = evaluate(model, val_store, ctx)
                clean = clean_val_loss(per_src)
                if ctx.is_master:
                    parts = "  ".join(
                        f"{k}={math.exp(min(20, v)):.2f}" for k, v in per_src.items()
                    )
                    log(ctx, f"  [eval] step {global_step}  ppl: {parts}  "
                             f"| clean={math.exp(min(20, clean)):.3f}")
                    if clean < best_val:
                        best_val = clean
                        stale_evals = 0
                        save_ckpt(C.CKPT_DIR / "best.pt", model, opt, global_step,
                                  epoch, best_val, {"val": per_src})
                    else:
                        stale_evals += 1
                        if stale_evals >= C.EARLY_STOP_PATIENCE:
                            log(ctx, f"  early stop: {stale_evals} evals without "
                                     f"clean-val improvement")
                            stop = True
                if ctx.ddp:
                    flag = torch.tensor([1.0 if stop else 0.0], device=ctx.device)
                    dist.broadcast(flag, src=0)
                    stop = bool(flag.item())

            if global_step % C.CKPT_EVERY_STEPS == 0 and ctx.is_master:
                save_ckpt(C.CKPT_DIR / "last.pt", model, opt, global_step, epoch,
                          best_val, {})
                save_ckpt(C.CKPT_DIR / f"step_{global_step:06d}.pt", model, opt,
                          global_step, epoch, best_val, {})
                prune_ckpts(C.CKPT_DIR, C.KEEP_LAST_N_CKPTS)

            if stop or global_step >= max_steps:
                break

        if ctx.is_master:
            save_hf(model, C.BASE_MODEL_DIR)
            log(ctx, f"epoch {epoch} complete -> {C.BASE_MODEL_DIR}")
        if stop or global_step >= max_steps:
            break

    if ctx.is_master:
        save_ckpt(C.CKPT_DIR / "last.pt", model, opt, global_step,
                  C.EPOCHS - 1, best_val, {})
        save_hf(model, C.BASE_MODEL_DIR)
        hrs = (time.time() - t_start) / 3600
        log(ctx, f"\ndone: {global_step:,} steps in {hrs:.1f} h, "
                 f"best clean val ppl {math.exp(min(20, best_val)):.3f}")
    if ctx.ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
