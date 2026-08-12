"""Single source of truth for the whole build.

Every other module imports from here. Nothing else defines model geometry,
data mix, paths, or training hyperparameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env.local")

# huggingface_hub and datasets authenticate off HF_TOKEN. Mirror our name onto
# it so streaming is authenticated (anonymous requests are rate-limited, and
# Phase 1 streams for hours).
if os.environ.get("HUGGINGFACE_TOKEN") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

# --------------------------------------------------------------------- paths --

SLM_ROOT = Path(os.environ.get("SLM_ROOT", "/home/michael/slm-125m"))

CLEAN_DIR = SLM_ROOT / "clean"
DEDUP_DIR = SLM_ROOT / "dedup"  # Phase 2 output; clean/ is never modified

# Phases 3 and 4 train the tokenizer and pack tokens from the DEDUPLICATED,
# decontaminated corpus. Reading CLEAN_DIR here would silently reintroduce the
# 122,245 documents Phase 2 removed.
CORPUS_DIR = DEDUP_DIR
TOKENIZER_DIR = SLM_ROOT / "tokenizer"
TOKENS_DIR = SLM_ROOT / "tokens"
TRAIN_TOKENS_DIR = TOKENS_DIR / "train"
VAL_TOKENS_DIR = TOKENS_DIR / "val"
CKPT_DIR = SLM_ROOT / "checkpoints"
BASE_MODEL_DIR = CKPT_DIR / "base"
LOG_DIR = SLM_ROOT / "logs"

# ------------------------------------------------------------------ data mix --


@dataclass(frozen=True)
class Source:
    """One streamed HuggingFace dataset."""

    hf_id: str
    field: str
    weight: float
    split: str = "train"
    config_name: str | None = None

    @property
    def token_budget(self) -> int:
        return int(TOKEN_BUDGET_B * 1e9 * self.weight)


# ACTUAL corpus, measured after Phase 2.
#   planned          10.00B   (70% legal)
#   after Phase 1     5.25B   case-law exhausted: the dataset holds only
#                             541,371 docs = 2.25B tokens, not the 7B assumed
#   after Phase 2     4.12B   -122,245 docs: 17,671 duplicates + 104,574
#                             contaminated against LexGLUE/CaseHOLD
# Weights below are the achieved mix, not the original plan.
# See docs/01-data.md and docs/02-dedup.md.
TOKEN_BUDGET_B: float = 4.124

SOURCES: dict[str, Source] = {
    "case-law": Source(
        hf_id="HFforLegal/case-law", split="us", field="document", weight=0.33
    ),
    "sec": Source(hf_id="PleIAs/SEC", split="train", field="text", weight=0.43),
    "fineweb-edu": Source(
        hf_id="HuggingFaceFW/fineweb-edu",
        split="train",
        config_name="sample-10BT",
        field="text",
        weight=0.24,
    ),
}

# Rough chars-per-token used only for budget accounting while streaming.
# Replaced by the real measured ratio after Phase 3.
CHARS_PER_TOKEN_ESTIMATE: float = 4.0

# ------------------------------------------------------------------ cleaning --

MIN_LINE_CHARS = 40
MAX_NON_ALNUM_RATIO = 0.30
MIN_DOC_CHARS = 600
REPETITION_TOP_K = 10
MAX_REPETITION_RATIO = 0.50
NGRAM_N = 4
LANGDETECT_PREFIX_CHARS = 5_000
MIN_ASCII_RATIO = 0.90
# Measured on 179 kept case-law docs: median 0.0000, p99 0.0123, max 0.0152.
# 0.02 sits above the observed distribution, so this is a safety net for badly
# garbled docs rather than a primary filter — the OCR-heavy scans in this corpus
# are short procedural orders already caught by the MIN_DOC_CHARS gate.
MAX_OCR_WORD_RATIO = 0.02
ENABLE_OCR_FILTER = True

# ----------------------------------------------------------------- tokenizer --

VOCAB_SIZE = 16_384

BOS, EOS, PAD, UNK = "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"
SPECIALS: tuple[str, ...] = (
    BOS,
    EOS,
    PAD,
    UNK,
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
)

TOKENIZER_SAMPLE_GB = 3.0  # text sampled to fit the BPE, not the whole corpus

# --------------------------------------------------------------------- model --

SEQ_LEN = 1024
N_LAYER = 12
D_MODEL = 768
N_HEAD = 12
N_KV_HEAD = 12  # MHA: kv heads == heads
D_FFN = 3072  # SwiGLU inner
TIE_EMBEDDINGS = True
ROPE_THETA = 10_000.0
RMS_NORM_EPS = 1e-5

# ------------------------------------------------------------------ training --

MICRO_BATCH = 32
GRAD_ACCUM = 4
N_GPU = 4
TOKENS_PER_STEP = MICRO_BATCH * GRAD_ACCUM * N_GPU * SEQ_LEN  # 524_288

EPOCHS = 5
STEPS_PER_EPOCH = int(TOKEN_BUDGET_B * 1e9) // TOKENS_PER_STEP  # 7_865
MAX_STEPS = STEPS_PER_EPOCH * EPOCHS  # 39_325

# The cosine schedule spans ALL epochs. Decaying over one epoch would leave
# epochs 2-5 training at the LR floor. See SLM_BUILD_GUIDE.md Phase 5.
LR_PEAK = 6e-4
LR_MIN = 6e-5
WARMUP_STEPS = 500
LR_DECAY_STEPS = MAX_STEPS

WEIGHT_DECAY = 0.1
BETAS: tuple[float, float] = (0.9, 0.95)
GRAD_CLIP = 1.0

# ----------------------------------------------------- checkpoints / resume --

CKPT_EVERY_STEPS = 500  # ~30 min at ~3.5 s/step
EVAL_EVERY_STEPS = 500
EVAL_BATCHES = 100
KEEP_LAST_N_CKPTS = 3
EARLY_STOP_PATIENCE = 5  # evals without val improvement before stopping
RESHUFFLE_EACH_EPOCH = True
DATA_SEED = 1337  # per-epoch seed = DATA_SEED + epoch

VAL_EVERY_NTH_WINDOW = 100  # deterministic 99/1 split

# --------------------------------------------------------------------- misc --

NPROC_CPU = 44  # of 48 - leave the desktop alive
SHARD_TARGET_BYTES = 256 * 1024 * 1024
SEED = 1337


def ensure_dirs() -> None:
    """Create every durable directory. Idempotent."""
    for d in (
        CLEAN_DIR,
        DEDUP_DIR,
        TOKENIZER_DIR,
        TRAIN_TOKENS_DIR,
        VAL_TOKENS_DIR,
        CKPT_DIR,
        LOG_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def n_parameters() -> int:
    """Exact-enough parameter count (norms and RoPE are negligible/zero).

    embeddings are tied, so the output projection adds nothing.
    """
    embed = VOCAB_SIZE * D_MODEL
    attn = 4 * D_MODEL * D_MODEL  # q, k, v, o
    mlp = 3 * D_MODEL * D_FFN  # SwiGLU: gate, up, down
    return embed + N_LAYER * (attn + mlp)


def summary() -> str:
    """Human-readable dump of the derived numbers, for the phase docs."""
    n_params = n_parameters()
    return "\n".join(
        [
            f"root            {SLM_ROOT}",
            f"corpus          {TOKEN_BUDGET_B}B tokens over {len(SOURCES)} sources",
            f"vocab           {VOCAB_SIZE}",
            f"model           {N_LAYER}L / {D_MODEL}d / {N_HEAD}h / ctx {SEQ_LEN}",
            f"params (approx) {n_params / 1e6:.0f}M",
            f"tokens/step     {TOKENS_PER_STEP:,}",
            f"steps/epoch     {STEPS_PER_EPOCH:,}",
            f"epochs          {EPOCHS}",
            f"max steps       {MAX_STEPS:,}",
            f"tokens seen     {MAX_STEPS * TOKENS_PER_STEP / 1e9:.1f}B "
            f"({MAX_STEPS * TOKENS_PER_STEP / n_params:.0f} tok/param)",
        ]
    )
