"""Phase 3: train a fresh 16K byte-level BPE on the deduplicated corpus.

    python -m slm.tokenizer             # train, save, validate
    python -m slm.tokenizer --validate  # validate an existing tokenizer only

Saves a HuggingFace PreTrainedTokenizerFast to $SLM_ROOT/tokenizer/.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

from . import config as C

RULE = "=" * 78


# ------------------------------------------------------------------ sampling --


def source_shards(source: str) -> list[Path]:
    return sorted((C.CORPUS_DIR / source).glob("shard_*.txt.gz"))


def sample_iterator(total_gb: float, verbose: bool = True):
    """Yield documents from every source, proportional to its corpus weight.

    Streams through a generator so the sample never touches disk. Each source
    stops at its own byte budget; shards are read in order, which is fine
    because Phase 1 wrote them in stream order (no sorting by length or source).
    """
    total_bytes = int(total_gb * 1024**3)
    for name, src in C.SOURCES.items():
        budget = int(total_bytes * src.weight)
        used = 0
        for path in source_shards(name):
            if used >= budget:
                break
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    text = line.rstrip("\n")
                    if not text:
                        continue
                    yield text
                    used += len(text)
                    if used >= budget:
                        break
        if verbose:
            print(f"  sampled {used/1e9:.2f} GB from {name} "
                  f"(target {budget/1e9:.2f} GB)", flush=True)


# ------------------------------------------------------------------ training --


def train() -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    print(RULE)
    print(f"training {C.VOCAB_SIZE}-token byte-level BPE on "
          f"~{C.TOKENIZER_SAMPLE_GB} GB from {C.CORPUS_DIR}")
    print(RULE)

    tok = Tokenizer(models.BPE())
    # add_prefix_space=False: legal/financial text is not sentence-fragmentary,
    # and a leading space would waste a token on every document start.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=C.VOCAB_SIZE,
        special_tokens=list(C.SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=True,
    )

    t0 = time.time()
    tok.train_from_iterator(sample_iterator(C.TOKENIZER_SAMPLE_GB), trainer=trainer)
    print(f"\ntrained in {(time.time()-t0)/60:.1f} min", flush=True)

    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=C.BOS,
        eos_token=C.EOS,
        pad_token=C.PAD,
        unk_token=C.UNK,
        additional_special_tokens=[s for s in C.SPECIALS if s not in
                                   (C.BOS, C.EOS, C.PAD, C.UNK)],
    )
    C.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(C.TOKENIZER_DIR)
    print(f"saved to {C.TOKENIZER_DIR}")


# ---------------------------------------------------------------- validation --

HOSTILE = {
    "ocr garble": "supe COURT OE ALABAMA STATS oF ALABAMA HAWAT'T ggg 1180400",
    "emoji": "The court ruled 🏛️⚖️ in favour of the plaintiff 👨‍⚖️.",
    "cjk": "最高裁判所は原告の主張を認めた。法律。",
    "control bytes": "a\x00b\x01c\x1fd\x7fe",
    "all bytes 0-255": bytes(range(256)).decode("latin-1"),
    "math/unicode": "∫∂ℝ≤α β γ — “quoted” … ﬁ ligature",
    "whitespace": "tabs\there\nand\r\nnewlines   and    spaces",
}


def validate() -> bool:
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(str(C.TOKENIZER_DIR))
    ok = True

    print(f"\n{RULE}\nVALIDATION\n{RULE}")
    print(f"  vocab_size        {tk.vocab_size}")
    print(f"  model_max_length  {tk.model_max_length}")
    for name in ("bos", "eos", "pad", "unk"):
        tokn = getattr(tk, f"{name}_token")
        tid = getattr(tk, f"{name}_token_id")
        print(f"  {name:<4} {tokn!r:<14} id={tid}")
    if tk.vocab_size != C.VOCAB_SIZE:
        print(f"  FAIL: vocab {tk.vocab_size} != {C.VOCAB_SIZE}")
        ok = False

    # --- no-UNK guarantee on hostile input ---------------------------------
    print("\n  hostile input (byte-level must never emit UNK, must round-trip):")
    unk_id = tk.unk_token_id
    for label, text in HOSTILE.items():
        ids = tk.encode(text, add_special_tokens=False)
        back = tk.decode(ids, skip_special_tokens=True)
        has_unk = unk_id is not None and unk_id in ids
        rt = back == text
        flag = "ok " if (rt and not has_unk) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"    {flag} {label:<18} {len(text):>4}ch -> {len(ids):>4}tok"
              f"  unk={has_unk} roundtrip={rt}")
        if not rt:
            print(f"         got: {back[:80]!r}")

    # --- round-trip + compression on real corpus docs ----------------------
    print("\n  real documents (round-trip + compression):")
    total_chars = total_tokens = 0
    per_source = {}
    for name in C.SOURCES:
        shards = source_shards(name)
        if not shards:
            continue
        chars = toks = n = 0
        bad = 0
        # Spread the sample across shards. Reading only shards[0] biases the
        # ratio toward whatever happened to stream first.
        step = max(1, len(shards) // 8)
        for shard in shards[::step][:8]:
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 120:
                        break
                    text = line.rstrip("\n")
                    ids = tk.encode(text, add_special_tokens=False)
                    if tk.decode(ids, skip_special_tokens=True) != text:
                        bad += 1
                    chars += len(text)
                    toks += len(ids)
                    n += 1
        ratio = chars / max(1, toks)
        per_source[name] = ratio
        total_chars += chars
        total_tokens += toks
        if bad:
            ok = False
        print(f"    {name:<14} {n} docs  {ratio:.2f} chars/token"
              f"  roundtrip_failures={bad}")

    # Weight by each source's CHAR count, never by document count. SEC docs are
    # ~40x longer than fineweb-edu docs, so pooling a fixed number of documents
    # per source reports SEC's ratio as if it were the corpus average (that
    # mistake produced a bogus 4.875 on the first run).
    est_tokens = sum(C.CORPUS_CHARS[n] / r for n, r in per_source.items())
    weighted = sum(C.CORPUS_CHARS[n] for n in per_source) / max(1.0, est_tokens)
    pooled = total_chars / max(1, total_tokens)
    print(f"\n  compression (char-weighted): {weighted:.3f} chars/token "
          f"(config has {C.CHARS_PER_TOKEN_ESTIMATE})")
    print(f"  compression (pooled, biased): {pooled:.3f}  <- do not use")
    print(f"  implied corpus size: {est_tokens/1e9:.3f}B tokens "
          f"(currently configured {C.TOKEN_BUDGET_B}B)")

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="skip training")
    args = ap.parse_args()

    if not args.validate:
        train()
    return 0 if validate() else 1


if __name__ == "__main__":
    sys.exit(main())
