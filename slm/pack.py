"""Phase 4: tokenize the corpus, pack into uint16 1024-token windows, split 99/1.

    python -m slm.pack            # tokenize, pack, split, write index.json
    python -m slm.pack --verify   # verify an existing token set

Split is by DOCUMENT, not by window - see docs/04-tokenize.md for why.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE = np.uint16
ENCODE_BATCH = 256
VAL_EVERY_NTH_DOC = C.VAL_EVERY_NTH_WINDOW  # 100 -> 1% of documents

_tk = None


def _init_worker() -> None:
    global _tk
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _tk = AutoTokenizer.from_pretrained(str(C.TOKENIZER_DIR))


@dataclass
class PackResult:
    source: str
    shard: int
    train_windows: int
    val_windows: int
    train_tokens: int
    val_tokens: int
    docs: int
    dropped_remainder: int


def _pack_stream(ids: list[int]) -> np.ndarray:
    """Cut a flat id stream into whole 1024-token windows. Remainder dropped."""
    n_win = len(ids) // C.SEQ_LEN
    if n_win == 0:
        return np.empty((0, C.SEQ_LEN), dtype=DTYPE)
    arr = np.asarray(ids[: n_win * C.SEQ_LEN], dtype=DTYPE)
    return arr.reshape(n_win, C.SEQ_LEN)


def pack_shard(args) -> PackResult:
    """Tokenize one input shard into its own train and val .bin files."""
    source, shard_idx, path = args
    assert _tk is not None
    eos = _tk.eos_token_id

    train_ids: list[int] = []
    val_ids: list[int] = []
    docs = 0
    batch: list[str] = []
    batch_is_val: list[bool] = []

    def flush() -> None:
        if not batch:
            return
        enc = _tk(batch, add_special_tokens=False)["input_ids"]
        for ids, is_val in zip(enc, batch_is_val):
            (val_ids if is_val else train_ids).extend(ids)
            (val_ids if is_val else train_ids).append(eos)

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if not text:
                continue
            # Document-level split: the whole document goes to one side, so no
            # val text has its surrounding context sitting in train.
            batch.append(text)
            batch_is_val.append(docs % VAL_EVERY_NTH_DOC == VAL_EVERY_NTH_DOC - 1)
            docs += 1
            if len(batch) >= ENCODE_BATCH:
                flush()
                batch, batch_is_val = [], []
    flush()

    dropped = (len(train_ids) % C.SEQ_LEN) + (len(val_ids) % C.SEQ_LEN)
    out = {}
    for split, ids, out_dir in (
        ("train", train_ids, C.TRAIN_TOKENS_DIR),
        ("val", val_ids, C.VAL_TOKENS_DIR),
    ):
        windows = _pack_stream(ids)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{source}_{shard_idx:05d}.bin"
        if windows.size:
            windows.tofile(fname)
        elif fname.exists():
            fname.unlink()
        out[split] = windows.shape[0]

    return PackResult(
        source=source,
        shard=shard_idx,
        train_windows=out["train"],
        val_windows=out["val"],
        train_tokens=out["train"] * C.SEQ_LEN,
        val_tokens=out["val"] * C.SEQ_LEN,
        docs=docs,
        dropped_remainder=dropped,
    )


def write_index(out_dir: Path, results: list[PackResult], split: str) -> dict:
    shards = []
    for r in sorted(results, key=lambda r: (r.source, r.shard)):
        n = r.train_windows if split == "train" else r.val_windows
        if n == 0:
            continue
        shards.append(
            dict(name=f"{r.source}_{r.shard:05d}.bin", source=r.source, windows=n,
                 tokens=n * C.SEQ_LEN)
        )
    index = dict(
        split=split,
        dtype="uint16",
        seq_len=C.SEQ_LEN,
        vocab_size=C.VOCAB_SIZE,
        eos_id=1,
        total_windows=sum(s["windows"] for s in shards),
        total_tokens=sum(s["tokens"] for s in shards),
        shards=shards,
    )
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def run() -> int:
    C.ensure_dirs()
    jobs = []
    for source in C.SOURCES:
        for i, p in enumerate(sorted((C.CORPUS_DIR / source).glob("shard_*.txt.gz"))):
            jobs.append((source, i, p))

    print(f"tokenizing {len(jobs)} shards from {C.CORPUS_DIR}")
    print(f"  -> {C.TOKENS_DIR}  (uint16, seq_len={C.SEQ_LEN}, "
          f"val = every {VAL_EVERY_NTH_DOC}th DOCUMENT)")
    t0 = time.time()

    with mp.Pool(C.NPROC_CPU, initializer=_init_worker) as pool:
        results = pool.map(pack_shard, jobs)

    dt = time.time() - t0
    train_idx = write_index(C.TRAIN_TOKENS_DIR, results, "train")
    val_idx = write_index(C.VAL_TOKENS_DIR, results, "val")

    rule = "=" * 78
    print(f"\n{rule}\nPHASE 4 REPORT\n{rule}")
    print(f"  tokenized in {dt/60:.1f} min ({sum(r.docs for r in results):,} docs)")
    for name in C.SOURCES:
        rs = [r for r in results if r.source == name]
        tr = sum(r.train_tokens for r in rs)
        va = sum(r.val_tokens for r in rs)
        print(f"    {name:<14} train {tr/1e9:.3f}B  val {va/1e6:.1f}M  "
              f"({sum(r.docs for r in rs):,} docs)")
    tot = train_idx["total_tokens"] + val_idx["total_tokens"]
    print(f"\n  train  {train_idx['total_tokens']/1e9:.4f}B tokens "
          f"({train_idx['total_windows']:,} windows, {len(train_idx['shards'])} shards)")
    print(f"  val    {val_idx['total_tokens']/1e6:.1f}M tokens "
          f"({val_idx['total_windows']:,} windows)")
    print(f"  val fraction: {100*val_idx['total_tokens']/tot:.2f}%")
    print(f"  remainder dropped: {sum(r.dropped_remainder for r in results):,} tokens "
          f"({100*sum(r.dropped_remainder for r in results)/tot:.4f}%)")
    return 0


# ------------------------------------------------------------------- verify --


def verify() -> bool:
    from transformers import AutoTokenizer

    ok = True
    tk = AutoTokenizer.from_pretrained(str(C.TOKENIZER_DIR))
    rule = "=" * 78
    print(f"{rule}\nVERIFY\n{rule}")

    for split, d in (("train", C.TRAIN_TOKENS_DIR), ("val", C.VAL_TOKENS_DIR)):
        idx = json.loads((d / "index.json").read_text())
        on_disk = sum(f.stat().st_size for f in d.glob("*.bin"))
        expected = idx["total_tokens"] * 2  # uint16
        match = on_disk == expected
        ok &= match
        print(f"  {split:<6} {idx['total_tokens']:>14,} tokens  "
              f"{idx['total_windows']:>10,} windows  "
              f"bytes {'OK' if match else f'MISMATCH {on_disk} vs {expected}'}")

    # id range must fit the vocab
    d = C.TRAIN_TOKENS_DIR
    sample = sorted(d.glob("*.bin"))[0]
    arr = np.fromfile(sample, dtype=DTYPE)
    print(f"\n  id range in {sample.name}: min={arr.min()} max={arr.max()} "
          f"(vocab {C.VOCAB_SIZE})")
    if arr.max() >= C.VOCAB_SIZE:
        print("  FAIL: token id >= vocab_size")
        ok = False

    # a window should decode to fluent text
    win = arr[: C.SEQ_LEN]
    text = tk.decode(win.tolist(), skip_special_tokens=False)
    eos_count = int((arr[: C.SEQ_LEN * 20] == 1).sum())
    print(f"  EOS density: {eos_count} in 20 windows "
          f"(~{20*C.SEQ_LEN/max(1,eos_count):.0f} tokens/doc)")
    print(f"\n  --- first window decoded (300 chars) ---\n  {text[:300]!r}")

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not args.verify:
        run()
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
