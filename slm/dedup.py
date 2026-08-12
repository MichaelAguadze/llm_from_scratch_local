"""Phase 2: exact dedup, MinHash near-dedup, and eval decontamination.

    python -m slm.dedup             # full run, writes $SLM_ROOT/dedup/
    python -m slm.dedup --report    # analyse only, write nothing
    python -m slm.dedup --source sec

Reads $SLM_ROOT/clean/ and never modifies it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C

# MinHash parameters
N_PERM = 64
N_BANDS = 8
ROWS_PER_BAND = N_PERM // N_BANDS  # 8 -> LSH threshold ~= (1/8)^(1/8) = 0.76
JACCARD_THRESHOLD = 0.80
SHINGLE_WORDS = 5
MAX_SHINGLES = 4096  # bottom-k sample, consistent across documents
MERSENNE_61 = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1

# Decontamination
CONTAM_NGRAM = 13
CONTAM_MIN_HITS = 5

_rng = np.random.default_rng(C.SEED)
_PERM_A = _rng.integers(1, MERSENNE_61, size=N_PERM, dtype=np.uint64)
_PERM_B = _rng.integers(0, MERSENNE_61, size=N_PERM, dtype=np.uint64)


# ----------------------------------------------------------------- hashing --


def exact_key(text: str) -> bytes:
    """Stable hash of normalised text, for exact duplicate detection."""
    norm = " ".join(text.lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).digest()


def word_hashes(words: list[str]) -> np.ndarray:
    """32-bit hash per word.

    crc32, not the builtin hash(): str hashing is salted per interpreter, which
    would make signatures irreproducible across runs and inconsistent under a
    'spawn' start method. crc32 is deterministic and C-speed.
    """
    return np.fromiter(
        (zlib.crc32(w.encode("utf-8", "ignore")) for w in words),
        dtype=np.uint64,
        count=len(words),
    )


def ngram_hashes(words: list[str], k: int, bits: int = 64) -> np.ndarray:
    """Rolling polynomial hash over k-word windows, vectorised. Sorted unique.

    `bits` sets the hash space, and it matters more than it looks. A membership
    test against an index of M entries has a false-match probability of
    M / 2**bits per gram. For decontamination M is 27.5M and documents carry
    ~4,400 grams, so 32 bits yields ~28 expected FALSE hits per document —
    above any sane threshold, flagging essentially the whole corpus. 64 bits
    drops that to ~1e-8. MinHash shingling compares ~4,400-element sets against
    each other rather than a huge index, so 32 bits is harmless there.
    """
    n = len(words)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    wh = word_hashes(words)
    out = np.zeros(n - k + 1, dtype=np.uint64)
    mult = np.uint64(1_000_003)
    use_mask = bits < 64
    mask = np.uint64((1 << bits) - 1) if use_mask else None
    with np.errstate(over="ignore"):  # uint64 wraparound is the intended mixing
        for i in range(k):
            out = out * mult + wh[i : n - k + 1 + i]
            if use_mask:
                out = out & mask
    return np.unique(out)


def shingle_hashes(words: list[str]) -> np.ndarray:
    # 32-bit: validated against ground-truth Jaccard, see docs/02-dedup.md
    return ngram_hashes(words, SHINGLE_WORDS, bits=32)


def minhash_signature(words: list[str]) -> np.ndarray | None:
    """64-perm MinHash over bottom-k sampled shingles."""
    sh = shingle_hashes(words)
    if sh.size == 0:
        return None
    if sh.size > MAX_SHINGLES:
        sh = np.partition(sh, MAX_SHINGLES)[:MAX_SHINGLES]  # bottom-k: consistent
    # (a * h + b) mod p, minimum over shingles, per permutation
    prod = (_PERM_A[:, None] * sh[None, :] + _PERM_B[:, None]) % np.uint64(MERSENNE_61)
    return prod.min(axis=1).astype(np.uint32)


# ------------------------------------------------------------------- shards --


def shard_paths(source: str, root: Path) -> list[Path]:
    return sorted((root / source).glob("shard_*.txt.gz"))


def read_shard(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if text:
                yield line_no, text


@dataclass
class ShardScan:
    shard: int
    keys: list[bytes]
    sigs: np.ndarray | None
    n_docs: int


def scan_shard(args) -> ShardScan:
    """Compute exact keys and (optionally) MinHash signatures for one shard."""
    path, shard_idx, want_minhash = args
    keys: list[bytes] = []
    sigs: list[np.ndarray] = []
    for _, text in read_shard(path):
        keys.append(exact_key(text))
        if want_minhash:
            sig = minhash_signature(text.split())
            sigs.append(sig if sig is not None else np.zeros(N_PERM, dtype=np.uint32))
    arr = np.array(sigs, dtype=np.uint32) if want_minhash and sigs else None
    return ShardScan(shard=shard_idx, keys=keys, sigs=arr, n_docs=len(keys))


# ------------------------------------------------------- near-dup clustering --


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # keep the earliest doc


def near_duplicates(sigs: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """LSH-band the signatures and return a boolean mask of near-duplicates.

    Within each band bucket, members are verified against the bucket's first
    document and unioned if signature agreement >= JACCARD_THRESHOLD. The
    survivor of each cluster is its lowest index.
    """
    n = sigs.shape[0]
    uf = UnionFind(n)
    idx_alive = np.flatnonzero(alive)

    for band in range(N_BANDS):
        lo = band * ROWS_PER_BAND
        rows = sigs[:, lo : lo + ROWS_PER_BAND]
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i in idx_alive:
            buckets[rows[i].tobytes()].append(int(i))
        for members in buckets.values():
            if len(members) < 2:
                continue
            rep = members[0]
            rep_sig = sigs[rep]
            for other in members[1:]:
                if uf.find(other) == uf.find(rep):
                    continue
                agree = float(np.count_nonzero(sigs[other] == rep_sig)) / N_PERM
                if agree >= JACCARD_THRESHOLD:
                    uf.union(rep, other)

    is_dup = np.zeros(n, dtype=bool)
    for i in idx_alive:
        if uf.find(int(i)) != int(i):
            is_dup[i] = True
    return is_dup


# ------------------------------------------------------------ contamination --


def build_contamination_index() -> np.ndarray:
    """Sorted 13-gram hashes from CaseHOLD / LexGLUE. Empty array if unavailable.

    Returned as a sorted numpy array rather than a set: 27M entries as a Python
    set is ~2 GB and would be pickled once per Pool worker. A numpy array is
    ~220 MB, is shared copy-on-write across forked workers with no pickling, and
    supports vectorised membership via searchsorted.
    """
    from datasets import load_dataset

    grams: set[int] = set()
    specs = [
        ("coastalcph/lex_glue", "case_hold", ("context", "endings")),
        ("coastalcph/lex_glue", "ecthr_a", ("text",)),
        ("coastalcph/lex_glue", "scotus", ("text",)),
    ]
    for repo, cfg, fields in specs:
        try:
            for split in ("test", "validation"):
                ds = load_dataset(repo, cfg, split=split)
                for row in ds:
                    for f in fields:
                        val = row.get(f)
                        if val is None:
                            continue
                        chunks = val if isinstance(val, list) else [val]
                        for chunk in chunks:
                            if isinstance(chunk, list):
                                chunk = " ".join(map(str, chunk))
                            grams.update(_ngram_hashes(str(chunk)))
            print(f"  contamination: indexed {repo}/{cfg}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  contamination: SKIP {repo}/{cfg} ({type(exc).__name__}: "
                  f"{str(exc)[:90]})", flush=True)
    return np.sort(np.fromiter(grams, dtype=np.uint64, count=len(grams)))


def _ngram_hashes(text: str) -> np.ndarray:
    return ngram_hashes(text.lower().split(), CONTAM_NGRAM, bits=64)


# Set in the parent before forking; workers inherit it copy-on-write.
_CONTAM: np.ndarray | None = None


def contamination_hits(path: Path) -> list[int]:
    """Indices within a shard whose 13-gram overlap exceeds the threshold."""
    assert _CONTAM is not None
    hits = []
    for i, (_, text) in enumerate(read_shard(path)):
        doc = _ngram_hashes(text)
        if doc.size == 0:
            continue
        pos = np.searchsorted(_CONTAM, doc)
        pos[pos >= _CONTAM.size] = 0
        if int(np.count_nonzero(_CONTAM[pos] == doc)) >= CONTAM_MIN_HITS:
            hits.append(i)
    return hits


# ------------------------------------------------------------------- driver --


def process_source(name: str, want_minhash: bool, report_only: bool) -> dict:
    t0 = time.time()
    paths = shard_paths(name, C.CLEAN_DIR)
    print(f"\n[{name}] {len(paths)} shards", flush=True)

    with mp.Pool(C.NPROC_CPU) as pool:
        scans = pool.map(
            scan_shard, [(p, i, want_minhash) for i, p in enumerate(paths)]
        )
    n_docs = sum(s.n_docs for s in scans)
    print(f"[{name}] scanned {n_docs:,} docs in {time.time()-t0:.0f}s", flush=True)

    # --- exact dedup, first occurrence wins -------------------------------
    alive = np.ones(n_docs, dtype=bool)
    seen: set[bytes] = set()
    offsets, off = [], 0
    for s in scans:
        offsets.append(off)
        off += s.n_docs
    for s, base in zip(scans, offsets):
        for j, key in enumerate(s.keys):
            if key in seen:
                alive[base + j] = False
            else:
                seen.add(key)
    n_exact = int((~alive).sum())
    print(f"[{name}] exact duplicates: {n_exact:,}", flush=True)

    # --- near dedup --------------------------------------------------------
    n_near = 0
    if want_minhash:
        sigs = np.concatenate([s.sigs for s in scans if s.sigs is not None])
        t1 = time.time()
        dup = near_duplicates(sigs, alive)
        alive &= ~dup
        n_near = int(dup.sum())
        print(f"[{name}] near duplicates: {n_near:,} ({time.time()-t1:.0f}s)",
              flush=True)

    # --- contamination -----------------------------------------------------
    n_contam = 0
    if _CONTAM is not None and _CONTAM.size:
        with mp.Pool(C.NPROC_CPU) as pool:
            per_shard = pool.map(contamination_hits, paths)
        for s, base, hits in zip(scans, offsets, per_shard):
            for i in hits:
                if alive[base + i]:
                    alive[base + i] = False
                    n_contam += 1
        print(f"[{name}] contaminated: {n_contam:,}", flush=True)

    # --- write -------------------------------------------------------------
    n_kept = int(alive.sum())
    if not report_only:
        out_dir = C.DEDUP_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("shard_*.txt.gz"):
            old.unlink()
        written, shard_i, buf_bytes = 0, 0, 0
        fh = gzip.open(out_dir / f"shard_{shard_i:05d}.txt.gz", "wt",
                       compresslevel=6, encoding="utf-8")
        try:
            for s, base, path in zip(scans, offsets, paths):
                for i, (_, text) in enumerate(read_shard(path)):
                    if not alive[base + i]:
                        continue
                    fh.write(text + "\n")
                    written += 1
                    buf_bytes += len(text) + 1
                    if buf_bytes >= C.SHARD_TARGET_BYTES:
                        fh.close()
                        shard_i += 1
                        buf_bytes = 0
                        fh = gzip.open(out_dir / f"shard_{shard_i:05d}.txt.gz",
                                       "wt", compresslevel=6, encoding="utf-8")
        finally:
            fh.close()
        print(f"[{name}] wrote {written:,} docs to {shard_i+1} shards", flush=True)

    return dict(
        source=name, n_docs=n_docs, n_exact=n_exact, n_near=n_near,
        n_contam=n_contam, n_kept=n_kept,
        pct_kept=round(100 * n_kept / max(1, n_docs), 2),
        seconds=round(time.time() - t0, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(C.SOURCES))
    ap.add_argument("--report", action="store_true", help="analyse, write nothing")
    ap.add_argument("--no-contam", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    names = [args.source] if args.source else list(C.SOURCES)

    global _CONTAM
    if not args.no_contam:
        print("building contamination index (CaseHOLD / LexGLUE) ...", flush=True)
        _CONTAM = build_contamination_index()
        print(f"  {_CONTAM.size:,} benchmark 13-grams indexed "
              f"({_CONTAM.nbytes/1e6:.0f} MB, fork-shared)", flush=True)

    stats = [
        process_source(n, want_minhash=(n == "case-law"), report_only=args.report)
        for n in names
    ]

    rule = "=" * 78
    print(f"\n{rule}\nPHASE 2 REPORT\n{rule}")
    tot_in = tot_out = 0
    for st in stats:
        tot_in += st["n_docs"]
        tot_out += st["n_kept"]
        print(f"\n  {st['source']}  ({st['seconds']}s)")
        print(f"    in {st['n_docs']:,}  ->  kept {st['n_kept']:,} ({st['pct_kept']}%)")
        print(f"      exact dup    {st['n_exact']:,}")
        print(f"      near dup     {st['n_near']:,}")
        print(f"      contaminated {st['n_contam']:,}")
    print(f"\n  TOTAL {tot_in:,} -> {tot_out:,} "
          f"({100*tot_out/max(1,tot_in):.1f}% kept)")

    if not args.report:
        (C.DEDUP_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
