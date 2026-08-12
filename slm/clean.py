"""Phase 1: stream each source from HuggingFace, clean on the fly, write shards.

    python -m slm.clean                    # full run, all sources
    python -m slm.clean --source sec       # one source
    python -m slm.clean --max-docs 500     # bounded trial run
    python -m slm.clean --dry-run          # clean but write nothing

Design: one process per source (they are independent), each streaming
sequentially and fanning the pure cleaning functions out over a worker pool.
Progress is checkpointed per shard so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config as C
from .filters import DropReason, clean_document

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MIN_FREE_BYTES = 15 * 1024**3  # abort below this much free disk
RETRY_BACKOFF_S = (5, 15, 60, 180)
CHUNK = 200  # docs handed to the pool at a time


# ------------------------------------------------------------------ progress --


@dataclass
class Progress:
    """Resume state and running tally for one source. Serialised to JSON."""

    source: str
    docs_seen: int = 0
    docs_kept: int = 0
    chars_kept: int = 0
    shard_index: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    finished: bool = False

    @property
    def est_tokens(self) -> int:
        return int(self.chars_kept / C.CHARS_PER_TOKEN_ESTIMATE)

    @classmethod
    def load(cls, path: Path, source: str) -> "Progress":
        if path.exists():
            try:
                return cls(**json.loads(path.read_text()))
            except Exception:
                pass
        return cls(source=source)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)  # atomic


# -------------------------------------------------------------------- writer --


class ShardWriter:
    """Appends cleaned documents, one per line, to rolling gzip shards."""

    def __init__(self, out_dir: Path, start_index: int, dry_run: bool = False):
        self.out_dir = out_dir
        self.index = start_index
        self.dry_run = dry_run
        self._fh = None
        self._bytes = 0
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.dry_run:
            return
        path = self.out_dir / f"shard_{self.index:05d}.txt.gz"
        self._fh = gzip.open(path, "at", compresslevel=6, encoding="utf-8")

    def write(self, text: str) -> None:
        if self.dry_run:
            self._bytes += len(text) + 1
            if self._bytes >= C.SHARD_TARGET_BYTES:
                self.index += 1
                self._bytes = 0
            return
        if self._fh is None:
            self._open()
        self._fh.write(text + "\n")
        self._bytes += len(text) + 1
        if self._bytes >= C.SHARD_TARGET_BYTES:
            self.close()
            self.index += 1
            self._bytes = 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# -------------------------------------------------------------------- stream --


def _iter_rows(src: C.Source):
    """Stream a source, retrying transient failures with backoff."""
    from datasets import load_dataset

    attempt = 0
    while True:
        try:
            kwargs = dict(split=src.split, streaming=True)
            ds = (
                load_dataset(src.hf_id, src.config_name, **kwargs)
                if src.config_name
                else load_dataset(src.hf_id, **kwargs)
            )
            yield from ds
            return
        except Exception as exc:  # noqa: BLE001
            if attempt >= len(RETRY_BACKOFF_S):
                raise
            wait = RETRY_BACKOFF_S[attempt]
            print(
                f"  [{src.hf_id}] stream error ({type(exc).__name__}), "
                f"retry in {wait}s: {str(exc)[:120]}",
                flush=True,
            )
            time.sleep(wait)
            attempt += 1


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def clean_source(
    name: str,
    src: C.Source,
    max_docs: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> Progress:
    """Stream one source to its token budget, cleaning as it goes."""
    out_dir = C.CLEAN_DIR / name
    prog_path = out_dir / "progress.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    prog = Progress.load(prog_path, name)
    if prog.finished and not dry_run:
        print(f"[{name}] already finished ({prog.est_tokens/1e9:.2f}B tokens)")
        return prog

    budget = src.token_budget
    skip = prog.docs_seen  # resume: fast-forward past what we already consumed
    writer = ShardWriter(out_dir, prog.shard_index, dry_run=dry_run)
    t0 = time.time()
    last_report = t0

    if skip and not quiet:
        print(f"[{name}] resuming, skipping {skip:,} already-consumed docs")

    pool = mp.Pool(processes=max(1, C.NPROC_CPU // len(C.SOURCES)))
    batch: list[str] = []

    def flush(batch: list[str]) -> None:
        for result in pool.map(clean_document, batch):
            prog.drops[result.reason.value] = prog.drops.get(result.reason.value, 0) + 1
            if result.kept:
                writer.write(result.text)
                prog.docs_kept += 1
                prog.chars_kept += len(result.text)

    try:
        for i, row in enumerate(_iter_rows(src)):
            if i < skip:
                continue
            if max_docs is not None and prog.docs_seen - skip >= max_docs:
                break

            prog.docs_seen += 1
            batch.append(row.get(src.field) or "")

            if len(batch) >= CHUNK:
                flush(batch)
                batch = []
                prog.shard_index = writer.index
                if not dry_run:
                    prog.save(prog_path)

                if prog.est_tokens >= budget:
                    prog.finished = True
                    break

                if _free_bytes(C.SLM_ROOT) < MIN_FREE_BYTES:
                    print(f"[{name}] ABORT: free disk below floor", flush=True)
                    break

                now = time.time()
                if not quiet and now - last_report >= 60:
                    rate = prog.est_tokens / max(1e-9, now - t0)
                    pct = 100 * prog.est_tokens / budget
                    eta_h = (budget - prog.est_tokens) / max(1.0, rate) / 3600
                    print(
                        f"[{name}] {prog.est_tokens/1e9:.3f}B/{budget/1e9:.1f}B "
                        f"({pct:.1f}%)  kept {prog.docs_kept:,}/{prog.docs_seen:,}  "
                        f"{rate/1e6:.1f}M tok/s  eta {eta_h:.1f}h",
                        flush=True,
                    )
                    last_report = now

        if batch:
            flush(batch)
    finally:
        pool.close()
        pool.join()
        writer.close()
        prog.shard_index = writer.index
        if not dry_run:
            prog.save(prog_path)

    dt = time.time() - t0
    if not quiet:
        print(
            f"[{name}] done in {dt/60:.1f} min: kept {prog.docs_kept:,}/"
            f"{prog.docs_seen:,} docs, ~{prog.est_tokens/1e9:.3f}B tokens"
        )
    return prog


# ---------------------------------------------------------------- reporting --


def report(progs: list[Progress]) -> None:
    rule = "=" * 78
    print(f"\n{rule}\nPHASE 1 REPORT\n{rule}")
    total_tokens = 0
    for p in progs:
        src = C.SOURCES[p.source]
        total_tokens += p.est_tokens
        kept_pct = 100 * p.docs_kept / max(1, p.docs_seen)
        print(f"\n  {p.source}  (budget {src.token_budget/1e9:.1f}B)")
        print(f"    seen {p.docs_seen:,}  kept {p.docs_kept:,} ({kept_pct:.1f}%)")
        print(f"    est tokens {p.est_tokens/1e9:.3f}B")
        for reason in sorted(p.drops, key=lambda r: -p.drops[r]):
            if reason != DropReason.KEPT.value:
                n = p.drops[reason]
                print(f"      drop {reason:<16} {n:>10,} ({100*n/max(1,p.docs_seen):.1f}%)")
    print(f"\n  TOTAL est tokens: {total_tokens/1e9:.2f}B")
    print(f"  free disk: {_free_bytes(C.SLM_ROOT)/1e9:.0f} GB")


def _run_one(args) -> Progress:
    name, max_docs, dry_run = args
    return clean_source(name, C.SOURCES[name], max_docs=max_docs, dry_run=dry_run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(C.SOURCES), help="only this source")
    ap.add_argument("--max-docs", type=int, help="stop after N docs per source")
    ap.add_argument("--dry-run", action="store_true", help="clean but write nothing")
    ap.add_argument("--serial", action="store_true", help="one source at a time")
    args = ap.parse_args()

    C.ensure_dirs()
    names = [args.source] if args.source else list(C.SOURCES)

    print(f"sources: {', '.join(names)}")
    print(f"output:  {C.CLEAN_DIR}")
    print(f"free:    {_free_bytes(C.SLM_ROOT)/1e9:.0f} GB\n")

    jobs = [(n, args.max_docs, args.dry_run) for n in names]
    if args.serial or len(names) == 1:
        progs = [_run_one(j) for j in jobs]
    else:
        # Threads, not processes, for the per-source fan-out: each source is
        # network-bound and delegates its CPU work to its own mp.Pool. Nesting
        # a Pool inside a Pool worker is impossible (workers are daemonic and
        # daemons cannot have children), and threads sidestep that entirely.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            progs = list(ex.map(_run_one, jobs))

    report(progs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
