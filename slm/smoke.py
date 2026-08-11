"""Phase 0 smoke test.

Streams a handful of documents from each source, runs the cleaning chain, and
prints before/after plus a drop-reason tally. Writes nothing durable.

    python -m slm.smoke [--n 10]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass

from . import config as C
from .filters import DropReason, clean_document

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

RULE = "=" * 78
PREVIEW_CHARS = 300


@dataclass
class SourceReport:
    name: str
    ok: bool
    n_seen: int = 0
    n_kept: int = 0
    reasons: Counter | None = None
    error: str = ""
    sample_before: str = ""
    sample_after: str = ""


def check_gpus() -> bool:
    """Confirm the GPUs are visible to torch and bf16 is available."""
    print(RULE)
    print("GPU CHECK")
    print(RULE)
    try:
        import torch
    except ImportError as exc:
        print(f"  FAIL: torch not importable: {exc}")
        return False

    n = torch.cuda.device_count()
    print(f"  torch            {torch.__version__}")
    print(f"  cuda available   {torch.cuda.is_available()}")
    print(f"  device count     {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(
            f"    [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB  "
            f"sm_{props.major}{props.minor}"
        )
    bf16 = torch.cuda.is_bf16_supported() if n else False
    print(f"  bf16 supported   {bf16}")

    ok = n == C.N_GPU and bf16
    if not ok:
        print(f"  WARN: expected {C.N_GPU} GPUs with bf16, got {n} / bf16={bf16}")
    return ok


def stream_source(name: str, src: C.Source, n: int) -> SourceReport:
    """Pull `n` documents from one source and run them through the cleaner."""
    from datasets import load_dataset

    reasons: Counter = Counter()
    report = SourceReport(name=name, ok=False, reasons=reasons)

    try:
        kwargs = dict(split=src.split, streaming=True)
        if src.config_name:
            ds = load_dataset(src.hf_id, src.config_name, **kwargs)
        else:
            ds = load_dataset(src.hf_id, **kwargs)

        for i, row in enumerate(ds):
            if i >= n:
                break
            report.n_seen += 1

            if src.field not in row:
                report.error = (
                    f"field {src.field!r} missing; available: {sorted(row)[:12]}"
                )
                return report

            raw = row[src.field] or ""
            result = clean_document(raw)
            reasons[result.reason.value] += 1

            if result.kept:
                report.n_kept += 1
                if not report.sample_after:
                    report.sample_before = raw
                    report.sample_after = result.text

        report.ok = report.n_seen > 0
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole test
        report.error = f"{type(exc).__name__}: {exc}"

    return report


def print_report(r: SourceReport, src: C.Source) -> None:
    print()
    print(RULE)
    print(f"SOURCE: {r.name}   ({src.hf_id}, split={src.split}"
          + (f", config={src.config_name}" if src.config_name else "")
          + f", field={src.field})")
    print(RULE)

    if r.error:
        print(f"  ERROR: {r.error}")
        return

    print(f"  streamed {r.n_seen}, kept {r.n_kept}, dropped {r.n_seen - r.n_kept}")
    print(f"  budget for this source: {src.token_budget / 1e9:.2f}B tokens")
    if r.reasons:
        print("  reasons: " + ", ".join(
            f"{k}={v}" for k, v in sorted(r.reasons.items(), key=lambda kv: -kv[1])
        ))

    if r.sample_after:
        before = r.sample_before[:PREVIEW_CHARS].replace("\n", "\\n")
        after = r.sample_after[:PREVIEW_CHARS].replace("\n", "\\n")
        print(f"\n  --- BEFORE ({len(r.sample_before):,} chars) ---")
        print(f"  {before}...")
        print(f"\n  --- AFTER  ({len(r.sample_after):,} chars) ---")
        print(f"  {after}...")
    else:
        print("  (no document survived the chain - inspect the filters)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="docs per source")
    args = ap.parse_args()

    print(RULE)
    print("CONFIG")
    print(RULE)
    print(C.summary())

    gpu_ok = check_gpus()

    reports = []
    for name, src in C.SOURCES.items():
        print(f"\nstreaming {args.n} docs from {name} ...", flush=True)
        r = stream_source(name, src, args.n)
        reports.append(r)
        print_report(r, src)

    print()
    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"  gpus            {'OK' if gpu_ok else 'CHECK ABOVE'}")
    for r in reports:
        status = "OK" if r.ok and not r.error else "FAIL"
        detail = r.error if r.error else f"{r.n_kept}/{r.n_seen} kept"
        print(f"  {r.name:<14} {status:<5} {detail}")

    all_ok = gpu_ok and all(r.ok and not r.error for r in reports)
    print(f"\n  PHASE 0: {'PASS' if all_ok else 'INCOMPLETE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
