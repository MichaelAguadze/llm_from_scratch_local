# Phase 1 — Stream + clean

**Goal:** build the durable cleaned corpus — ~10B tokens of legal/financial/web
text — by streaming from HuggingFace and cleaning on the fly. This is the asset
every later phase and every future model reuses.

## What runs

```bash
tmux new -s clean
source .venv/bin/activate && source .env.local
python -m slm.clean 2>&1 | tee $SLM_ROOT/logs/01-clean.log
```

One process per source, each fanning cleaning out over a worker pool. Documents
are pulled one at a time, cleaned, and appended to gzipped shards. **The raw
datasets are never materialised** — we stop each source at its token budget.

| Source | HF id | Weight | Budget | Field |
|---|---|---|---|---|
| case-law | `HFforLegal/case-law` (split `us`) | 0.70 | 7.0B | `document` |
| sec | `PleIAs/SEC` | 0.20 | 2.0B | `text` |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 0.10 | 1.0B | `text` |

## Output layout

```
$SLM_ROOT/clean/<source>/shard_00000.txt.gz   one cleaned document per line
$SLM_ROOT/clean/<source>/progress.json        resume state + drop tally
```

**One document per line** means a document can never contain a raw newline —
`normalize_whitespace` collapses them, which is why it runs before writing.
Shards roll at ~256 MB uncompressed.

## Design decisions

- **Gzip, not plain text.** ~3× compression turns ~40 GB into ~14 GB. Costs
  almost nothing at streaming speeds, and every reader opens with `gzip.open`.
- **Resumable.** `progress.json` records documents consumed and tokens kept per
  source, flushed with each shard. An interrupted run skips forward instead of
  restarting an 8-hour stream.
- **Budget in tokens, estimated at 4 chars/token.** Real token counts only exist
  after Phase 3, so the budget is approximate by construction. Overshoot is
  harmless; undershoot is not, so the estimate errs high.
- **Sequential, not sharded-parallel per source.** HF streaming is inherently
  sequential over a shard list. Parallelism comes from running the three sources
  concurrently and pooling the *cleaning* work, which is where the CPU goes.
- **Network is the bottleneck, not the 48 threads.** Expect the pool to idle.

## Failure modes anticipated

| Risk | Mitigation |
|---|---|
| Rate limiting / throttling | `HF_TOKEN` now set (Phase 0 finding) |
| Transient network drop mid-stream | retry with backoff; `progress.json` resumes |
| Disk fills during an unattended run | hard abort below a free-space floor |
| A source runs dry before budget | log it and stop that source cleanly |

## Success criteria

1. Three `clean/<source>/` dirs with gzipped shards.
2. A drop-count report per source, by reason.
3. Estimated tokens within ~10 % of each source's budget.
4. `df -h /` still comfortable.

---

# RESULTS — **COMPLETE** (2026-08-11), with one plan change

Ran in **33 minutes**, not the 8–24 h budgeted. The guide's estimate assumed
network-bound streaming; actual throughput was ~1.1M tokens/s sustained.

## Final corpus

| Source | Docs seen | Kept | Keep % | Tokens | Shards | On disk |
|---|---|---|---|---|---|---|
| case-law | 528,777 | 511,339 | 96.7 % | 2.251B | 34 | 2.6 GB |
| sec | 73,400 | 72,496 | 98.8 % | 2.001B | 30 | 2.0 GB |
| fineweb-edu | 872,000 | 837,755 | 96.1 % | 1.000B | 15 | 1.4 GB |
| **Total** | **1,474,177** | **1,421,590** | **96.4 %** | **5.25B** | **79** | **6.0 GB** |

Zero errors, zero retries, zero disk-floor aborts.

## The plan change: case-law is exhausted, the 10B corpus is not achievable

`HFforLegal/case-law` split `us` contains **541,371 documents / 9.1 GB total**.
We consumed essentially all of it and got **2.251B tokens against a 7B budget**.
The guide's 70 %-legal / 10B-token plan was never possible from this source — the
data does not exist. Confirmed exhaustion, not error: the stream ended normally
and the log has no exceptions or retries.

**Decision: keep the 5.25B corpus as-is** rather than backfilling from SEC
(~10B available) or fineweb-edu (~9B available). Rationale: the achieved mix is
*better balanced* for a legal/financial model than the top-up alternatives, which
would have made SEC dominant (61 %) and shifted the model's character from
legal-leaning to distinctly financial.

| | Planned | Achieved |
|---|---|---|
| legal | 70 % | **43 %** |
| financial | 20 % | **38 %** |
| web | 10 % | **19 %** |
| total tokens | 10B | **5.25B** |

## Consequences for training

`config.TOKEN_BUDGET_B = 5.25` now drives everything downstream:

```
steps/epoch   19,073 → 10,013
max steps     95,365 → 50,065   (still 5 epochs)
tokens seen   50.0B  → 26.2B
tok/param     397    → 209      (still ~10× Chinchilla)
Phase 5 time  ~107 h → ~56 h    (2.3 days, was 4.5)
electricity   170 kWh → 90 kWh
token files   20 GB  → 10.5 GB
```

209 tokens/parameter remains a strongly over-trained small model, which is the
regime we want for inference quality.

## Drop analysis

| Reason | case-law | sec | fineweb-edu |
|---|---|---|---|
| `too_short` | 14,190 (2.7 %) | 16 | 29,859 (3.4 %) |
| `no_lines_left` | 1,881 | 1 | 74 |
| `ocr_garbage` | 1,349 | 63 | 4,300 |
| `non_english` | 18 | — | 12 |
| `empty` | — | 824 (1.1 %) | — |

The OCR filter, tightened from 0.03 → 0.02 during the trial run, fired **5,712
times** across the full corpus. At the original 0.03 it measured as a complete
no-op, so that tightening was worth doing.

Note the trial run's 41 % case-law drop rate did **not** hold at scale — it fell
to 3.3 %. The trial's first 400 documents were disproportionately short
procedural orders; the dataset as a whole is much healthier.

## Findings

1. **`tmux` and `screen` are both absent** and can't be installed without sudo.
   Long runs use `setsid nohup … > log 2>&1 < /dev/null &`, which survives
   terminal close and SSH drop identically. Install tmux if you want attachable
   sessions: `sudo apt install tmux`.
2. **Nested `mp.Pool` is impossible** — Pool workers are daemonic and daemons
   cannot have children. The per-source fan-out uses threads; only the cleaning
   work runs in process pools.
3. **Resume works.** The bounded trial consumed 400 docs/source; the full run
   correctly skipped them rather than reprocessing.
4. **case-law `progress.json` is marked `finished: true, exhausted: true`** so a
   re-run doesn't pointlessly re-stream 528k documents.

## Disk

```
$SLM_ROOT/clean/   6.0 GB   (79 gzipped shards)
free on /          114 GB
```

Ready for Phase 2 (dedup + decontamination).
