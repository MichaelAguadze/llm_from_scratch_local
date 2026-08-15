# Phase 6 — Local inference

**Goal:** serve the Phase 5 model from this machine, with a UI, and measure what
it actually does per second.

```bash
# pin inference to GPU 0, leaving 1-3 free for the next training run
CUDA_VISIBLE_DEVICES=0 uvicorn slm.serve:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000` → prompt box, sampling controls, streamed completion,
and a "what this is" panel. One origin, so no CORS and no cold start.

## The artifact

```
$SLM_ROOT/checkpoints/base/
├── config.json            LlamaConfig: 12L/768d/12h, vocab 16384, ctx 1024, RoPE
├── model.safetensors      503 MB, bf16
├── generation_config.json bos/eos/pad ids
├── tokenizer.json         byte-level BPE, 16,384 merges + vocab
├── tokenizer_config.json
└── training.json          step, epoch, tokens seen, per-source val loss (added here)
```

`training.json` is new in this phase: `/health` reports what the weights were
trained to, and the UI shows it. Extracted once from `last.pt` / `best.pt` so the
server never has to open a 1.5 GB checkpoint.

There is no `special_tokens_map.json` — transformers 5.x carries the specials in
`tokenizer_config.json`, and the guide's file listing predates that.

## Prompt format — the guide is wrong here

The guide says prompt = `<|bos|>` + your text. **Do not.** Phase 4's packer wrote
`document tokens + <|eos|>` with `add_special_tokens=False`, so id 0 (`<|bos|>`)
never appeared in 17.7B training tokens. Its embedding is untouched initialisation.
Measured on one legal sentence:

| Prefix | NLL | ppl |
|---|---|---|
| none | 1.7948 | 6.02 |
| `<|eos|>` | 1.9022 | 6.70 |
| `<|bos|>` | 2.1812 | 8.86 |

Embedding norms tell the same story: `<|bos|>` 0.863, `<|pad|>` 0.850,
`<|unk|>` 0.862 — all near their init — against a trained-token mean of 1.165.
`<|eos|>` sits at 1.034 because it is the only special the corpus ever contained.

So: **send raw text, no special tokens.** The `document_start` flag prepends
`<|eos|>` instead of `<|bos|>`, which is the only in-distribution way to say
"a fresh document begins here."

## API

| Route | Purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /health` | model dir, device, dtype, params, context, training.json |
| `POST /generate` | blocking; returns text + tokens + seconds + tokens_per_s |
| `POST /generate/stream` | SSE, one `data:` frame per chunk, `{done:true}` last |

Body: `prompt`, `max_new_tokens`, `min_new_tokens`, `temperature`, `top_p`,
`top_k`, `repetition_penalty`, `seed`, `ascii_only`, `document_start`.

```bash
curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"The Court held that","max_new_tokens":256,"seed":7}'
```

A `threading.Lock` serialises generation: one GPU, one request at a time.
Concurrent callers queue rather than fight over the device.

## Gotchas baked in

1. **`min_new_tokens`** (default 16). A base model will emit EOS as its first
   token and hand back an empty string, especially at low temperature.
2. **Non-ASCII suppression** (default on). 186 of 16,384 ids decode to lone
   UTF-8 continuation bytes; the corpus was filtered to ≥90 % ASCII, so these
   are almost always unicode confetti. `ascii_only: false` re-enables them.
   Note this suppresses *all* non-ASCII output, curly quotes included.
3. **`top_k` 50 and `repetition_penalty` 1.1** as defaults, against the
   definitional looping a 126M base model falls into on open-domain prompts.
4. **`use_cache`** — training wrote `use_cache: false` into
   `generation_config.json`; the server flips it back. Worth less than expected
   at this scale (127.6 vs 131.5 tok/s over 256 tokens — decoding is
   kernel-launch-bound, not recompute-bound) but correct as context grows.
5. **Prompt truncation, not rejection.** Context is 1024 with no RoPE scaling,
   so an over-long prompt loses its oldest tokens and the response reports
   `truncated_prompt_tokens`.

## Measured

RTX A6000, bf16, batch 1, 256 new tokens, greedy-equivalent settings:

| Metric | Value |
|---|---|
| tokens/s (steady state) | **113–116 tok/s** |
| tokens/s (first request after load) | 92 (CUDA warmup) |
| 256-token completion | **2.21 s** |
| time to first token (SSE) | **15–25 ms** |
| model load → serving | 1.1 s |
| VRAM resident | **628 MiB** of 49,140 |
| CPU fp32, 24 threads | 25.9 tok/s |

628 MiB means inference costs nothing worth reclaiming — the other three GPUs
are free for the next run, and so is 98 % of GPU 0. CPU at 25.9 tok/s is faster
than reading speed, so the GPU is a convenience here, not a requirement.

## Behaviour, honestly

```
prompt:  IN THE UNITED STATES DISTRICT COURT FOR THE
output:  WESTERN DISTRICT OF NEBRASKA ... (a) As to the judgment of July 29,
         1995 in favor of the plaintiff, and the order thereon, is reversed.
         The case is remanded for further proceedings consistent with this
         opinion.

prompt:  What is the capital of France?
output:  In the French Revolution of 1789, the monarch was crowned as king on
         December 20. The monarchy was ruled by King Louis XIV who had ruled
         France since 1795.
```

The first is the model at its best: it knows the shape of a caption, a
disposition line, a remand. The second is the honest failure mode — it is a
**completer, not an assistant**, so a question gets continued rather than
answered, and the history is confidently wrong (Louis XIV did not rule from
1795). At 126M parameters style is learned far better than fact. The UI says
this in plain language rather than letting a visitor discover it.

## Always-on (optional, not installed)

```ini
# ~/.config/systemd/user/slm.service
[Unit]
Description=slm-125m inference server
[Service]
WorkingDirectory=/home/michael/Desktop/kasa/llm/llm_from_scratch_local
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/michael/Desktop/kasa/llm/llm_from_scratch_local/.venv/bin/uvicorn \
          slm.serve:app --host 127.0.0.1 --port 8000
Restart=on-failure
[Install]
WantedBy=default.target
```

`systemctl --user enable --now slm`, then `loginctl enable-linger michael` if it
should survive logout.

## Public access (Cloudflare Quick Tunnel)

This machine is behind NAT on `192.168.2.17` with ufw active, so port forwarding
is not the route. `cloudflared` dials out and Cloudflare terminates TLS:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
# -> https://<random-words>.trycloudflare.com, printed in the log
```

Run detached, logging to `$SLM_ROOT/logs/06-tunnel.log`. No account, no domain,
no inbound firewall change. **The URL is ephemeral** — a new one is issued every
time `cloudflared` restarts, and the tunnel dies with the process.

Measured through the tunnel (vs local in the table above):

| Metric | Local | Via tunnel |
|---|---|---|
| `GET /` | — | 237 ms |
| 128-token completion | 1.11 s | 1.11 s |
| time to first token | 15–25 ms | **106 ms** |
| tokens/s | 115 | 115 |

Streaming survives the hop: the `X-Accel-Buffering: no` header keeps Cloudflare
from buffering the SSE stream, so tokens still arrive one at a time. The extra
~85 ms of TTFT is round-trip to the edge and back, not throughput loss.

To take it down: `pkill -f "cloudflared tunnel"`. The server keeps running on
localhost.

## The public site (GitHub Pages)

`site/` is a three-file static site — `index.html`, `data.js`, `config.js` — with
no build step, no dependencies and no external requests. `.github/workflows/pages.yml`
publishes it on every push to `main` that touches `site/`.

It tells the project's story (corpus → tokenizer → training → results), and its
demo panel talks to the tunnel above. Two things make that work:

**CORS.** The Pages origin is not the tunnel origin, so the browser preflights
every `/generate` call. `serve.py` now mounts `CORSMiddleware` with
`allow_origins=["*"]` — wide open on purpose, because the server is already
unauthenticated and public, so an allowlist would gate nothing the URL does not
already gate. Verified through the tunnel with an explicit `Origin:` header:
preflight `200` with `access-control-allow-origin: *`, then the real POST the same.

**The URL moves.** `site/config.js` holds `window.SLM_API` as a one-line edit,
and the demo panel's "endpoint" link lets any visitor point their own browser at
a different backend (kept in their `localStorage`). Every time `cloudflared`
restarts, edit that line and push.

**Offline is the default state, so it is designed for.** The page polls
`/health` on load and every 60 s. Live → a green badge with the real device,
dtype and parameter count read from the server. Not live → the Generate button
disables, an explanation replaces it ("this is a tunnel to a desktop, not a
hosted service"), and the preset prompts replay **recorded real completions**
from these weights instead. A visitor who arrives while the desktop is off still
sees what the model does; nothing is faked as live.

The charts (validation perplexity per source, training loss) are hand-rolled SVG
built from `logs/05-pretrain.log` at Phase 6 — 67 eval points and 112 downsampled
loss points, inlined into `data.js`. Palette validated for colour-blind
separation in both light and dark modes; each line is direct-labelled with its
final value, and the perplexity chart carries a table view for screen readers.

One thing the charts made visible that the log did not: **the training loss
crosses every epoch boundary smoothly.** No step down at 6,743 / 13,486 / 20,229
/ 26,972. That is the per-epoch reshuffle doing its job — a drop as the model
began a second pass over the same windows would have been memorisation.

## Security

The server has **no auth and no rate limit** — a deliberate choice for this
deployment, not an oversight. While the tunnel is up, anyone who has the URL can
run unlimited generation on GPU 0. The realistic exposure:

- Random URL, unlisted and unindexed, so discovery is the only real barrier.
  It holds until the link is posted somewhere crawlable.
- A single `threading.Lock` serialises generation, so the failure mode of abuse
  is a queue, not a meltdown; `max_new_tokens` is capped at 1023 by the schema.
- No filesystem or shell reachable from any route. The blast radius is GPU time
  and whatever the model says.

Kill the tunnel when the demo is over rather than leaving it up indefinitely.
If it needs to live longer, the upgrade is a named tunnel with Cloudflare Access
in front, or a bearer token checked in `serve.py`.

## Still optional

- **Publish** `base/` to `your-user/slm-125m-base` — a backup of a run that cost
  20 GPU-hours to produce.
- **GGUF export** (Q4/Q8) via llama.cpp for laptop inference.
- **Instruction tuning** — the tokenizer already reserves `<|user|>`,
  `<|assistant|>`, `<|system|>`, unused so far. That is the actual next step if
  the goal is something you can talk to.
