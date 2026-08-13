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

## Security

The server has **no auth and no rate limit**. It binds `127.0.0.1` deliberately.
Bind `0.0.0.0` only for a trusted LAN; anything beyond that wants a tunnel with
auth in front. Nothing about this model is dangerous, but an open port that
executes arbitrary GPU work on request is.

## Still optional

- **Publish** `base/` to `your-user/slm-125m-base` — a backup of a run that cost
  20 GPU-hours to produce.
- **GGUF export** (Q4/Q8) via llama.cpp for laptop inference.
- **Instruction tuning** — the tokenizer already reserves `<|user|>`,
  `<|assistant|>`, `<|system|>`, unused so far. That is the actual next step if
  the goal is something you can talk to.
