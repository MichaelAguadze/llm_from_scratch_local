"""Phase 6: local inference server for the 125M base model.

    CUDA_VISIBLE_DEVICES=0 uvicorn slm.serve:app --host 127.0.0.1 --port 8000

FastAPI, one process, model loaded once at startup:

    GET  /               the single-page UI
    GET  /health         model path, device, dtype, what it was trained to
    POST /generate       {prompt, ...} -> {generated, tokens, seconds, tokens_per_s}
    POST /generate/stream  same body, Server-Sent Events, token by token

There is no auth here. Bind 127.0.0.1 (see docs/06-deploy.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config as C

MODEL_DIR = Path(os.environ.get("SLM_MODEL_DIR", C.BASE_MODEL_DIR))
STATIC_DIR = Path(__file__).parent / "static"

# The corpus was filtered to >=90% ASCII, so every token that decodes to a
# non-ASCII string is a lone UTF-8 continuation byte the model almost never
# meant to emit. Suppressing them is the difference between clean output and
# occasional unicode confetti. Computed once at startup; 186 of 16,384 ids.
_state: dict = {}


def _nonascii_token_ids(tok) -> list[int]:
    return [i for i in range(len(tok)) if any(ord(c) > 127 for c in tok.decode([i]))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=dtype)
    model.to(device).eval()
    # Training wrote use_cache=False into generation_config.json; flip it back.
    # Measured, it is worth little at this scale (127.6 vs 131.5 tok/s over 256
    # tokens) because decoding 126M params at ctx 1024 is kernel-launch-bound,
    # not prefix-recompute-bound. It is still the correct setting: the cost of
    # recomputation grows with context, and this model has none to spare.
    model.config.use_cache = True
    model.generation_config.use_cache = True

    meta_path = MODEL_DIR / "training.json"
    _state.update(
        tok=tok,
        model=model,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        suppress=_nonascii_token_ids(tok),
        params=sum(p.numel() for p in model.parameters()),
        training=json.loads(meta_path.read_text()) if meta_path.exists() else {},
        lock=threading.Lock(),  # one GPU, one generation at a time
        load_seconds=round(time.time() - t0, 2),
    )
    yield
    _state.clear()


app = FastAPI(title="slm-125m-base", lifespan=lifespan)


class GenRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(256, ge=1, le=C.SEQ_LEN - 1)
    min_new_tokens: int = Field(16, ge=0)
    temperature: float = Field(0.8, gt=0, le=2.0)
    top_p: float = Field(0.95, gt=0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    seed: int | None = None
    ascii_only: bool = True
    # The packer wrote `document + <|eos|>` and never a BOS, so <|bos|> is an
    # untrained embedding at inference. A leading <|eos|> is the only in-
    # distribution way to say "start of a fresh document". See docs/06-deploy.md.
    document_start: bool = False


def _prepare(req: GenRequest):
    tok, model = _state["tok"], _state["model"]
    ids = tok(req.prompt, add_special_tokens=False)["input_ids"]
    if req.document_start:
        ids = [tok.eos_token_id] + ids
    # Context is 1024 with no RoPE scaling; keep room for the completion by
    # dropping the oldest tokens rather than failing the request.
    budget = C.SEQ_LEN - req.max_new_tokens
    truncated = max(0, len(ids) - budget)
    ids = ids[-budget:] if budget > 0 else ids[-1:]
    input_ids = torch.tensor([ids], device=_state["device"])

    kwargs = dict(
        max_new_tokens=req.max_new_tokens,
        # A base model under a low temperature will happily emit EOS as its
        # first token and return an empty completion. This is the floor.
        min_new_tokens=min(req.min_new_tokens, req.max_new_tokens),
        do_sample=req.temperature > 0,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k or None,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=_state["tok"].eos_token_id,
        pad_token_id=_state["tok"].pad_token_id,
        use_cache=True,
    )
    if req.ascii_only:
        kwargs["suppress_tokens"] = _state["suppress"]
    return input_ids, kwargs, truncated


@app.get("/health")
def health() -> dict:
    if not _state:
        raise HTTPException(503, "model not loaded")
    return {
        "model_dir": str(MODEL_DIR),
        "device": _state["device"],
        "dtype": _state["dtype"],
        "parameters": _state["params"],
        "context": C.SEQ_LEN,
        "vocab_size": C.VOCAB_SIZE,
        "suppressed_tokens": len(_state["suppress"]),
        "load_seconds": _state["load_seconds"],
        "training": _state["training"],
    }


@app.post("/generate")
def generate(req: GenRequest) -> dict:
    if not _state:
        raise HTTPException(503, "model not loaded")
    if req.seed is not None:
        torch.manual_seed(req.seed)
    input_ids, kwargs, truncated = _prepare(req)

    with _state["lock"]:
        t0 = time.time()
        with torch.no_grad():
            out = _state["model"].generate(input_ids, **kwargs)
        if _state["device"] == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    new = out[0][input_ids.shape[1] :]
    text = _state["tok"].decode(new, skip_special_tokens=True)
    return {
        "generated": text,
        "prompt_tokens": int(input_ids.shape[1]),
        "truncated_prompt_tokens": truncated,
        "tokens": int(new.numel()),
        "seconds": round(dt, 3),
        "tokens_per_s": round(new.numel() / dt, 1),
    }


@app.post("/generate/stream")
async def generate_stream(req: GenRequest) -> StreamingResponse:
    if not _state:
        raise HTTPException(503, "model not loaded")
    if req.seed is not None:
        torch.manual_seed(req.seed)
    from transformers import TextIteratorStreamer

    input_ids, kwargs, _ = _prepare(req)
    streamer = TextIteratorStreamer(
        _state["tok"], skip_prompt=True, skip_special_tokens=True
    )

    def run() -> None:
        with _state["lock"]:
            with torch.no_grad():
                _state["model"].generate(input_ids, streamer=streamer, **kwargs)

    threading.Thread(target=run, daemon=True).start()

    async def events():
        loop = asyncio.get_running_loop()
        it = iter(streamer)
        t0 = time.time()
        n = 0
        while True:
            # next() on the streamer blocks; keep the event loop free so the
            # client sees each chunk as it lands.
            chunk = await loop.run_in_executor(None, lambda: next(it, None))
            if chunk is None:
                break
            n += 1
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        dt = time.time() - t0
        done = {"done": True, "chunks": n, "seconds": round(dt, 3)}
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
