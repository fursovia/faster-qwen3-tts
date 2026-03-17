#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single model, single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Multiple models on one GPU:
    python examples/openai_server.py \\
        --model tts-1=Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --model tts-1-hd=Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
        --ref-audio voice.wav --ref-text "transcript" --language English

    # Load 3 replicas of the same model for concurrent requests:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --replicas 3 \\
        --ref-audio voice.wav --ref-text "transcript" --language English

    # Use gunicorn as the ASGI server:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "transcript" --language English \\
        --engine gunicorn --workers 1

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model pool – holds N replicas of a model behind an asyncio.Queue
# ---------------------------------------------------------------------------


class ModelPool:
    """Pool of model replicas for a single alias, enabling concurrent inference."""

    def __init__(self, alias: str, replicas: list, sample_rate: int):
        self.alias = alias
        self.sample_rate = sample_rate
        self.num_replicas = len(replicas)
        # asyncio.Queue acts as a semaphore: get() blocks when all are busy
        self._pool: asyncio.Queue = asyncio.Queue()
        for r in replicas:
            self._pool.put_nowait(r)

    async def acquire(self):
        """Wait for an available replica and return it."""
        return await self._pool.get()

    def release(self, model):
        """Return a replica to the pool."""
        self._pool.put_nowait(model)


# ---------------------------------------------------------------------------
# Configuration (set by main(), or via env vars for gunicorn workers)
# ---------------------------------------------------------------------------

_server_config = {
    "models": {},       # alias -> model_path
    "replicas": 1,      # number of replicas per model
    "voices": {},       # voice_name -> {ref_audio, ref_text, language, ...}
    "default_voice": None,
    "device": "cuda",
}

# ---------------------------------------------------------------------------
# Global state (populated by lifespan)
# ---------------------------------------------------------------------------

model_pools: dict[str, ModelPool] = {}   # alias -> ModelPool
default_model_alias: Optional[str] = None
voices: dict = {}
default_voice: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    """Load config from env vars (gunicorn workers) or module-level _server_config."""
    models_json = os.environ.get("_FQTTS_MODELS")
    if models_json:
        return {
            "models": json.loads(models_json),
            "replicas": int(os.environ.get("_FQTTS_REPLICAS", "1")),
            "voices": json.loads(os.environ.get("_FQTTS_VOICES", "{}")),
            "default_voice": os.environ.get("_FQTTS_DEFAULT_VOICE") or None,
            "device": os.environ.get("_FQTTS_DEVICE", "cuda"),
        }
    return _server_config


# ---------------------------------------------------------------------------
# Lifespan – load models on startup (works with both uvicorn and gunicorn)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pools, default_model_alias
    global voices, default_voice

    config = _load_config()
    voices = config["voices"]
    default_voice = config["default_voice"]
    n_replicas = config["replicas"]

    from faster_qwen3_tts import FasterQwen3TTS

    for alias, path in config["models"].items():
        replicas = []
        for i in range(n_replicas):
            tag = f"{alias} replica {i + 1}/{n_replicas}" if n_replicas > 1 else alias
            logger.info("Loading %s -> %s on %s ...", tag, path, config["device"])
            model = FasterQwen3TTS.from_pretrained(
                path, device=config["device"], dtype=torch.bfloat16,
            )
            replicas.append(model)
        pool = ModelPool(alias, replicas, replicas[0].sample_rate)
        model_pools[alias] = pool
        if default_model_alias is None:
            default_model_alias = alias

    total = sum(p.num_replicas for p in model_pools.values())
    logger.info(
        "Loaded %d model(s) with %d total replica(s): %s  (default: %s)",
        len(model_pools), total,
        {a: p.num_replicas for a, p in model_pools.items()},
        default_model_alias,
    )
    yield
    model_pools.clear()


app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model(model_name: str) -> ModelPool:
    """Return the ModelPool for the requested model name."""
    if model_name in model_pools:
        return model_pools[model_name]
    if default_model_alias:
        return model_pools[default_model_alias]
    raise HTTPException(
        status_code=400,
        detail=f"Model {model_name!r} not loaded. Available: {list(model_pools.keys())}",
    )


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    pool: ModelPool, voice_cfg: dict, text: str,
) -> AsyncGenerator[bytes, None]:
    """
    Acquire a replica from the pool, run generate_voice_clone_streaming in a
    background thread, yield raw PCM bytes, then release the replica.
    """
    model = await pool.acquire()
    try:
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def producer():
            try:
                for chunk, _sr, _timing in model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    non_streaming_mode=False,
                ):
                    q.put(chunk)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_DONE)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield _to_pcm16(item)
    finally:
        pool.release(model)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            alias: pool.num_replicas for alias, pool in model_pools.items()
        },
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "owned_by": "faster-qwen3-tts",
            }
            for alias in model_pools
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not model_pools:
        raise HTTPException(status_code=503, detail="No models loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    pool = resolve_model(req.model)
    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_running_loop()
        model = await pool.acquire()
        try:
            def _generate():
                return model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                )

            audio_arrays, sr = await loop.run_in_executor(None, _generate)
        finally:
            pool.release(model)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(pool.sample_rate)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(pool, voice_cfg, req.input):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_model_specs(raw_models: list[str]) -> dict:
    """Parse alias=path or bare path model specs into {alias: path} dict."""
    specs = {}
    for spec in raw_models:
        if "=" in spec:
            alias, path = spec.split("=", 1)
        else:
            path = spec
            # Derive alias from path: "Qwen/Qwen3-TTS-12Hz-0.6B-Base" -> "Qwen3-TTS-12Hz-0.6B-Base"
            alias = path.rsplit("/", 1)[-1] if "/" in path else path
        if alias in specs:
            print(f"ERROR: duplicate model alias {alias!r}", file=sys.stderr)
            sys.exit(1)
        specs[alias] = path
    return specs


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help=(
            "Model to load (repeatable). Format: alias=path or just path. "
            "Example: --model tts-1=Qwen/Qwen3-TTS-12Hz-0.6B-Base "
            "--model tts-1-hd=Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ),
    )
    p.add_argument(
        "--replicas",
        type=int,
        default=1,
        help=(
            "Number of replicas per model (default: 1). "
            "Each replica is an independent model copy, enabling concurrent inference."
        ),
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, ...) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--engine",
        choices=["uvicorn", "gunicorn"],
        default="uvicorn",
        help="ASGI server engine (default: uvicorn)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of gunicorn worker processes (default: 1). Each worker loads all models independently.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # Parse model specs
    raw_models = args.models or [
        os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ]
    model_specs = _parse_model_specs(raw_models)

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices_dict = json.load(f)
        if not voices_dict:
            print("ERROR: voices config is empty", file=sys.stderr)
            sys.exit(1)
        default_voice_name = next(iter(voices_dict))
        logger.info("Loaded %d voice(s) from %s", len(voices_dict), args.voices)
    elif args.ref_audio:
        voices_dict = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice_name = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info(
        "Model(s) to load: %s (%d replica(s) each)", model_specs, args.replicas,
    )

    # Store config for lifespan handler
    _server_config.update({
        "models": model_specs,
        "replicas": args.replicas,
        "voices": voices_dict,
        "default_voice": default_voice_name,
        "device": args.device,
    })

    if args.engine == "gunicorn":
        # Pass config via env vars so gunicorn workers can read it
        os.environ["_FQTTS_MODELS"] = json.dumps(model_specs)
        os.environ["_FQTTS_REPLICAS"] = str(args.replicas)
        os.environ["_FQTTS_VOICES"] = json.dumps(voices_dict)
        os.environ["_FQTTS_DEFAULT_VOICE"] = default_voice_name or ""
        os.environ["_FQTTS_DEVICE"] = args.device

        server_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(server_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            server_dir + os.pathsep
            + project_root + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        cmd = [
            sys.executable, "-m", "gunicorn",
            "openai_server:app",
            "-k", "uvicorn.workers.UvicornWorker",
            "--workers", str(args.workers),
            "--bind", f"{args.host}:{args.port}",
        ]
        logger.info("Starting gunicorn: %s", " ".join(cmd))
        sys.exit(subprocess.call(cmd, env=env))
    else:
        logger.info("Server listening on http://%s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
