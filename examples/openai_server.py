#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts (single model per worker).

Start it with the Typer CLI below. The CLI always launches gunicorn:

    pip install "faster-qwen3-tts[demo]"

    python examples/openai_server.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio voice.wav \
        --ref-text "Reference transcription" \
        --language English

    python examples/openai_server.py --voices voices.json --workers 1

Clients can then call:

    curl -s http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \
        --output speech.wav
"""
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import numpy as np
import torch
import typer
import typer.core
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from faster_qwen3_tts import FasterQwen3TTS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SERVER_CONFIG_ENV_VAR = "_FQTTS_SERVER_CONFIG"


@dataclass(frozen=True)
class VoiceSpec:
    ref_audio: str
    ref_text: str = ""
    language: str = "Auto"
    chunk_size: int = 6
    xvec_only: bool = False
    append_silence: bool = True


@dataclass(frozen=True)
class PreparedVoice:
    spec: VoiceSpec
    voice_clone_prompt: dict[str, Any]


@dataclass
class LoadedModel:
    alias: str
    model: "FasterQwen3TTS"
    voices: dict[str, PreparedVoice]
    sample_rate: int
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def acquire(self) -> "LoadedModel":
        await self._lock.acquire()
        return self

    def release(self) -> None:
        self._lock.release()


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    voices: dict[str, VoiceSpec]
    default_voice: str | None
    device: str = "cuda"

    def to_json(self) -> str:
        payload = {
            "model_path": self.model_path,
            "voices": {name: asdict(spec) for name, spec in self.voices.items()},
            "default_voice": self.default_voice,
            "device": self.device,
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, raw_json: str) -> "ServerConfig":
        data = json.loads(raw_json)
        voices = {
            name: VoiceSpec(**voice_cfg)
            for name, voice_cfg in data["voices"].items()
        }
        return cls(
            model_path=data["model_path"],
            voices=voices,
            default_voice=data.get("default_voice"),
            device=data.get("device", "cuda"),
        )


@dataclass(frozen=True)
class ServerRuntime:
    config: ServerConfig
    loaded: LoadedModel


_in_process_config: ServerConfig | None = None
_runtime: ServerRuntime | None = None


def _set_server_config(config: ServerConfig) -> None:
    global _in_process_config
    _in_process_config = config
    os.environ[SERVER_CONFIG_ENV_VAR] = config.to_json()


def _load_server_config() -> ServerConfig:
    if raw_json := os.environ.get(SERVER_CONFIG_ENV_VAR):
        return ServerConfig.from_json(raw_json)
    if _in_process_config is not None:
        return _in_process_config
    raise RuntimeError(
        f"{SERVER_CONFIG_ENV_VAR} is not set. Start the server via "
        "'python examples/openai_server.py ...' or export the config before "
        "running gunicorn."
    )


def _runtime_or_503() -> ServerRuntime:
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Server startup is not complete")
    return _runtime


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm
    speed: float = 1.0            # accepted but not yet applied


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header. Use data_len=0xFFFFFFFF for streaming."""
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
    buf.write(
        struct.pack(
            "<IHHIIHH",
            16,
            1,
            n_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
        )
    )
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _resolve_ref_audio_path(raw_path: str, base_dir: Path | None) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Reference audio file not found: {path}")
    return str(path)


def _build_voice_spec(
    voice_name: str,
    raw_config: dict[str, Any],
    *,
    base_dir: Path | None,
) -> VoiceSpec:
    ref_audio = raw_config.get("ref_audio")
    if not ref_audio:
        raise ValueError(f"Voice {voice_name!r} is missing required field 'ref_audio'")

    chunk_size = int(raw_config.get("chunk_size", 12))
    if chunk_size < 1:
        raise ValueError(f"Voice {voice_name!r} has invalid chunk_size={chunk_size}")

    voice = VoiceSpec(
        ref_audio=_resolve_ref_audio_path(str(ref_audio), base_dir),
        ref_text=str(raw_config.get("ref_text", "")),
        language=str(raw_config.get("language", "Auto")),
        chunk_size=chunk_size,
        xvec_only=bool(raw_config.get("xvec_only", False)),
        append_silence=bool(raw_config.get("append_silence", True)),
    )
    if not voice.xvec_only and not voice.ref_text.strip():
        raise ValueError(
            f"Voice {voice_name!r} must define non-empty 'ref_text' unless xvec_only=true"
        )
    return voice


def _load_voice_specs_from_file(path: Path) -> tuple[dict[str, VoiceSpec], str]:
    with path.open() as handle:
        raw_voices = json.load(handle)

    if not isinstance(raw_voices, dict) or not raw_voices:
        raise ValueError("Voices config must be a non-empty JSON object")

    voices = {
        voice_name: _build_voice_spec(
            voice_name,
            voice_cfg,
            base_dir=path.parent,
        )
        for voice_name, voice_cfg in raw_voices.items()
    }
    return voices, next(iter(voices))


def _build_single_voice_config(
    ref_audio: Path,
    ref_text: str,
    language: str,
    xvec_only: bool,
) -> tuple[dict[str, VoiceSpec], str]:
    voice = VoiceSpec(
        ref_audio=_resolve_ref_audio_path(str(ref_audio), None),
        ref_text=ref_text,
        language=language,
        xvec_only=xvec_only,
    )
    if not voice.xvec_only and not voice.ref_text.strip():
        raise ValueError("--ref-text is required unless --xvec-only is enabled")
    return {"default": voice}, "default"


def _build_server_config(
    model_path: str,
    voices_path: Path | None,
    ref_audio: Path | None,
    ref_text: str,
    language: str,
    device: str,
    xvec_only: bool,
) -> ServerConfig:
    if voices_path and ref_audio:
        raise ValueError("Use either --voices or --ref-audio, not both")

    if voices_path is not None:
        voices, default_voice = _load_voice_specs_from_file(voices_path)
        logger.info("Loaded %d voice(s) from %s", len(voices), voices_path)
    elif ref_audio is not None:
        voices, default_voice = _build_single_voice_config(
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            xvec_only=xvec_only,
        )
        logger.info("Using single voice from --ref-audio: %s", ref_audio)
    else:
        raise ValueError("Provide --ref-audio <file> or --voices <config.json>")

    return ServerConfig(
        model_path=model_path,
        voices=voices,
        default_voice=default_voice,
        device=device,
    )


def _prepare_voice_prompt(
    model: "FasterQwen3TTS",
    voice_name: str,
    voice_spec: VoiceSpec,
) -> PreparedVoice:
    mode = "x-vector" if voice_spec.xvec_only else "ICL"
    logger.info(
        "Preparing voice %r from %s (%s mode)",
        voice_name,
        voice_spec.ref_audio,
        mode,
    )
    with torch.inference_mode():
        voice_clone_prompt, _ref_ids, _using_icl_mode = (
            model._resolve_voice_clone_prompt_from_reference(
                input_ids=[0],
                ref_audio=voice_spec.ref_audio,
                ref_text=voice_spec.ref_text,
                xvec_only=voice_spec.xvec_only,
                append_silence=voice_spec.append_silence,
            )
        )
    return PreparedVoice(spec=voice_spec, voice_clone_prompt=voice_clone_prompt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global _runtime

    config = _load_server_config()

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s ...", config.model_path, config.device)
    model = FasterQwen3TTS.from_pretrained(
        config.model_path,
        device=config.device,
        dtype=torch.bfloat16,
    )
    prepared_voices = {
        voice_name: _prepare_voice_prompt(model, voice_name, voice_spec)
        for voice_name, voice_spec in config.voices.items()
    }
    loaded = LoadedModel(
        alias=config.model_path,
        model=model,
        voices=prepared_voices,
        sample_rate=model.sample_rate,
    )

    _runtime = ServerRuntime(config=config, loaded=loaded)
    logger.info(
        "Model ready (%s), %d voice(s) loaded (default voice: %s)",
        config.model_path,
        len(config.voices),
        config.default_voice,
    )

    try:
        yield
    finally:
        _runtime = None


app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API", lifespan=lifespan)


def _get_loaded_model() -> LoadedModel:
    return _runtime_or_503().loaded


def _resolve_voice_name(voice_name: str) -> str:
    runtime = _runtime_or_503()
    if voice_name in runtime.config.voices:
        return voice_name
    if runtime.config.default_voice and runtime.config.default_voice in runtime.config.voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            runtime.config.default_voice,
        )
        return runtime.config.default_voice
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(runtime.config.voices)}"
        ),
    )


async def _stream_chunks(
    loaded: LoadedModel,
    voice: PreparedVoice,
    text: str,
) -> AsyncGenerator[bytes, None]:
    """
    Run the sync streaming generator in a background thread and yield PCM bytes.

    The thread is always joined before the model lock is released, so a
    cancelled request cannot race with a later request on the same model instance.
    """
    result_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()

    def producer() -> None:
        try:
            for chunk, _sr, _timing in loaded.model.generate_voice_clone_streaming(
                text=text,
                language=voice.spec.language,
                ref_text=voice.spec.ref_text,
                chunk_size=voice.spec.chunk_size,
                xvec_only=voice.spec.xvec_only,
                append_silence=voice.spec.append_silence,
                non_streaming_mode=False,
                voice_clone_prompt=voice.voice_clone_prompt,
            ):
                result_queue.put(chunk)
        except BaseException as exc:
            result_queue.put(exc)
        finally:
            result_queue.put(done_sentinel)

    worker = threading.Thread(
        target=producer,
        daemon=True,
        name="faster-qwen3-tts-stream",
    )
    worker.start()

    loop = asyncio.get_running_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, result_queue.get)
            if item is done_sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield _to_pcm16(item)
    finally:
        await asyncio.shield(loop.run_in_executor(None, worker.join))


@app.get("/health")
async def health():
    runtime = _runtime_or_503()
    return {
        "status": "ok",
        "model": runtime.config.model_path,
        "voices_loaded": list(runtime.config.voices),
        "default_voice": runtime.config.default_voice,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    runtime = _runtime_or_503()
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.loaded.alias,
                "object": "model",
                "owned_by": "faster-qwen3-tts",
            }
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    fmt = req.response_format.lower()
    content_types = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    if fmt not in content_types:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm",
        )

    loaded = _get_loaded_model()
    voice_name = _resolve_voice_name(req.voice)

    async def audio_stream() -> AsyncGenerator[bytes, None]:
        await loaded.acquire()
        try:
            prepared_voice = loaded.voices[voice_name]
            if fmt == "wav":
                yield _wav_header(loaded.sample_rate)
            async for raw_chunk in _stream_chunks(loaded, prepared_voice, req.input):
                yield raw_chunk
        finally:
            loaded.release()

    return StreamingResponse(audio_stream(), media_type=content_types[fmt])


def _gunicorn_command(host: str, port: int, workers: int) -> list[str]:
    project_root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        "-m",
        "gunicorn",
        "--chdir",
        str(project_root),
        "examples.openai_server:app",
        "-k",
        "uvicorn.workers.UvicornWorker",
        "--workers",
        str(workers),
        "--bind",
        f"{host}:{port}",
    ]


def main(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        envvar="QWEN_TTS_MODEL",
        help="HuggingFace model ID or local path. Default: " + DEFAULT_MODEL_ID,
    ),
    voices: Optional[Path] = typer.Option(
        None,
        "--voices",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        envvar="QWEN_TTS_VOICES",
        help=(
            "JSON file mapping voice names to {ref_audio, ref_text, language, "
            "chunk_size, xvec_only, append_silence}."
        ),
    ),
    ref_audio: Optional[Path] = typer.Option(
        None,
        "--ref-audio",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        envvar="QWEN_TTS_REF_AUDIO",
        help="Reference audio file when --voices is not used.",
    ),
    ref_text: str = typer.Option(
        "",
        "--ref-text",
        envvar="QWEN_TTS_REF_TEXT",
        help="Transcript of --ref-audio. Required for ICL mode.",
    ),
    language: str = typer.Option(
        "Auto",
        "--language",
        envvar="QWEN_TTS_LANGUAGE",
        help="Target language when --voices is not used.",
    ),
    xvec_only: bool = typer.Option(
        False,
        "--xvec-only",
        help="Use only the speaker embedding for the single default voice.",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Bind host for gunicorn.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        min=1,
        max=65535,
        help="Bind port for gunicorn.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Torch device for model loading.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help=(
            "Gunicorn worker processes. Each worker loads the model, "
            "so values > 1 multiply VRAM usage."
        ),
    ),
) -> None:
    model_path = model or DEFAULT_MODEL_ID

    try:
        config = _build_server_config(
            model_path=model_path,
            voices_path=voices,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            device=device,
            xvec_only=xvec_only,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _set_server_config(config)
    logger.info("Model to load: %s", config.model_path)
    if workers > 1 and device.startswith("cuda"):
        logger.warning(
            "workers=%d will load the model in each gunicorn worker, "
            "multiplying GPU memory usage.",
            workers,
        )

    cmd = _gunicorn_command(host=host, port=port, workers=workers)
    logger.info("Starting gunicorn: %s", " ".join(cmd))
    raise typer.Exit(subprocess.call(cmd, env=os.environ.copy()))


if __name__ == "__main__":
    typer.run(main)