# OpenAI-Compatible TTS Server Guide

This guide explains how to start and use the OpenAI-compatible text-to-speech API server provided by `faster-qwen3-tts`. The server exposes a `POST /v1/audio/speech` endpoint that is compatible with OpenAI's TTS API, making it easy to integrate with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **Python 3.10+**
- A **reference audio file** (`.wav`) for voice cloning, along with its transcript

## Installation

Install the package with the `demo` extra, which pulls in FastAPI, Uvicorn, and Gunicorn:

```bash
# Using uv (recommended)
uv pip install --system "faster-qwen3-tts[demo]"

# Or using pip
pip install "faster-qwen3-tts[demo]"
```

### Optional dependencies

- **MP3 output**: Requires `pydub` and `ffmpeg`:
  ```bash
  uv pip install --system pydub
  apt-get install -y ffmpeg  # or your system's package manager
  ```

## Starting the Server

### Single voice (quickstart)

Provide a reference audio file and its transcript to start with one default voice:

```bash
python examples/openai_server.py \
    --ref-audio voice.wav \
    --ref-text "The exact transcript of the reference audio." \
    --language English
```

This registers a single voice named `"default"`. Any voice name requested by clients will fall back to it.

### Multiple named voices

Create a JSON config file mapping voice names to their reference audio, transcript, and language:

```json
{
    "alloy": {
        "ref_audio": "voices/alloy.wav",
        "ref_text": "This is the transcript for the alloy voice reference.",
        "language": "English"
    },
    "echo": {
        "ref_audio": "voices/echo.wav",
        "ref_text": "This is the transcript for the echo voice reference.",
        "language": "English",
        "chunk_size": 12
    },
    "narrator": {
        "ref_audio": "voices/narrator.wav",
        "ref_text": "This is the transcript for the narrator voice reference.",
        "language": "French"
    }
}
```

Then start the server with:

```bash
python examples/openai_server.py --voices voices.json
```

The first voice in the JSON file becomes the default fallback voice.

### Multiple models on one GPU

You can load multiple models simultaneously and route requests based on the `model` field. Use `--model` multiple times with `alias=path` format:

```bash
python examples/openai_server.py \
    --model tts-1=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --model tts-1-hd=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio voice.wav \
    --ref-text "Transcript" \
    --language English
```

Clients select the model via the `model` field in the request body. If the requested model name doesn't match any alias, the first loaded model is used as the fallback.

You can also use bare paths (the alias is derived from the last path component):

```bash
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio voice.wav --ref-text "Transcript" --language English
```

This creates aliases `Qwen3-TTS-12Hz-0.6B-Base` and `Qwen3-TTS-12Hz-1.7B-Base`.

Use `GET /v1/models` to list loaded models:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

### Specifying a single model

When only one model is needed, `--model` accepts a plain path (backward compatible):

```bash
# Use the 0.6B model for lower VRAM usage
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio voice.wav \
    --ref-text "Transcript" \
    --language English
```

If `--model` is omitted entirely, the default is `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (overridable via `QWEN_TTS_MODEL` env var).

### Replicas for concurrent requests

By default, the server loads one copy of each model and serves requests sequentially. To handle multiple requests in parallel, use `--replicas` to load N independent copies of each model:

```bash
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --replicas 3 \
    --ref-audio voice.wav --ref-text "Transcript" --language English
```

This loads 3 independent copies of the model. Up to 3 requests can be processed concurrently — each request acquires a replica from the pool and releases it when done. Additional requests queue up automatically.

Replicas work with multiple models too:

```bash
python examples/openai_server.py \
    --model tts-1=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --model tts-1-hd=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --replicas 2 \
    --ref-audio voice.wav --ref-text "Transcript" --language English
```

This loads 2 copies of each model (4 total), allowing 2 concurrent requests per model.

**VRAM note**: Each replica consumes additional GPU memory. For the 1.7B model, expect ~3.5 GB per replica. Use `nvidia-smi` to monitor.

### Using gunicorn (production)

For production deployments, use gunicorn as the ASGI server instead of uvicorn. Gunicorn provides process management, graceful restarts, and better signal handling:

```bash
python examples/openai_server.py \
    --ref-audio voice.wav --ref-text "Transcript" --language English \
    --engine gunicorn
```

With multiple models:

```bash
python examples/openai_server.py \
    --model tts-1=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --model tts-1-hd=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio voice.wav --ref-text "Transcript" --language English \
    --engine gunicorn --workers 1
```

Each gunicorn worker loads all specified models independently. For GPU inference, `--workers 1` (the default) is recommended to avoid GPU memory contention.

## Server Options Reference

| Flag | Default | Env Variable | Description |
|---|---|---|---|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN_TTS_MODEL` | Model to load. Repeatable with `alias=path` format for multi-model. |
| `--replicas` | `1` | _(none)_ | Number of independent copies per model for concurrent inference |
| `--voices` | _(none)_ | `QWEN_TTS_VOICES` | Path to a JSON file mapping voice names to configs |
| `--ref-audio` | _(none)_ | `QWEN_TTS_REF_AUDIO` | Reference audio file (used when `--voices` is not set) |
| `--ref-text` | `""` | `QWEN_TTS_REF_TEXT` | Transcript of the reference audio |
| `--language` | `Auto` | `QWEN_TTS_LANGUAGE` | Target language (e.g. `English`, `French`, `Auto`) |
| `--host` | `0.0.0.0` | _(none)_ | Bind address |
| `--port` | `8000` | _(none)_ | Bind port |
| `--device` | `cuda` | _(none)_ | Torch device (`cuda` or `cpu`) |
| `--engine` | `uvicorn` | _(none)_ | ASGI server: `uvicorn` or `gunicorn` |
| `--workers` | `1` | _(none)_ | Gunicorn worker count (each worker loads all models) |

All flags can also be set via the corresponding environment variable. For example:

```bash
export QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export QWEN_TTS_REF_AUDIO=voice.wav
export QWEN_TTS_REF_TEXT="Transcript of the reference audio."
export QWEN_TTS_LANGUAGE=English

python examples/openai_server.py
```

## API Endpoints

### Health Check

```
GET /health
```

Returns:
```json
{"status": "ok", "models_loaded": {"tts-1": 3, "tts-1-hd": 3}}
```

The values indicate the number of replicas per model.

### List Models

```
GET /v1/models
```

Returns an OpenAI-compatible model list:
```json
{
    "object": "list",
    "data": [
        {"id": "tts-1", "object": "model", "owned_by": "faster-qwen3-tts"},
        {"id": "tts-1-hd", "object": "model", "owned_by": "faster-qwen3-tts"}
    ]
}
```

### Generate Speech

```
POST /v1/audio/speech
```

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"tts-1"` | Model alias to use for generation (must match a loaded model, or falls back to default) |
| `input` | string | _(required)_ | The text to synthesize |
| `voice` | string | `"alloy"` | Voice name (must match a configured voice, or falls back to default) |
| `response_format` | string | `"wav"` | Output format: `wav`, `pcm`, or `mp3` |
| `speed` | float | `1.0` | Accepted for compatibility but not yet applied |

**Response**: Audio bytes in the requested format.

- **`wav`** and **`pcm`**: Streamed as chunks are generated (low latency)
- **`mp3`**: Generated fully before encoding (requires `pydub` + `ffmpeg`)

## Usage Examples

### curl

```bash
# Generate a WAV file
curl -s http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "tts-1", "input": "Hello, world!", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav

# Use a specific model (when multiple are loaded)
curl -s http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "tts-1-hd", "input": "Hello, world!", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav

# Generate raw PCM (16-bit, 24kHz, mono)
curl -s http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "response_format": "pcm"}' \
    --output speech.pcm

# Generate MP3 (requires pydub + ffmpeg on the server)
curl -s http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "response_format": "mp3"}' \
    --output speech.mp3
```

### Python (OpenAI SDK)

Since the endpoint is OpenAI-compatible, you can use the official OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello from faster-qwen3-tts!",
    response_format="wav",
)
response.stream_to_file("speech.wav")
```

### Python (requests)

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Hello from faster-qwen3-tts!",
        "voice": "alloy",
        "response_format": "wav",
    },
)
with open("speech.wav", "wb") as f:
    f.write(resp.content)
```

## Audio Output Details

- **Sample rate**: 24,000 Hz (24 kHz)
- **Channels**: 1 (mono)
- **Bit depth**: 16-bit signed integer (PCM)
- **WAV streaming**: The WAV header is sent first with an unknown data length (`0xFFFFFFFF`), followed by PCM chunks as they are generated. Most players handle this correctly.

## Concurrency

With `--replicas 1` (default), requests are served one at a time. Each model instance owns its own CUDA graphs and static KV caches, so concurrent use of a single instance is not safe.

With `--replicas N`, N independent copies of each model are loaded into a pool. Up to N requests can run concurrently, each using its own replica. When all replicas are busy, additional requests wait in a FIFO queue until a replica becomes available.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SoX could not be found!` warning on startup | `sox` is not installed. This is a non-critical warning from the `qwen-tts` dependency. | Install SoX (`apt-get install -y sox`) or ignore the warning -- the server works without it. |
| `torchvision` `RuntimeError: operator torchvision::nms does not exist` | Incompatible `torchvision` version for the installed `torch`. | Uninstall torchvision (`uv pip uninstall --system torchvision`) -- it is not required by this project. |
| `response_format='mp3' requires pydub` | MP3 encoding dependencies missing. | `uv pip install --system pydub` and install `ffmpeg`. |
| `Voice 'X' is not configured` (400 error) | Requested voice name doesn't match any configured voice and no default is set. | Use a voice name from your config, or ensure `--ref-audio` was provided for a default fallback. |
| `Model 'X' not loaded` (400 error) | Requested model alias doesn't match any loaded model. | Use `GET /v1/models` to check available aliases, or use a recognized alias in the `model` field. |
| Model download is slow or fails | First run downloads model weights from HuggingFace. | Ensure internet access and sufficient disk space (~3.5 GB for 1.7B, ~1.5 GB for 0.6B). You can also pre-download to a local path and pass it via `--model /path/to/model`. |
| GPU OOM with multiple models/replicas | Too many models or replicas for available VRAM. | Reduce `--replicas`, use smaller models (0.6B), reduce the number of loaded models, or use a GPU with more VRAM. |
