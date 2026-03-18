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

## Starting the Server

### Multiple named voices (recommended)

Create a `voices.json` file mapping voice names to their reference audio and settings:

```json
{
  "alloy": {
    "ref_audio": "../ref_audio.wav",
    "ref_text": "The exact transcript of the reference audio.",
    "language": "English",
    "chunk_size": 12,
    "xvec_only": false,
    "append_silence": true
  },
  "echo": {
    "ref_audio": "../ref_audio_2.wav",
    "language": "English",
    "chunk_size": 12,
    "xvec_only": true,
    "append_silence": true
  }
}
```

Paths in `ref_audio` are resolved relative to the directory containing the JSON file. When `xvec_only` is `true`, `ref_text` can be omitted.

Start the server:

```bash
python examples/openai_server.py --voices examples/voices.json
```

The first voice in the JSON file becomes the default fallback voice.

### Single voice (quickstart)

Provide a reference audio file and its transcript to start with one default voice:

```bash
python examples/openai_server.py \
    --ref-audio voice.wav \
    --ref-text "The exact transcript of the reference audio." \
    --language English
```

This registers a single voice named `"default"`. Any voice name requested by clients will fall back to it.

### Choosing a model

By default, the server loads `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Use `--model` to override:

```bash
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --voices examples/voices.json
```

## Server Options Reference

| Flag | Default | Env Variable | Description |
|---|---|---|---|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN_TTS_MODEL` | HuggingFace model ID or local path |
| `--voices` | _(none)_ | `QWEN_TTS_VOICES` | Path to a JSON file mapping voice names to configs |
| `--ref-audio` | _(none)_ | `QWEN_TTS_REF_AUDIO` | Reference audio file (used when `--voices` is not set) |
| `--ref-text` | `""` | `QWEN_TTS_REF_TEXT` | Transcript of the reference audio |
| `--language` | `Auto` | `QWEN_TTS_LANGUAGE` | Target language (e.g. `English`, `French`, `Auto`) |
| `--xvec-only` | `false` | _(none)_ | Use only the speaker embedding for the single default voice |
| `--host` | `0.0.0.0` | _(none)_ | Bind address |
| `--port` | `8000` | _(none)_ | Bind port |
| `--device` | `cuda` | _(none)_ | Torch device (`cuda` or `cpu`) |
| `--workers` | `1` | _(none)_ | Gunicorn worker count (each worker loads the model, multiplying VRAM usage) |

All flags with an env variable can also be set via that variable:

```bash
export QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export QWEN_TTS_REF_AUDIO=voice.wav
export QWEN_TTS_REF_TEXT="Transcript of the reference audio."
export QWEN_TTS_LANGUAGE=English

python examples/openai_server.py
```

## Voices JSON Reference

Each voice entry supports these fields:

| Field | Required | Default | Description |
|---|---|---|---|
| `ref_audio` | yes | — | Path to reference audio file (relative to the JSON file or absolute) |
| `ref_text` | when `xvec_only` is false | `""` | Transcript of the reference audio |
| `language` | no | `"Auto"` | Target language |
| `chunk_size` | no | `12` | Streaming chunk size in frames |
| `xvec_only` | no | `false` | Use only the speaker embedding (no in-context learning) |
| `append_silence` | no | `true` | Append silence to the reference audio |

## API Endpoints

### Health Check

```
GET /health
```

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "voices_loaded": ["alloy", "echo"],
  "default_voice": "alloy"
}
```

### List Models

```
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "object": "model", "owned_by": "faster-qwen3-tts"}
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
| `model` | string | `"tts-1"` | Accepted for compatibility but ignored (server loads a single model) |
| `input` | string | _(required)_ | The text to synthesize |
| `voice` | string | `"alloy"` | Voice name (must match a configured voice, or falls back to default) |
| `response_format` | string | `"wav"` | Output format: `wav` or `pcm` |
| `speed` | float | `1.0` | Accepted for compatibility but not yet applied |

**Response**: Streamed audio bytes in the requested format.

## Usage Examples

### curl

```bash
curl -s http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav
```

### Python (OpenAI SDK)

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

Requests are served one at a time. The model instance owns its own CUDA graphs and static KV caches, so concurrent use of a single instance is not safe. A second request will queue until the first completes.

To handle more throughput, increase `--workers` (each worker is a separate gunicorn process with its own model copy, multiplying VRAM usage).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `SoX could not be found!` warning | Install SoX (`apt-get install -y sox`) or ignore it — the server works without it. |
| `torchvision` `RuntimeError: operator torchvision::nms does not exist` | Uninstall torchvision (`pip uninstall torchvision`) — it is not required. |
| `Voice 'X' is not configured` (400 error) | Use a voice name from your config, or ensure `--ref-audio` was provided for a default fallback. |
| Model download is slow or fails | Ensure internet access and sufficient disk space. Pre-download to a local path and pass it via `--model /path/to/model`. |
