# OpenAI-Compatible TTS Server Guide

This guide explains how to start and use the OpenAI-compatible text-to-speech API server provided by `faster-qwen3-tts`. The server exposes a `POST /v1/audio/speech` endpoint that is compatible with OpenAI's TTS API, making it easy to integrate with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **Python 3.10+**
- A **reference audio file** (`.wav`) for voice cloning, along with its transcript

## Installation

Install the package with the `demo` extra, which pulls in FastAPI and Uvicorn:

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

### Specifying the model

By default the server loads `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. You can override this with `--model`:

```bash
# Use the 0.6B model for lower VRAM usage
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio voice.wav \
    --ref-text "Transcript" \
    --language English

# Use the 1.7B model (default)
python examples/openai_server.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio voice.wav \
    --ref-text "Transcript" \
    --language English
```

## Server Options Reference

| Flag | Default | Env Variable | Description |
|---|---|---|---|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN_TTS_MODEL` | HuggingFace model ID or local path |
| `--voices` | _(none)_ | `QWEN_TTS_VOICES` | Path to a JSON file mapping voice names to configs |
| `--ref-audio` | _(none)_ | `QWEN_TTS_REF_AUDIO` | Reference audio file (used when `--voices` is not set) |
| `--ref-text` | `""` | `QWEN_TTS_REF_TEXT` | Transcript of the reference audio |
| `--language` | `Auto` | `QWEN_TTS_LANGUAGE` | Target language (e.g. `English`, `French`, `Auto`) |
| `--host` | `0.0.0.0` | _(none)_ | Bind address |
| `--port` | `8000` | _(none)_ | Bind port |
| `--device` | `cuda` | _(none)_ | Torch device (`cuda` or `cpu`) |

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
{"status": "ok", "model_loaded": true}
```

### Generate Speech

```
POST /v1/audio/speech
```

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"tts-1"` | Model identifier (accepted but does not switch models at runtime) |
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

The server uses an `asyncio.Lock` to serialize GPU inference. Only one request is processed at a time; additional requests queue up and are served in order. This prevents GPU out-of-memory errors from concurrent inference.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SoX could not be found!` warning on startup | `sox` is not installed. This is a non-critical warning from the `qwen-tts` dependency. | Install SoX (`apt-get install -y sox`) or ignore the warning -- the server works without it. |
| `torchvision` `RuntimeError: operator torchvision::nms does not exist` | Incompatible `torchvision` version for the installed `torch`. | Uninstall torchvision (`uv pip uninstall --system torchvision`) -- it is not required by this project. |
| `response_format='mp3' requires pydub` | MP3 encoding dependencies missing. | `uv pip install --system pydub` and install `ffmpeg`. |
| `Voice 'X' is not configured` (400 error) | Requested voice name doesn't match any configured voice and no default is set. | Use a voice name from your config, or ensure `--ref-audio` was provided for a default fallback. |
| Model download is slow or fails | First run downloads model weights from HuggingFace. | Ensure internet access and sufficient disk space (~3.5 GB for 1.7B, ~1.5 GB for 0.6B). You can also pre-download to a local path and pass it via `--model /path/to/model`. |
