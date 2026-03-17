# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**faster-qwen3-tts** is a high-performance text-to-speech engine for Qwen3-TTS models using CUDA graph capture. It wraps the upstream `qwen-tts` library and achieves ~5x real-time performance by capturing the talker and predictor decode loops as CUDA graphs with static KV caches.

## Build & Development Commands

```bash
# Install in editable mode
pip install -e .

# Run all tests (requires CUDA GPU and downloads models)
pytest tests/ -v

# Run a single test
pytest tests/test_e2e_parity.py::test_voice_clone_xvec_prefix_parity -v

# Override model used in tests via env vars
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base pytest tests/test_e2e_parity.py -v

# Run benchmarks
./benchmark.sh              # All models
./benchmark.sh 0.6B         # Just 0.6B
./benchmark.sh custom       # CustomVoice modes

# CLI usage
faster-qwen3-tts clone --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --text "Hello" --language English --ref-audio ref.wav --ref-text "..." --output out.wav
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker aiden --text "Hello" --language English --output out.wav
faster-qwen3-tts design --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "Warm narrator" --text "Hello" --language English --output out.wav
```

No linter or formatter is configured in the project.

## Architecture

### Core Flow

The main class `FasterQwen3TTS` (in `model.py`) wraps `Qwen3TTSModel` from the `qwen-tts` library. Generation follows this pipeline:

1. **Prefill** — Standard HuggingFace forward pass processes variable-length input (text embeddings + speaker info + optional reference audio codec tokens)
2. **Talker decode** — CUDA graph replays single-token decode steps through the 28-layer talker transformer, producing the first codebook token per frame
3. **Predictor decode** — CUDA graph replays the full 15-step predictor loop as a single graph, producing codebook tokens 0-14
4. **Codec decode** — Accumulated codebook tokens are decoded to audio waveform

### Key Modules

| Module | Role |
|---|---|
| `model.py` | `FasterQwen3TTS` — main API surface. All `generate_*` and `generate_*_streaming` methods live here. Handles input preparation, voice prompt resolution, and caching. |
| `talker_graph.py` | `TalkerGraph` — captures the talker's single-token decode as a CUDA graph using `transformers.StaticCache` |
| `predictor_graph.py` | `PredictorGraph` — captures the predictor's unrolled 15-step codebook loop as a single CUDA graph |
| `generate.py` | Non-streaming generation loop: prefill → graph decode → sample → collect all tokens |
| `streaming.py` | Streaming generation loop: same decode but yields audio chunks every `chunk_size` frames with sliding-window codec decoding |
| `sampling.py` | Token sampling utilities (temperature, top-k, top-p, repetition penalty) |
| `cli.py` | CLI with subcommands: `clone`, `custom`, `design`, `serve` |

### CUDA Graph Constraints

CUDA graphs require deterministic tensor shapes. This is achieved via:
- `transformers.StaticCache` for fixed-size KV buffers (no dynamic allocation)
- Pre-allocated `cache_position` tensors and attention mask tables
- The predictor's 15-step loop is fully unrolled into a single graph

### Voice Cloning Modes

Three approaches, selected via API parameters:
- **X-vector only** — short prefill (~10 tokens), no reference audio at runtime, clean cross-language. Set `xvec_only=True`.
- **ICL (In-Context Learning)** — full reference audio codec tokens in context (upstream default), longer prefill, more faithful voice reproduction
- **Precomputed voice clone prompt** — save speaker embedding once via `extract_speaker_embedding()`, pass as `voice_clone_prompt` to skip re-encoding

### Test Structure

- `test_e2e_parity.py` — validates token-level parity between the CUDA graph fast path and upstream dynamic cache. Uses env vars `QWEN_TTS_MODEL`, `QWEN_TTS_CUSTOM_MODEL`, `QWEN_TTS_VOICE_DESIGN_MODEL` for model selection.
- `test_voice_clone_prompt_api.py` — API surface validation for precomputed voice clone prompts
- `test_sample_rate.py` — sample rate inference from model config
- `test_sampling.py` — sampling logic correctness

### Commit Convention

Commits use conventional prefixes: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, often scoped like `fix(base):` or `feat(cli):`.
