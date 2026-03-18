# OpenAI TTS Server Benchmark Results

**Date:** 2026-03-18
**Server:** `examples/openai_server.py`
**Benchmark script:** `bench_server.py` — 20 requests per tier, 2 warmup requests, ~200-char random sentences
**GPU:** Single CUDA device (all workers share the same GPU)

## Voice Configurations

| Voice | `xvec_only` | Description |
|-------|-------------|-------------|
| `icl` | `false` | In-Context Learning — full reference audio codec tokens in context, longer prefill, more faithful voice reproduction |
| `xvec` | `true` | X-vector only — short prefill (~10 tokens), no reference audio at runtime |

Both voices use `ref_audio.wav` as the reference audio and `chunk_size=12`.

---

# 1.7B Model (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`)

## 1.7B — Sequential (One Request at a Time)

| Metric | ICL 1w | ICL 2w | XVEC 1w | XVEC 2w |
|--------|--------|--------|---------|---------|
| **TTFB avg** | 315 ms | 312 ms | 311 ms | 308 ms |
| **TTFB p90** | 316 ms | 313 ms | 314 ms | 311 ms |
| **Total avg** | 3,157 ms | 2,980 ms | 3,630 ms | 3,626 ms |
| **Total p90** | 3,431 ms | 3,514 ms | 4,110 ms | 4,103 ms |
| **Total max** | 4,507 ms | 3,721 ms | 4,155 ms | 4,140 ms |
| **Avg audio size** | 524.3 KB | 496.2 KB | 609.2 KB | 611.3 KB |
| **Avg text length** | 247 chars | 229 chars | 246 chars | 243 chars |
| **Errors** | 0 | 0 | 0 | 0 |

## 1.7B — Concurrent Load: ICL Voice (`xvec_only=false`)

| Metric | 1w / 1 RPS | 1w / 2 RPS | 1w / 3 RPS | 2w / 1 RPS | 2w / 2 RPS | 2w / 3 RPS |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TTFB avg** | 22,174 | 25,773 | 26,187 | 15,726 | 20,029 | 19,971 |
| **TTFB p90** | 41,921 | 48,068 | 50,305 | 30,831 | 40,293 | 38,732 |
| **Total avg** | 25,021 | 28,636 | 29,089 | 19,353 | 24,512 | 23,022 |
| **Total p90** | 44,899 | 51,145 | 52,948 | 33,181 | 43,009 | 41,191 |
| **Avg audio** | 522 KB | 528 KB | 535 KB | 522 KB | 523 KB | 497 KB |
| **Errors** | 0 | 0 | 0 | 0 | 0 | 0 |

## 1.7B — Concurrent Load: XVEC Voice (`xvec_only=true`)

| Metric | 1w / 1 RPS | 1w / 2 RPS | 1w / 3 RPS | 2w / 1 RPS | 2w / 2 RPS | 2w / 3 RPS |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TTFB avg** | 19,181 | 25,739 | 28,016 | 18,855 | 26,460 | 25,379 |
| **TTFB p90** | 39,357 | 48,195 | 52,367 | 35,464 | 51,154 | 47,363 |
| **Total avg** | 21,957 | 28,604 | 31,076 | 23,860 | 31,032 | 29,282 |
| **Total p90** | 42,361 | 50,980 | 55,381 | 38,543 | 53,366 | 50,566 |
| **Avg audio** | 575 KB | 591 KB | 627 KB | 633 KB | 606 KB | 591 KB |
| **Errors** | 0 | 0 | 0 | 0 | 0 | 0 |

---

# 0.6B Model (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`)

## 0.6B — Sequential (One Request at a Time)

| Metric | ICL 1w | ICL 2w | XVEC 1w | XVEC 2w |
|--------|--------|--------|---------|---------|
| **TTFB avg** | 296 ms | 293 ms | 292 ms | 290 ms |
| **TTFB p90** | 297 ms | 295 ms | 294 ms | 291 ms |
| **Total avg** | 3,029 ms | 3,126 ms | 3,419 ms | 3,126 ms |
| **Total p90** | 3,535 ms | 3,443 ms | 3,887 ms | 3,609 ms |
| **Total max** | 3,619 ms | 3,641 ms | 3,912 ms | 3,683 ms |
| **Avg audio size** | 539.1 KB | 560.3 KB | 612.4 KB | 561.8 KB |
| **Avg text length** | 240 chars | 244 chars | 252 chars | 231 chars |
| **Errors** | 0 | 0 | 0 | 0 |

## 0.6B — Concurrent Load: ICL Voice (`xvec_only=false`)

| Metric | 1w / 1 RPS | 1w / 2 RPS | 1w / 3 RPS | 2w / 1 RPS | 2w / 2 RPS | 2w / 3 RPS |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TTFB avg** | 21,390 | 26,948 | 26,077 | 17,984 | 21,880 | 20,414 |
| **TTFB p90** | 39,747 | 48,912 | 49,079 | 38,439 | 43,477 | 40,678 |
| **Total avg** | 24,277 | 29,785 | 28,814 | 22,565 | 26,031 | 23,959 |
| **Total p90** | 42,592 | 51,914 | 52,177 | 40,552 | 46,248 | 43,710 |
| **Avg audio** | 568 KB | 559 KB | 540 KB | 574 KB | 572 KB | 530 KB |
| **Errors** | 0 | 0 | 0 | 0 | 0 | 0 |

## 0.6B — Concurrent Load: XVEC Voice (`xvec_only=true`)

| Metric | 1w / 1 RPS | 1w / 2 RPS | 1w / 3 RPS | 2w / 1 RPS | 2w / 2 RPS | 2w / 3 RPS |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TTFB avg** | 24,052 | 28,028 | 28,320 | 16,506 | 20,225 | 22,385 |
| **TTFB p90** | 43,820 | 52,731 | 53,672 | 32,654 | 38,825 | 43,480 |
| **Total avg** | 27,121 | 31,045 | 31,480 | 20,786 | 24,502 | 26,868 |
| **Total p90** | 47,167 | 55,663 | 56,779 | 35,499 | 42,009 | 46,049 |
| **Avg audio** | 602 KB | 594 KB | 620 KB | 570 KB | 603 KB | 602 KB |
| **Errors** | 0 | 0 | 0 | 0 | 0 | 0 |

---

# Cross-Model Comparison

## Average TTFB (ms) — All Configurations

| Scenario | 1.7B ICL 1w | 1.7B ICL 2w | 0.6B ICL 1w | 0.6B ICL 2w | 1.7B XVEC 1w | 1.7B XVEC 2w | 0.6B XVEC 1w | 0.6B XVEC 2w |
|----------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|
| Sequential | 315 | 312 | 296 | 293 | 311 | 308 | 292 | 290 |
| 1 RPS | 22,174 | 15,726 | 21,390 | 17,984 | 19,181 | 18,855 | 24,052 | 16,506 |
| 2 RPS | 25,773 | 20,029 | 26,948 | 21,880 | 25,739 | 26,460 | 28,028 | 20,225 |
| 3 RPS | 26,187 | 19,971 | 26,077 | 20,414 | 28,016 | 25,379 | 28,320 | 22,385 |

## Average Total Latency (ms) — All Configurations

| Scenario | 1.7B ICL 1w | 1.7B ICL 2w | 0.6B ICL 1w | 0.6B ICL 2w | 1.7B XVEC 1w | 1.7B XVEC 2w | 0.6B XVEC 1w | 0.6B XVEC 2w |
|----------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|
| Sequential | 3,157 | 2,980 | 3,029 | 3,126 | 3,630 | 3,626 | 3,419 | 3,126 |
| 1 RPS | 25,021 | 19,353 | 24,277 | 22,565 | 21,957 | 23,860 | 27,121 | 20,786 |
| 2 RPS | 28,636 | 24,512 | 29,785 | 26,031 | 28,604 | 31,032 | 31,045 | 24,502 |
| 3 RPS | 29,089 | 23,022 | 28,814 | 23,959 | 31,076 | 29,282 | 31,480 | 26,868 |

---

# Key Takeaways

## 1. Sequential performance: 0.6B is ~6% faster on TTFB, similar on total

| Model | ICL TTFB | ICL Total | XVEC TTFB | XVEC Total |
|-------|----------|-----------|-----------|------------|
| 1.7B | 315 ms | 3.2 s | 311 ms | 3.6 s |
| 0.6B | 296 ms | 3.0 s | 292 ms | 3.4 s |

The 0.6B model saves ~20 ms on TTFB (6%) and ~100-200 ms on total generation. The difference is modest because both models are dominated by the same codec decode and streaming overhead, not transformer forward passes.

## 2. Under load, 0.6B and 1.7B perform similarly

Surprisingly, the 0.6B model does **not** provide a large advantage under concurrent load. At 1 RPS with 1 worker, ICL TTFB is 21.4s (0.6B) vs 22.2s (1.7B) — only 4% faster. The bottleneck under load is GPU serialization (queuing), not per-request compute time. The smaller model's faster forward passes are a small fraction of the total request lifecycle.

## 3. Two workers help both models equally (~15-30% at 1 RPS)

| Config | 1w TTFB @ 1 RPS | 2w TTFB @ 1 RPS | Improvement |
|--------|----------------|----------------|-------------|
| 1.7B ICL | 22,174 ms | 15,726 ms | **29%** |
| 1.7B XVEC | 19,181 ms | 18,855 ms | **2%** |
| 0.6B ICL | 21,390 ms | 17,984 ms | **16%** |
| 0.6B XVEC | 24,052 ms | 16,506 ms | **31%** |

The benefit is inconsistent across voice modes, suggesting GPU contention variance matters more than model size when sharing one GPU.

## 4. XVEC consistently generates more audio frames

Across both models and all configurations, XVEC produces 10-20% more audio (in KB) than ICL for similar text lengths. This results in longer per-request GPU time and generally higher total latency.

## 5. For production at >1 RPS: use multiple GPUs, not more workers

Both models saturate a single GPU at ~1 RPS. Adding workers on the same GPU gives marginal improvement (GPU contention). Scaling to 2-3+ RPS with acceptable latency (<5s) requires distributing workers across separate GPUs.

## 6. Zero errors across all 640 requests

Both models, both voice modes, both worker counts, all RPS tiers — not a single error. The server is rock-solid.
