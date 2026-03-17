# OpenAI-Compatible Server Latency Benchmark (3 Replicas, Single GPU)

**Date:** 2026-03-17
**Server:** `examples/openai_server.py`
**Model:** Qwen/Qwen3-TTS-12Hz-1.7B-Base
**Replicas:** 3 (all on the same GPU)
**Text length:** ~N(200, 20) chars
**Requests:** 20 per tier (+ 4 warmup for sequential, + 2 warmup per RPS tier)

## Sequential Request Latency

Each request waits for the previous one to complete before starting.

| #  | text_len | TTFB      | Total     | audio_KB |
|----|----------|-----------|-----------|----------|
| 1  | 226      | 331 ms    | 3305 ms   | 555.0    |
| 2  | 185      | 340 ms    | 2816 ms   | 453.8    |
| 3  | 219      | 334 ms    | 3714 ms   | 626.3    |
| 4  | 172      | 330 ms    | 2348 ms   | 375.0    |
| 5  | 193      | 341 ms    | 2741 ms   | 446.3    |
| 6  | 196      | 335 ms    | 2516 ms   | 408.8    |
| 7  | 213      | 330 ms    | 4128 ms   | 697.5    |
| 8  | 189      | 336 ms    | 2513 ms   | 408.8    |
| 9  | 229      | 325 ms    | 3591 ms   | 600.0    |
| 10 | 221      | 327 ms    | 3733 ms   | 626.3    |
| 11 | 224      | 334 ms    | 2803 ms   | 457.5    |
| 12 | 191      | 326 ms    | 3369 ms   | 570.0    |
| 13 | 201      | 323 ms    | 3423 ms   | 573.8    |
| 14 | 172      | 330 ms    | 2351 ms   | 371.3    |
| 15 | 204      | 344 ms    | 3106 ms   | 510.0    |
| 16 | 189      | 331 ms    | 2584 ms   | 423.8    |
| 17 | 219      | 330 ms    | 3461 ms   | 585.0    |
| 18 | 210      | 333 ms    | 3028 ms   | 498.8    |
| 19 | 191      | 328 ms    | 2750 ms   | 453.8    |
| 20 | 207      | 328 ms    | 3104 ms   | 513.8    |

### Summary Statistics (Sequential)

| Metric                    | min    | avg     | p50     | p90     | p99     | max     |
|---------------------------|--------|---------|---------|---------|---------|---------|
| **TTFB (streaming start)**| 323 ms | 332 ms  | 331 ms  | 341 ms  | 344 ms  | 344 ms  |
| **Total (full audio)**    | 2348 ms| 3069 ms | 3104 ms | 3733 ms | 4128 ms | 4128 ms |

- **Avg text length:** 203 chars
- **Avg audio size:** 507.8 KB

## Concurrent Request Latency

Requests are launched at the target RPS; multiple requests run in parallel across the 3 replicas.

### 1 RPS (20 requests)

| Metric                    | min     | avg      | p50      | p90      | p99      | max      |
|---------------------------|---------|----------|----------|----------|----------|----------|
| **TTFB (streaming start)**| 333 ms  | 15010 ms | 15276 ms | 30163 ms | 31460 ms | 31460 ms |
| **Total (full audio)**    | 7210 ms | 21379 ms | 21680 ms | 35487 ms | 35742 ms | 35742 ms |

- **Avg text length:** 194 chars
- **Avg audio size:** 495.2 KB

### 2 RPS (20 requests)

| Metric                    | min     | avg      | p50      | p90      | p99      | max      |
|---------------------------|---------|----------|----------|----------|----------|----------|
| **TTFB (streaming start)**| 334 ms  | 21190 ms | 22847 ms | 41864 ms | 45550 ms | 45550 ms |
| **Total (full audio)**    | 6973 ms | 28200 ms | 29433 ms | 49311 ms | 49650 ms | 49650 ms |

- **Avg text length:** 208 chars
- **Avg audio size:** 535.4 KB

### 3 RPS (20 requests)

| Metric                    | min     | avg      | p50      | p90      | p99      | max      |
|---------------------------|---------|----------|----------|----------|----------|----------|
| **TTFB (streaming start)**| 343 ms  | 21644 ms | 22813 ms | 43183 ms | 46047 ms | 46047 ms |
| **Total (full audio)**    | 6302 ms | 28558 ms | 30259 ms | 49112 ms | 50335 ms | 50335 ms |

- **Avg text length:** 202 chars
- **Avg audio size:** 511.7 KB

## Concurrency Isolation Test

3 identical requests fired simultaneously vs sequentially (same texts, ~200 chars each):

| Mode           | Wall clock | Per-request avg |
|----------------|------------|-----------------|
| 3 sequential   | 10.2s      | ~3.4s           |
| 3 concurrent   | 9.4s       | ~9.3s           |
| **Speedup**    | **1.08x**  |                 |

Each concurrent request takes ~3x longer individually due to GPU contention, yielding only ~8% wall-clock improvement.

## Notes

- **3 replicas on a single GPU provides negligible throughput gain.** The GPU is the bottleneck: CUDA graph replays and transformer kernels from concurrent replicas contend for the same compute units. Each concurrent request takes proportionally longer, so total throughput barely improves (~1.08x) while VRAM usage triples.
- Sequential performance: ~332 ms TTFB, ~3.1s total — identical to a 1-replica server.
- Max sustainable throughput remains ~0.33 RPS (1 request / 3.1s), same as 1 replica.
- At 1 RPS: requests arrive 3x faster than the server can process them. Queue depth grows linearly, leading to ~15s avg TTFB (queuing) and ~21s avg total.
- TTFB for the *first* concurrent request remains ~330 ms; the high avg TTFB at higher RPS tiers is queuing time.
- To achieve true N× throughput, replicas should be placed on separate GPUs.
