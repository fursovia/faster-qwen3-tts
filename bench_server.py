#!/usr/bin/env python3
"""
Benchmark an OpenAI-compatible TTS server.

Measures:
  - TTFB  (time to first audio byte, i.e. streaming start)
  - Total (time to receive the complete response)

at sustained 1, 2, and 3 RPS with text lengths ~ N(200, 20) chars.
"""

import argparse
import asyncio
import math
import random
import statistics
import string
import time

import httpx

# ---------------------------------------------------------------------------
# Text generation – produce realistic-ish sentences of a target length
# ---------------------------------------------------------------------------

_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a]bright sun shone across the rolling hills while birds sang softly in "
    "the distance and a gentle breeze carried the scent of fresh flowers "
    "through the meadow where children played among the tall grass and laughed "
    "together under the clear blue sky that stretched endlessly above them "
    "every morning she walked to the library to read stories about faraway "
    "lands and ancient civilizations that once thrived along the banks of "
    "great rivers flowing through dense forests and open plains connecting "
    "villages and cities in a web of trade and culture that shaped the modern "
    "world we know today"
).split()


def _generate_text(target_len: int) -> str:
    """Generate pseudo-random English text of approximately *target_len* chars."""
    words: list[str] = []
    length = 0
    while length < target_len:
        w = random.choice(_WORDS)
        words.append(w)
        length += len(w) + 1  # +1 for space
    text = " ".join(words)
    # trim to approximate target (cut at word boundary)
    if len(text) > target_len + 30:
        text = text[: target_len + 30].rsplit(" ", 1)[0]
    return text


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    voice: str,
) -> dict:
    """Fire one streaming TTS request; return timing dict."""
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }

    t_start = time.perf_counter()
    ttfb = None
    total_bytes = 0

    async with client.stream("POST", url, json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes(chunk_size=4096):
            if ttfb is None:
                ttfb = time.perf_counter() - t_start
            total_bytes += len(chunk)

    t_total = time.perf_counter() - t_start

    return {
        "ttfb": ttfb if ttfb is not None else t_total,
        "total": t_total,
        "text_len": len(text),
        "audio_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Run one RPS tier
# ---------------------------------------------------------------------------

async def _run_tier(
    url: str,
    voice: str,
    rps: float,
    n_requests: int,
) -> list[dict]:
    """Send *n_requests* at the given RPS and collect results."""
    interval = 1.0 / rps
    texts = [
        _generate_text(max(20, int(random.gauss(200, 20))))
        for _ in range(n_requests)
    ]

    results: list[dict] = []
    tasks: list[asyncio.Task] = []

    async with httpx.AsyncClient() as client:
        for i, text in enumerate(texts):
            task = asyncio.create_task(
                _send_request(client, url, text, voice)
            )
            tasks.append(task)
            # pace: wait before launching the next request
            if i < len(texts) - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:8.0f} ms"


def _report(rps: float, results: list):
    errors = [r for r in results if isinstance(r, Exception)]
    ok = [r for r in results if isinstance(r, dict)]

    print(f"\n{'=' * 60}")
    print(f"  RPS = {rps}  |  requests = {len(results)}  |  errors = {len(errors)}")
    print(f"{'=' * 60}")

    if errors:
        for e in errors[:3]:
            print(f"  ERROR: {e}")

    if not ok:
        print("  No successful requests.")
        return

    ttfbs = [r["ttfb"] for r in ok]
    totals = [r["total"] for r in ok]

    for label, vals in [("TTFB (streaming start)", ttfbs), ("Total (full audio)", totals)]:
        vals_sorted = sorted(vals)
        p50 = vals_sorted[len(vals_sorted) // 2]
        p90 = vals_sorted[int(len(vals_sorted) * 0.9)]
        p99 = vals_sorted[int(len(vals_sorted) * 0.99)]
        mn = min(vals_sorted)
        mx = max(vals_sorted)
        avg = statistics.mean(vals_sorted)

        print(f"\n  {label}:")
        print(f"    min  = {_fmt(mn)}")
        print(f"    avg  = {_fmt(avg)}")
        print(f"    p50  = {_fmt(p50)}")
        print(f"    p90  = {_fmt(p90)}")
        print(f"    p99  = {_fmt(p99)}")
        print(f"    max  = {_fmt(mx)}")

    avg_text = statistics.mean(r["text_len"] for r in ok)
    avg_audio_kb = statistics.mean(r["audio_bytes"] for r in ok) / 1024
    print(f"\n  avg text length  = {avg_text:.0f} chars")
    print(f"  avg audio size   = {avg_audio_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run_sequential(url: str, voice: str, n_requests: int, warmup: int):
    """Send requests one at a time, wait for each to complete."""
    texts = [
        _generate_text(max(20, int(random.gauss(200, 20))))
        for _ in range(warmup + n_requests)
    ]

    async with httpx.AsyncClient() as client:
        # Warmup
        if warmup > 0:
            print(f"Warming up ({warmup} requests) ...")
            for i in range(warmup):
                await _send_request(client, url, texts[i], voice)

        print(f"Running {n_requests} sequential requests ...\n")
        print(f"  {'#':>3}  {'text_len':>8}  {'TTFB':>10}  {'Total':>10}  {'audio_KB':>10}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

        results = []
        for i in range(n_requests):
            text = texts[warmup + i]
            r = await _send_request(client, url, text, voice)
            results.append(r)
            print(f"  {i+1:3d}  {r['text_len']:8d}  {r['ttfb']*1000:8.0f} ms  {r['total']*1000:8.0f} ms  {r['audio_bytes']/1024:8.1f} KB")

    return results


async def _main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible TTS server")
    parser.add_argument("--url", default="http://localhost:8000/v1/audio/speech")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--mode", choices=["sequential", "rps"], default="sequential")
    parser.add_argument("--rps", nargs="+", type=float, default=[1, 2, 3])
    parser.add_argument("-n", "--requests", type=int, default=20,
                        help="number of requests per tier/run")
    parser.add_argument("--warmup", type=int, default=2,
                        help="warmup requests")
    args = parser.parse_args()

    print(f"Target: {args.url}")
    print(f"Voice:  {args.voice}")
    print(f"Text:   ~N(200, 20) chars\n")

    if args.mode == "sequential":
        results = await _run_sequential(args.url, args.voice, args.requests, args.warmup)
        _report("sequential", results)
    else:
        print(f"Tiers:  {args.rps} RPS  x  {args.requests} requests each")
        for rps in args.rps:
            if args.warmup > 0:
                print(f"\nWarming up ({args.warmup} requests) for {rps} RPS tier ...")
                await _run_tier(args.url, args.voice, rps=1, n_requests=args.warmup)
            print(f"Running {rps} RPS tier ...")
            results = await _run_tier(args.url, args.voice, rps=rps, n_requests=args.requests)
            _report(rps, results)

    print()


if __name__ == "__main__":
    asyncio.run(_main())
