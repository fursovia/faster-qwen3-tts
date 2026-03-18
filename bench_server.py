#!/usr/bin/env python3
"""
Benchmark an OpenAI-compatible TTS server.

Measures:
  - TTFB  (time to first audio byte, i.e. streaming start)
  - Total (time to receive the complete response)

at sustained 1, 2, and 3 RPS with text lengths ~ N(200, 20) chars.
"""

import asyncio
import enum
import random
import statistics
import time
from pathlib import Path
from typing import Optional

import httpx
import typer

# ---------------------------------------------------------------------------
# Text generation – pool of complete sentences for realistic diversity
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "A bright sun shone across the rolling hills while birds sang softly in the distance.",
    "She opened her laptop and began typing the quarterly report that was due by Friday.",
    "The train arrived exactly on time, which surprised everyone waiting on the platform.",
    "Heavy rain pounded against the windows throughout the entire afternoon.",
    "He picked up the phone and dialed the number he had memorized years ago.",
    "The children ran through the park laughing and chasing each other around the fountain.",
    "Scientists have discovered a new species of deep-sea fish near the Pacific trench.",
    "Please remember to submit your timesheet before the end of the business day.",
    "The old library on Main Street has been serving the community for over a hundred years.",
    "After careful consideration, the committee decided to approve the new building proposal.",
    "Traffic was unusually light on the highway this morning despite the holiday weekend.",
    "The restaurant on the corner makes the best homemade pasta in the entire neighborhood.",
    "We need to schedule a follow-up meeting to discuss the results of the pilot program.",
    "The temperature dropped below freezing overnight and ice covered every surface outside.",
    "Her presentation at the conference received outstanding feedback from the audience.",
    "The museum exhibit features artifacts from ancient civilizations along the Silk Road.",
    "Construction crews have been working around the clock to repair the damaged bridge.",
    "The new software update includes several important security patches and bug fixes.",
    "Volunteers planted over three hundred trees in the park as part of the green initiative.",
    "The documentary explores how migration patterns have shifted over the past century.",
    "Fresh bread from the bakery filled the kitchen with a warm and inviting aroma.",
    "The orchestra performed a stunning rendition of Beethoven's Fifth Symphony last night.",
    "Investors remain cautious as global markets continue to show signs of uncertainty.",
    "The hiking trail winds through dense forests before opening up to a breathtaking vista.",
    "Local authorities issued a warning about potential flooding in low-lying areas.",
    "The startup secured ten million dollars in funding during its latest investment round.",
    "Students gathered in the auditorium to hear the guest speaker talk about climate science.",
    "The mechanic said the car would be ready by tomorrow afternoon at the latest.",
    "A thick fog rolled in from the coast and reduced visibility to nearly zero.",
    "The novel tells the story of a young woman who travels across Europe searching for answers.",
    "Engineers are testing a prototype that could revolutionize how we generate clean energy.",
    "The garden was full of colorful flowers that attracted butterflies and hummingbirds.",
    "He carefully balanced the tray of glasses as he navigated through the crowded room.",
    "The annual festival draws thousands of visitors from neighboring towns and cities.",
    "Researchers published their findings in a peer-reviewed journal earlier this week.",
    "The pilot announced that they would be experiencing some turbulence over the mountains.",
    "She spent the entire weekend organizing her closet and donating clothes she no longer wore.",
    "The city council voted unanimously to increase funding for public transportation.",
    "A gentle breeze carried the scent of fresh flowers through the open bedroom window.",
    "The software engineer debugged the critical issue just minutes before the product launch.",
    "Ocean currents play a vital role in regulating the Earth's climate and weather patterns.",
    "The bookstore on Elm Street hosts a weekly reading club every Wednesday evening.",
    "Firefighters responded quickly and managed to contain the blaze before it spread further.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Astronomers observed a rare celestial event that occurs only once every few decades.",
    "The company announced plans to open three new offices in major European cities.",
    "Morning dew glistened on the grass as the first rays of sunlight touched the valley.",
    "The professor explained the concept using a series of clear and simple diagrams.",
    "Wind turbines along the ridge generate enough electricity to power several small towns.",
]


def _generate_text(target_len: int) -> str:
    """Build text of approximately *target_len* chars by joining random sentences."""
    parts: list[str] = []
    length = 0
    pool = list(_SENTENCES)
    random.shuffle(pool)
    idx = 0
    while length < target_len:
        if idx >= len(pool):
            random.shuffle(pool)
            idx = 0
        s = pool[idx]
        idx += 1
        parts.append(s)
        length += len(s) + 1
    text = " ".join(parts)
    if len(text) > target_len + 80:
        text = text[: target_len + 80].rsplit(". ", 1)[0] + "."
    return text


def _load_texts_file(path: Path) -> list[str]:
    """Load benchmark texts from a file (one per line, blank lines skipped)."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Texts file is empty: {path}")
    return lines


def _make_texts(n: int, texts_file: Path | None) -> list[str]:
    """Return *n* texts: from file (cycled) if given, otherwise randomly generated."""
    if texts_file is not None:
        corpus = _load_texts_file(texts_file)
        return [corpus[i % len(corpus)] for i in range(n)]
    return [_generate_text(max(20, int(random.gauss(200, 20)))) for _ in range(n)]


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
    texts_file: Path | None = None,
) -> list[dict | BaseException]:
    """Send *n_requests* at the given RPS with Poisson-distributed arrivals."""
    texts = _make_texts(n_requests, texts_file)
    tasks: list[asyncio.Task] = []

    async with httpx.AsyncClient() as client:
        for i, text in enumerate(texts):
            task = asyncio.create_task(
                _send_request(client, url, text, voice)
            )
            tasks.append(task)
            if i < len(texts) - 1:
                await asyncio.sleep(random.expovariate(rps))

        return list(await asyncio.gather(*tasks, return_exceptions=True))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:8.0f} ms"


def _report(rps: float | str, results: list):
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

class Mode(str, enum.Enum):
    sequential = "sequential"
    rps = "rps"


async def _run_sequential(
    url: str,
    voice: str,
    n_requests: int,
    warmup: int,
    texts_file: Path | None = None,
):
    """Send requests one at a time, wait for each to complete."""
    texts = _make_texts(warmup + n_requests, texts_file)

    async with httpx.AsyncClient() as client:
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


def _parse_rps_list(raw: str) -> list[float]:
    """Parse a comma-separated string like '1,2,3' into a list of floats."""
    try:
        return [float(v) for v in raw.split(",") if v.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"invalid RPS list: {raw!r}") from exc


def main(
    url: str = typer.Option(
        "http://localhost:8000/v1/audio/speech",
        help="TTS server endpoint URL.",
    ),
    voice: str = typer.Option(
        "alloy",
        help="Voice name to request.",
    ),
    mode: Mode = typer.Option(
        Mode.sequential,
        help="Benchmark mode: sequential (one at a time) or rps (concurrent load tiers).",
    ),
    rps: str = typer.Option(
        "1,2,3",
        help="Comma-separated RPS tiers for --mode rps (e.g. '1,2,3').",
    ),
    requests: int = typer.Option(
        20,
        "-n", "--requests",
        min=1,
        help="Number of requests per tier/run.",
    ),
    warmup: int = typer.Option(
        2,
        min=0,
        help="Warmup requests before measurement.",
    ),
    texts_file: Optional[Path] = typer.Option(
        None,
        "--texts-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="File with benchmark texts (one per line); cycled if fewer lines than requests.",
    ),
) -> None:
    """Benchmark an OpenAI-compatible TTS server (TTFB + total latency)."""
    rps_tiers = _parse_rps_list(rps)

    print(f"Target: {url}")
    print(f"Voice:  {voice}")
    if texts_file:
        n_lines = len(_load_texts_file(texts_file))
        print(f"Text:   {n_lines} samples from {texts_file}\n")
    else:
        print(f"Text:   random ~N(200, 20) chars  (50 built-in sentences)\n")

    async def _run() -> None:
        if mode == Mode.sequential:
            results = await _run_sequential(url, voice, requests, warmup, texts_file)
            _report("sequential", results)
        else:
            print(f"Tiers:  {rps_tiers} RPS  x  {requests} requests each")
            for tier_rps in rps_tiers:
                if warmup > 0:
                    print(f"\nWarming up ({warmup} requests) for {tier_rps} RPS tier ...")
                    await _run_tier(url, voice, rps=1, n_requests=warmup,
                                    texts_file=texts_file)
                print(f"Running {tier_rps} RPS tier ...")
                results = await _run_tier(url, voice, rps=tier_rps,
                                          n_requests=requests, texts_file=texts_file)
                _report(tier_rps, results)
        print()

    asyncio.run(_run())


if __name__ == "__main__":
    typer.run(main)
