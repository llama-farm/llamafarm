"""ROW-79: Sustained load test for edge runtime.

Runs 5 phases over ~30 minutes against the edge runtime, simulating
realistic FunctionGemma drone mission inference patterns.

Usage:
    EDGE_URL=http://192.168.1.100:11540 python -m pytest test_stability.py -s
    EDGE_URL=http://192.168.1.100:11540 python test_stability.py
"""

from __future__ import annotations

import asyncio
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

EDGE_URL = os.environ.get("EDGE_URL", "http://localhost:11540")
MODEL = os.environ.get("EDGE_MODEL", "mission-router-v3")
LOG_DIR = Path(os.environ.get("LOG_DIR", "."))

# Phase durations in seconds (total ~30 min)
PHASE_DURATIONS = {
    "startup": 5 * 60,
    "search_pattern": 10 * 60,
    "detection_burst": 5 * 60,
    "idle_resume": 5 * 60,
    "mission_end": 5 * 60,
}

# FunctionGemma-style prompts: [STATE] + [CMD] -> drone_tool() calls
SEARCH_PROMPTS = [
    "[STATE] altitude=120m heading=045 speed=12m/s battery=78% mode=AUTO [CMD] scan area grid pattern",
    "[STATE] altitude=120m heading=090 speed=12m/s battery=76% mode=AUTO [CMD] check waypoint ALPHA",
    "[STATE] altitude=100m heading=180 speed=10m/s battery=72% mode=AUTO [CMD] survey zone B sector 3",
    "[STATE] altitude=110m heading=270 speed=11m/s battery=70% mode=GUIDED [CMD] orbit point of interest",
    "[STATE] altitude=120m heading=000 speed=12m/s battery=68% mode=AUTO [CMD] continue search pattern",
    "[STATE] altitude=115m heading=135 speed=10m/s battery=65% mode=AUTO [CMD] adjust altitude for terrain",
]

DETECTION_PROMPTS = [
    "[STATE] altitude=80m heading=045 speed=5m/s battery=60% mode=GUIDED [CMD] target found at 34.052N 118.244W confirm identity",
    "[STATE] altitude=60m heading=045 speed=3m/s battery=58% mode=GUIDED [CMD] begin tracking target ID-7",
    "[STATE] altitude=60m heading=050 speed=4m/s battery=56% mode=GUIDED [CMD] classify target thermal signature",
    "[STATE] altitude=50m heading=045 speed=2m/s battery=55% mode=LOITER [CMD] hold position over target",
    "[STATE] altitude=50m heading=045 speed=0m/s battery=54% mode=LOITER [CMD] capture high-res image target zone",
]

MISSION_END_PROMPTS = [
    "[STATE] altitude=50m heading=180 speed=0m/s battery=40% mode=LOITER [CMD] RTL",
    "[STATE] altitude=100m heading=180 speed=15m/s battery=38% mode=RTL [CMD] status report",
    "[STATE] altitude=120m heading=180 speed=15m/s battery=35% mode=RTL [CMD] confirm landing zone clear",
]


@dataclass
class RequestLog:
    phase: str
    timestamp: float
    latency_ms: float
    status: int
    error: str | None = None


@dataclass
class TestResults:
    start_time: float = field(default_factory=time.time)
    logs: list[RequestLog] = field(default_factory=list)
    errors: int = 0
    timeouts: int = 0

    def add(self, log: RequestLog) -> None:
        self.logs.append(log)
        if log.error:
            self.errors += 1
        if log.status == 0:
            self.timeouts += 1

    def write_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "timestamp", "latency_ms", "status", "error"])
            for log in self.logs:
                writer.writerow([log.phase, log.timestamp, f"{log.latency_ms:.1f}", log.status, log.error or ""])

    def summary(self) -> str:
        if not self.logs:
            return "No requests recorded."
        latencies = [log.latency_ms for log in self.logs if log.status == 200]
        elapsed = time.time() - self.start_time
        lines = [
            f"Duration: {elapsed:.0f}s",
            f"Total requests: {len(self.logs)}",
            f"Successful: {len(latencies)}",
            f"Errors: {self.errors}",
            f"Timeouts: {self.timeouts}",
        ]
        if latencies:
            latencies.sort()
            lines += [
                f"Latency avg: {sum(latencies)/len(latencies):.0f}ms",
                f"Latency p50: {latencies[len(latencies)//2]:.0f}ms",
                f"Latency p95: {latencies[int(len(latencies)*0.95)]:.0f}ms",
                f"Latency max: {max(latencies):.0f}ms",
            ]
        return "\n".join(lines)


def _chat_body(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.1,
    }


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    phase: str,
    results: TestResults,
    timeout: float = 30.0,
) -> None:
    t0 = time.time()
    try:
        resp = await client.post(
            f"{EDGE_URL}/v1/chat/completions",
            json=_chat_body(prompt),
            timeout=timeout,
        )
        latency = (time.time() - t0) * 1000
        error = None if resp.status_code == 200 else resp.text[:200]
        results.add(RequestLog(phase=phase, timestamp=t0, latency_ms=latency, status=resp.status_code, error=error))
    except httpx.TimeoutException:
        results.add(RequestLog(phase=phase, timestamp=t0, latency_ms=(time.time() - t0) * 1000, status=0, error="timeout"))
    except Exception as e:
        results.add(RequestLog(phase=phase, timestamp=t0, latency_ms=(time.time() - t0) * 1000, status=0, error=str(e)[:200]))


async def phase_startup(client: httpx.AsyncClient, results: TestResults) -> bool:
    """Phase 1: Verify model is loaded, send warmup requests."""
    phase = "startup"
    print(f"\n{'='*60}")
    print("PHASE 1: Startup (0-5 min)")
    print(f"{'='*60}")

    # Health check
    try:
        resp = await client.get(f"{EDGE_URL}/health", timeout=10)
        print(f"  /health: {resp.status_code}")
        if resp.status_code != 200:
            print(f"  ERROR: health check failed: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ERROR: health check failed: {e}")
        return False

    # Check models
    try:
        resp = await client.get(f"{EDGE_URL}/v1/models", timeout=10)
        print(f"  /v1/models: {resp.status_code}")
        if resp.status_code == 200:
            models = resp.json()
            model_ids = [m.get("id", "?") for m in models.get("data", [])]
            print(f"  Loaded models: {model_ids}")
    except Exception as e:
        print(f"  WARNING: /v1/models check failed: {e}")

    # Warmup requests
    print("  Sending warmup requests...")
    for i in range(3):
        await send_request(client, SEARCH_PROMPTS[0], phase, results)
        last = results.logs[-1]
        print(f"  Warmup {i+1}: {last.latency_ms:.0f}ms (status {last.status})")
        await asyncio.sleep(2)

    # Remaining startup time: idle with periodic health checks
    elapsed = sum(1 for log in results.logs if log.phase == phase) * 3  # rough estimate
    remaining = max(0, PHASE_DURATIONS["startup"] - elapsed)
    check_interval = 30
    checks = int(remaining / check_interval)
    for _ in range(checks):
        await asyncio.sleep(check_interval)
        await send_request(client, random.choice(SEARCH_PROMPTS), phase, results)

    return True


async def phase_search_pattern(client: httpx.AsyncClient, results: TestResults) -> None:
    """Phase 2: Steady search-pattern requests every 10-15s."""
    phase = "search_pattern"
    print(f"\n{'='*60}")
    print("PHASE 2: Search Pattern (5-15 min)")
    print(f"{'='*60}")

    end_time = time.time() + PHASE_DURATIONS["search_pattern"]
    count = 0
    while time.time() < end_time:
        prompt = random.choice(SEARCH_PROMPTS)
        await send_request(client, prompt, phase, results)
        count += 1
        last = results.logs[-1]
        if count % 10 == 0:
            print(f"  Requests sent: {count}, last latency: {last.latency_ms:.0f}ms")
        await asyncio.sleep(random.uniform(10, 15))

    print(f"  Phase 2 complete: {count} requests")


async def phase_detection_burst(client: httpx.AsyncClient, results: TestResults) -> None:
    """Phase 3: Rapid requests every 2-3s with concurrent bursts."""
    phase = "detection_burst"
    print(f"\n{'='*60}")
    print("PHASE 3: Detection Burst (15-20 min)")
    print(f"{'='*60}")

    end_time = time.time() + PHASE_DURATIONS["detection_burst"]
    count = 0
    while time.time() < end_time:
        # Every ~15s, send a burst of 2-3 concurrent requests
        if count % 5 == 0 and count > 0:
            coros = [
                send_request(client, random.choice(DETECTION_PROMPTS), phase, results)
                for _ in range(random.randint(2, 3))
            ]
            await asyncio.gather(*coros)
            count += len(coros)
        else:
            await send_request(client, random.choice(DETECTION_PROMPTS), phase, results)
            count += 1

        if count % 10 == 0:
            phase_logs = [log for log in results.logs if log.phase == phase]
            recent = phase_logs[-5:] if len(phase_logs) >= 5 else phase_logs
            avg = sum(log.latency_ms for log in recent) / len(recent)
            print(f"  Requests sent: {count}, recent avg latency: {avg:.0f}ms")

        await asyncio.sleep(random.uniform(2, 3))

    print(f"  Phase 3 complete: {count} requests")


async def phase_idle_resume(client: httpx.AsyncClient, results: TestResults) -> None:
    """Phase 4: 5 min idle, then burst to verify no cold-start lag."""
    phase = "idle_resume"
    print(f"\n{'='*60}")
    print("PHASE 4: Idle + Resume (20-25 min)")
    print(f"{'='*60}")

    # Record pre-idle latency baseline
    if results.logs:
        recent = [log.latency_ms for log in results.logs[-10:] if log.status == 200]
        baseline = sum(recent) / len(recent) if recent else 0
        print(f"  Pre-idle baseline latency: {baseline:.0f}ms")

    # Idle for 5 minutes (just health checks every 60s)
    print("  Entering idle period (5 min)...")
    idle_duration = PHASE_DURATIONS["idle_resume"]
    idle_end = time.time() + idle_duration
    while time.time() < idle_end:
        await asyncio.sleep(60)
        try:
            resp = await client.get(f"{EDGE_URL}/health", timeout=10)
            print(f"  Idle health check: {resp.status_code}")
        except Exception as e:
            print(f"  Idle health check failed: {e}")

    # Resume with burst
    print("  Resuming with burst requests...")
    for i in range(5):
        await send_request(client, random.choice(SEARCH_PROMPTS), phase, results)
        last = results.logs[-1]
        print(f"  Resume {i+1}: {last.latency_ms:.0f}ms (status {last.status})")
        await asyncio.sleep(1)

    # Check for cold-start lag
    resume_logs = [log for log in results.logs if log.phase == phase and log.status == 200]
    if resume_logs and baseline > 0:
        resume_avg = sum(log.latency_ms for log in resume_logs) / len(resume_logs)
        ratio = resume_avg / baseline
        print(f"  Resume avg latency: {resume_avg:.0f}ms (ratio to baseline: {ratio:.2f}x)")
        if ratio > 3.0:
            print(f"  WARNING: possible cold-start detected (>{ratio:.1f}x baseline)")


async def phase_mission_end(client: httpx.AsyncClient, results: TestResults) -> None:
    """Phase 5: RTL commands, final checks."""
    phase = "mission_end"
    print(f"\n{'='*60}")
    print("PHASE 5: Mission End (25-30 min)")
    print(f"{'='*60}")

    # Send mission-end prompts
    for prompt in MISSION_END_PROMPTS:
        await send_request(client, prompt, phase, results)
        last = results.logs[-1]
        print(f"  RTL request: {last.latency_ms:.0f}ms (status {last.status})")
        await asyncio.sleep(5)

    # Fill remaining time with periodic status checks
    end_time = time.time() + PHASE_DURATIONS["mission_end"] - len(MISSION_END_PROMPTS) * 6
    while time.time() < end_time:
        await send_request(client, MISSION_END_PROMPTS[-1], phase, results)
        await asyncio.sleep(30)

    # Final model check
    try:
        resp = await client.get(f"{EDGE_URL}/v1/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json()
            model_ids = [m.get("id", "?") for m in models.get("data", [])]
            print(f"  Final model check: {model_ids}")
        else:
            print(f"  WARNING: final /v1/models returned {resp.status_code}")
    except Exception as e:
        print(f"  WARNING: final model check failed: {e}")

    # Final health check
    try:
        resp = await client.get(f"{EDGE_URL}/health", timeout=10)
        print(f"  Final /health: {resp.status_code}")
    except Exception as e:
        print(f"  Final health check failed: {e}")


async def run_load_test() -> TestResults:
    results = TestResults()
    csv_path = LOG_DIR / f"stability_{int(time.time())}.csv"

    print("Edge runtime load test (ROW-79)")
    print(f"Target: {EDGE_URL}")
    print(f"Model: {MODEL}")
    print(f"Log: {csv_path}")

    async with httpx.AsyncClient() as client:
        ok = await phase_startup(client, results)
        if not ok:
            print("\nABORTED: startup checks failed.")
            results.write_csv(csv_path)
            return results

        await phase_search_pattern(client, results)
        await phase_detection_burst(client, results)
        await phase_idle_resume(client, results)
        await phase_mission_end(client, results)

    results.write_csv(csv_path)
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(results.summary())
    print(f"\nLatency CSV: {csv_path}")
    return results


# -- pytest entry point --


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stability():
    """Run full 30-minute stability test."""
    results = await run_load_test()
    assert results.errors == 0, f"{results.errors} requests returned errors"
    assert results.timeouts == 0, f"{results.timeouts} requests timed out"
    ok_latencies = [log.latency_ms for log in results.logs if log.status == 200]
    assert ok_latencies, "No successful requests"
    assert max(ok_latencies) < 5000, f"Max latency {max(ok_latencies):.0f}ms exceeds 5s threshold"


# -- standalone entry point --

if __name__ == "__main__":
    results = asyncio.run(run_load_test())
    sys.exit(1 if results.errors or results.timeouts else 0)
