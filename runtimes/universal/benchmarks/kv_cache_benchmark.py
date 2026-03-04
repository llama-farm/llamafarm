#!/usr/bin/env python3
"""KV Cache Benchmark — Multi-Turn Conversation with Cache Key Chaining

Measures Time To First Token (TTFT) savings from server-side KV cache across
a realistic multi-turn conversation. Proves that cached turns only decode new
tokens regardless of conversation length.

## What it tests

1. **Turn 1 (cold start)**: ~4000-token system prompt + RAG context + user
   question. No cache — full prompt processing. Response returns `cache_key_1`.

2. **Polluter call**: A different agent (cooking assistant) makes a request,
   completely trashing the model's in-memory KV state. This simulates
   multi-agent sharing of a single model.

3. **Turn 2 (cached)**: Same conversation + short follow-up (~20 tokens).
   Sends `cache_key_1` — server restores serialized KV state from Turn 1,
   only decodes the new user message. Returns `cache_key_2`.

4. **Polluter call**: Trashes KV again.

5. **Turn 3 (cached)**: Full 6-message history + another follow-up (~30
   tokens). Sends `cache_key_2` — restores KV from Turn 2, only decodes
   the newest turn.

6. **Turn 3 (no cache baseline)**: Same 6 messages, no cache key — full
   reprocessing of entire conversation from scratch. This is the baseline
   to compare against.

## How cache key chaining works

Each response includes `x_cache.new_cache_key` — a key for the KV state
after that turn (including the assistant's response). The next request sends
this key, and the server:

1. Validates segment hashes (system prompt, tools, each turn) to confirm
   the conversation prefix is unchanged
2. Restores the serialized KV state (skipping prompt processing entirely)
3. Decodes only the new delta tokens (the latest user message)
4. After generation, saves a new cache entry for the next turn

## Running

    cd runtimes/universal
    uv run python benchmarks/kv_cache_benchmark.py

    # Against a different host:
    uv run python benchmarks/kv_cache_benchmark.py --base-url http://localhost:14345

## Expected results (Qwen3-8B Q4_K_M, Apple M1 Max)

    Turn 1 (cold):     ~5500ms TTFT
    Turn 2 (cached):    ~440ms TTFT  (12x faster)
    Turn 3 (cached):    ~440ms TTFT  (13x faster)
    Turn 3 (no cache): ~5900ms TTFT  (baseline)

Cached TTFT is constant (~440ms) regardless of conversation length — only
the new user message is decoded. The ~440ms includes KV restore (~60ms) plus
first-token sampling time.
"""

import argparse
import json
import time

import httpx

# ── System prompt + RAG context (~4000 tokens) ──────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst at GlobalTech Investments. Your role is to provide detailed analysis of market conditions, portfolio performance, and investment recommendations.

## Current Portfolio Holdings

### Equities (60% allocation)
1. **AAPL** (Apple Inc.) - 500 shares @ $178.50 avg cost
   - Current: $185.20 (+3.75%)
   - Sector: Technology
   - Market Cap: $2.87T
   - P/E Ratio: 29.8
   - Dividend Yield: 0.55%
   - Recent catalysts: Vision Pro launch, services revenue growth, India manufacturing expansion
   - Risk factors: China exposure, antitrust regulation, smartphone market saturation

2. **MSFT** (Microsoft) - 300 shares @ $380.00 avg cost
   - Current: $415.60 (+9.37%)
   - Sector: Technology
   - Market Cap: $3.09T
   - P/E Ratio: 36.2
   - Dividend Yield: 0.72%
   - Recent catalysts: Azure AI integration, Copilot monetization, gaming division growth
   - Risk factors: Enterprise spending slowdown, regulatory scrutiny, Activision integration

3. **JPM** (JPMorgan Chase) - 200 shares @ $195.00 avg cost
   - Current: $198.40 (+1.74%)
   - Sector: Financials
   - Market Cap: $571B
   - P/E Ratio: 11.8
   - Dividend Yield: 2.32%
   - Recent catalysts: Net interest income growth, trading revenue, First Republic integration
   - Risk factors: Commercial real estate exposure, recession risk, regulatory capital requirements

4. **UNH** (UnitedHealth Group) - 100 shares @ $520.00 avg cost
   - Current: $485.30 (-6.67%)
   - Sector: Healthcare
   - Market Cap: $447B
   - P/E Ratio: 19.2
   - Dividend Yield: 1.52%
   - Recent catalysts: Optum Health expansion, Medicare Advantage enrollment
   - Risk factors: DOJ investigation, Change Healthcare breach impact, pharmacy benefit reform

5. **NVDA** (NVIDIA) - 400 shares @ $480.00 avg cost
   - Current: $875.50 (+82.40%)
   - Sector: Technology/Semiconductors
   - Market Cap: $2.16T
   - P/E Ratio: 64.8
   - Dividend Yield: 0.02%
   - Recent catalysts: H100/H200 demand, data center revenue, sovereign AI initiatives
   - Risk factors: Valuation compression, export controls, supply chain concentration

### Fixed Income (25% allocation)
1. US Treasury 10Y Notes - $500,000 face value @ 4.25% yield
2. Investment Grade Corporate Bonds (LQD ETF) - 1000 shares @ $108.50
3. Municipal Bonds (MUB ETF) - 800 shares @ $106.20
4. TIPS (TIP ETF) - 500 shares @ $104.80

### Alternatives (15% allocation)
1. Gold (GLD ETF) - 200 shares @ $190.50
2. Real Estate (VNQ ETF) - 300 shares @ $82.40
3. Commodities (DBC ETF) - 400 shares @ $22.80
4. Private Equity Fund - $100,000 committed capital

## Market Context (RAG-retrieved data)

### Federal Reserve Policy
- Current Fed Funds Rate: 5.25-5.50%
- Market pricing: 3 cuts expected in 2024 (June, September, December)
- Quantitative Tightening: $95B/month balance sheet reduction ongoing
- Inflation: CPI 3.1% YoY (Dec 2023), Core PCE 2.9%
- Employment: NFP +216K (Dec), Unemployment 3.7%
- GDP Growth: Q4 2023 annualized 3.3%, full year 2.5%

### Sector Performance (YTD)
- Technology: +5.2%
- Healthcare: -2.1%
- Financials: +3.8%
- Energy: -1.5%
- Consumer Discretionary: +4.1%
- Industrials: +2.7%
- Materials: -0.8%
- Utilities: -3.2%
- Real Estate: +1.9%
- Communication Services: +6.1%

### Risk Indicators
- VIX: 13.2 (low volatility regime)
- 10Y-2Y Spread: -0.35% (inverted, recession signal)
- Credit Spreads (IG): +95bps (normal)
- Credit Spreads (HY): +350bps (slightly elevated)
- Put/Call Ratio: 0.82 (slightly bullish)
- Margin Debt: $680B (elevated)
- Insider Selling Ratio: 3.2:1 (above average)

### Geopolitical Risk Assessment
1. US-China relations: Elevated tension, export controls expanding
2. Middle East: Houthi attacks disrupting Red Sea shipping, oil supply risk
3. Ukraine-Russia: Frozen conflict, European energy security concerns
4. Taiwan: TSMC concentration risk for semiconductor supply
5. US Election 2024: Policy uncertainty premium building

### Technical Analysis Summary
- S&P 500: Trading above 200-day MA, RSI 62 (neutral-bullish)
- NASDAQ: New 52-week highs, breadth improving
- Russell 2000: Below 200-day MA, lagging large caps
- Dollar Index (DXY): 103.2, range-bound
- Bitcoin: $43,500, institutional adoption narrative

### Client Risk Profile
- Risk Tolerance: Moderate-Aggressive
- Investment Horizon: 10+ years
- Tax Situation: High income bracket (37% federal)
- Liquidity Needs: Low (no major purchases planned)
- ESG Preferences: Moderate (avoid tobacco, weapons)
- Rebalancing Schedule: Quarterly
- Target Return: 8-10% annualized
- Max Drawdown Tolerance: -20%

### Recent Trade History
- 2024-01-15: Bought 100 NVDA @ $548 (added to position)
- 2024-01-10: Sold 50 TSLA @ $235 (trimmed position)
- 2024-01-05: Bought 200 MUB @ $106.20 (tax-loss harvest)
- 2023-12-28: Rebalanced alternatives allocation (+2% gold, -2% RE)
- 2023-12-15: Bought 100 UNH @ $520 (new position on pullback)

## Response Guidelines
- Always cite specific data points from the portfolio and market context
- Provide risk-adjusted return estimates where applicable
- Consider tax implications for any trade recommendations
- Flag any positions that exceed 10% portfolio concentration
- Use precise figures, not approximations
- Consider correlation between holdings when assessing risk
"""

# ── Conversation turns ───────────────────────────────────────────────────────

TURN1_USER = "Given the current market conditions and our portfolio, what are the top 3 risks I should be most concerned about right now? Be specific to our holdings."
TURN2_USER = "What about NVDA specifically? Should we trim after the 82% run?"
TURN3_USER = "OK, if we trim NVDA to 10% weight, where should we redeploy that capital given the current rate environment?"

# Different agent conversation (to pollute model KV between turns)
POLLUTER_MESSAGES = [
    {"role": "system", "content": "You are a helpful cooking assistant."},
    {"role": "user", "content": "How do I make scrambled eggs?"},
]

# ── Helpers ──────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3-8B-GGUF"  # overridden by --model
N_CTX = None  # overridden by --n-ctx


def make_request(
    base_url: str,
    messages: list[dict],
    cache_key: str | None = None,
    return_cache_key: bool = False,
    max_tokens: int = 80,
) -> dict:
    """Make a streaming chat completion request. Returns TTFT and cache info."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    if N_CTX:
        payload["n_ctx"] = N_CTX
    if cache_key:
        payload["cache_key"] = cache_key
    if return_cache_key:
        payload["return_cache_key"] = return_cache_key

    t0 = time.perf_counter()
    t_first_token = None
    full_content = ""
    x_cache = None

    with httpx.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload, timeout=120.0
    ) as resp:
        next_is_cache = False
        for line in resp.iter_lines():
            # Named SSE event: "event: x_cache" means the next data line is cache info
            if line == "event: x_cache":
                next_is_cache = True
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if next_is_cache:
                x_cache = chunk
                next_is_cache = False
                continue
            # Legacy: inline x_cache in chunk (non-streaming responses)
            if "x_cache" in chunk:
                x_cache = chunk["x_cache"]
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content and t_first_token is None:
                t_first_token = time.perf_counter()
            full_content += content

    t_end = time.perf_counter()
    return {
        "ttft_ms": round((t_first_token - t0) * 1000, 1) if t_first_token else None,
        "total_ms": round((t_end - t0) * 1000, 1),
        "full_content": full_content,
        "x_cache": x_cache,
        "cache_key": x_cache.get("new_cache_key") if x_cache else None,
    }


def pollute(base_url: str):
    """Simulate a different agent trashing the model's KV state."""
    make_request(base_url, POLLUTER_MESSAGES, max_tokens=30)


# ── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(base_url: str):
    print("KV Cache Benchmark — Multi-Turn Conversation with Cache Key Chaining")
    print("=" * 70)

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    history = [system_msg]

    # ── Turn 1: Cold start ───────────────────────────────────────────────
    history.append({"role": "user", "content": TURN1_USER})
    print("\n[Turn 1] ~4000-token system+RAG context + user question (cold start)")
    r1 = make_request(base_url, history, return_cache_key=True)
    print(f"  TTFT: {r1['ttft_ms']}ms")
    print(f"  cache_key_1 = {r1['cache_key']}")
    history.append({"role": "assistant", "content": r1["full_content"]})

    # ── Pollute ──────────────────────────────────────────────────────────
    print("\n  ↳ [polluter] Different agent call (cooking assistant) trashes model KV")
    pollute(base_url)

    # ── Turn 2: Cached ───────────────────────────────────────────────────
    history.append({"role": "user", "content": TURN2_USER})
    print("\n[Turn 2] Full history + follow-up (~20 new tokens), sends cache_key_1")
    r2 = make_request(base_url, history, cache_key=r1["cache_key"], return_cache_key=True)
    xc2 = r2.get("x_cache") or {}
    print(f"  TTFT: {r2['ttft_ms']}ms")
    print(f"  Cache: hit={xc2.get('hit')}, reused_tokens={xc2.get('reused_tokens')}")
    print(f"  cache_key_2 = {r2['cache_key']}")
    history.append({"role": "assistant", "content": r2["full_content"]})

    # ── Pollute ──────────────────────────────────────────────────────────
    print("\n  ↳ [polluter] Different agent call trashes model KV")
    pollute(base_url)

    # ── Turn 3: Cached ───────────────────────────────────────────────────
    history.append({"role": "user", "content": TURN3_USER})
    n_msgs = len(history)
    print(f"\n[Turn 3] {n_msgs}-message history + follow-up (~30 new tokens), sends cache_key_2")
    r3 = make_request(base_url, history, cache_key=r2["cache_key"], return_cache_key=True)
    xc3 = r3.get("x_cache") or {}
    print(f"  TTFT: {r3['ttft_ms']}ms")
    print(f"  Cache: hit={xc3.get('hit')}, reused_tokens={xc3.get('reused_tokens')}")

    # ── Pollute ──────────────────────────────────────────────────────────
    print("\n  ↳ [polluter] Different agent call trashes model KV")
    pollute(base_url)

    # ── Turn 3 Baseline: No cache ────────────────────────────────────────
    print(f"\n[Turn 3 BASELINE] Same {n_msgs} messages, NO cache key (full reprocess)")
    r3_base = make_request(base_url, history)
    print(f"  TTFT: {r3_base['ttft_ms']}ms")

    # ── Results ──────────────────────────────────────────────────────────
    t1 = r1["ttft_ms"]
    t2 = r2["ttft_ms"]
    t3 = r3["ttft_ms"]
    t3b = r3_base["ttft_ms"]

    # Guard against None TTFT values (e.g. if streaming response had no tokens)
    if any(t is None for t in (t1, t2, t3, t3b)):
        print("\n  WARNING: Some TTFT values are None — cannot compute speedup ratios")
        print(f"  t1={t1}, t2={t2}, t3={t3}, t3_baseline={t3b}")
        return

    print(f"\n{'─'*70}")
    print("  RESULTS (TTFT, streaming)")
    print(f"{'─'*70}")
    print(f"  {'Step':<35} {'TTFT':>8}  {'vs baseline':>12}")
    print(f"  {'─'*35} {'─'*8}  {'─'*12}")
    print(f"  {'Turn 1 (cold start)':<35} {t1:>7.0f}ms  {'—':>12}")
    t2_speedup = f"{t1/t2:>10.1f}x faster" if t2 > 0 else "N/A"
    t3_speedup = f"{t3b/t3:>10.1f}x faster" if t3 > 0 else "N/A"
    print(f"  {'Turn 2 (cached, cache_key_1)':<35} {t2:>7.0f}ms  {t2_speedup}")
    print(f"  {'Turn 3 (cached, cache_key_2)':<35} {t3:>7.0f}ms  {t3_speedup}")
    print(f"  {'Turn 3 (no cache, baseline)':<35} {t3b:>7.0f}ms  {'baseline':>12}")
    if t3 > 0:
        print(f"\n  Cache saves ~{t3b - t3:.0f}ms per turn ({t3b/t3:.0f}x faster)")
    avg = (t2 + t3) / 2
    print(f"  Cached TTFT is constant (~{avg:.0f}ms) regardless of history length")
    print()


def run_prepare_benchmark(base_url: str):
    """Test pre-warming system prompt + tools via /v1/cache/prepare."""
    print("\n\nKV Cache Benchmark — Pre-Warmed System Prompt via /v1/cache/prepare")
    print("=" * 70)

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    # ── Pre-warm the system prompt ───────────────────────────────────────
    print("\n[Prepare] Pre-warming ~4000-token system prompt + RAG context...")
    t0 = time.perf_counter()
    try:
        resp = httpx.post(f"{base_url}/v1/cache/prepare", json={
            "model": MODEL,
            "messages": [system_msg],
            "warm": True,
            "pinned": True,
        }, timeout=120.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  /prepare failed: {e}")
        print("  Skipping pre-warm benchmark (runtime may have crashed under memory pressure)")
        return
    t1 = time.perf_counter()
    prep = resp.json()
    ck_warm = prep["cache_key"]
    print(f"  Time: {(t1-t0)*1000:.0f}ms")
    print(f"  Tokens: {prep['token_count']}, KV size: {prep['size_bytes']/1024:.0f}KB")
    print(f"  cache_key = {ck_warm}")

    # ── Pollute ──────────────────────────────────────────────────────────
    print("\n  ↳ [polluter] Different agent call trashes model KV")
    pollute(base_url)

    # ── First user message WITH pre-warmed cache ─────────────────────────
    msgs_cached = [system_msg, {"role": "user", "content": TURN1_USER}]
    print("\n[Turn 1 — pre-warmed] System+RAG + user question, sends prepare cache_key")
    r_warm = make_request(base_url, msgs_cached, cache_key=ck_warm, return_cache_key=True)
    print(f"  TTFT: {r_warm['ttft_ms']}ms")
    xc = r_warm.get("x_cache") or {}
    print(f"  Cache: hit={xc.get('hit')}, reused_tokens={xc.get('reused_tokens')}")

    # ── Pollute ──────────────────────────────────────────────────────────
    print("\n  ↳ [polluter] Different agent call trashes model KV")
    pollute(base_url)

    # ── Same message WITHOUT cache (baseline) ────────────────────────────
    print("\n[Turn 1 — no cache] Same messages, full reprocess (baseline)")
    r_cold = make_request(base_url, msgs_cached)
    print(f"  TTFT: {r_cold['ttft_ms']}ms")

    # ── Results ──────────────────────────────────────────────────────────
    tw = r_warm["ttft_ms"]
    tc = r_cold["ttft_ms"]
    print(f"\n{'─'*70}")
    print("  RESULTS — Pre-Warmed System Prompt (TTFT, streaming)")
    print(f"{'─'*70}")
    print(f"  {'Step':<40} {'TTFT':>8}  {'vs baseline':>12}")
    print(f"  {'─'*40} {'─'*8}  {'─'*12}")
    print(f"  {'Turn 1 (pre-warmed via /prepare)':<40} {tw:>7.0f}ms  {tc/tw:>10.1f}x faster")
    print(f"  {'Turn 1 (no cache, baseline)':<40} {tc:>7.0f}ms  {'baseline':>12}")
    print(f"\n  Pre-warm eliminates cold start: first message gets {tc/tw:.0f}x faster TTFT")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Benchmark — measures TTFT savings from multi-turn cache chaining"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11540",
        help="Base URL of the Universal Runtime (default: http://localhost:11540)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B-GGUF",
        help="Model ID to benchmark (default: Qwen/Qwen3-8B-GGUF)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Context size override (useful for models with large default ctx)",
    )
    parser.add_argument(
        "--unload-first",
        action="store_true",
        help="Unload all models before starting (frees memory for large models)",
    )
    args = parser.parse_args()

    global MODEL, N_CTX
    MODEL = args.model
    N_CTX = args.n_ctx

    print(f"Target: {args.base_url}")
    print(f"Model:  {MODEL}")

    if args.unload_first:
        print("Unloading all models to free memory...")
        try:
            r = httpx.post(f"{args.base_url}/v1/models/unload", timeout=30)
            data = r.json()
            print(f"  Unloaded {data.get('unloaded', 0)} model(s)")
        except Exception as e:
            print(f"  Unload failed (may not be supported): {e}")

    print("Warming up model...")
    warmup = make_request(args.base_url, [
        {"role": "user", "content": "Say hello"},
    ], max_tokens=5)
    print(f"Model loaded ({warmup['total_ms']:.0f}ms)\n")

    run_benchmark(args.base_url)
    run_prepare_benchmark(args.base_url)


if __name__ == "__main__":
    main()
