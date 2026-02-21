#!/usr/bin/env python3
"""Live integration test — Python SDK against running LlamaFarm server."""

import asyncio
import sys

from llamafarm import LlamaFarm

NAMESPACE = "default"
PROJECT = "sdk-test"


def main():
    lf = LlamaFarm()
    print(f"SDK url: {lf.url}")
    print(f"Runtime: {lf.runtime_url}\n")

    # 1. Health check
    print("── Health ──")
    h = lf.health()
    print(f"  Status: {h['status']}")
    print(f"  Summary: {h.get('summary', 'n/a')}\n")

    # 2. List models
    print("── Models (first 5) ──")
    models = lf.models()
    for m in models.get("data", models)[:5]:
        name = m.get("id", m.get("name", "?"))
        print(f"  • {name}")
    print()

    # 3. List projects
    print("── Projects in 'default' (first 5) ──")
    projects = lf.projects(NAMESPACE)
    for p in projects[:5]:
        print(f"  • {p.name}")
    print()

    # 4. Chat (sync, non-streaming)
    print("── Chat (sync) ──")
    p = lf.project(NAMESPACE, PROJECT)
    resp = p.chat("What is 2 + 2? Answer in one sentence.", stateless=True, max_tokens=60)
    print(f"  Model: {resp.model}")
    print(f"  Reply: {resp.text}")
    print(f"  Usage: {resp.usage}\n")

    # 5. Chat streaming
    print("── Chat (streaming) ──")
    print("  ", end="", flush=True)
    full = []
    for chunk in p.chat_stream("Name 3 colors. One word each, comma-separated.", stateless=True, max_tokens=30):
        print(chunk.delta_content or "", end="", flush=True)
        full.append(chunk.delta_content or "")
    print(f"\n  (collected: {''.join(full)})\n")

    # 6. Async chat
    print("── Chat (async) ──")

    async def async_chat():
        resp = await p.achat("What planet is closest to the sun? One word.", stateless=True, max_tokens=20)
        print(f"  Reply: {resp.text}")

        # Async streaming
        print("  Streaming: ", end="", flush=True)
        async for chunk in p.achat_stream("Say goodbye briefly.", stateless=True, max_tokens=20):
            print(chunk.delta_content or "", end="", flush=True)
        print()

    asyncio.run(async_chat())

    # 7. Vision (if available)
    print("\n── Vision ──")
    try:
        models_list = lf.vision.models()
        print(f"  Vision models: {len(models_list)}")
    except Exception as e:
        print(f"  Vision not available: {e}")

    print("\n✅ All live tests passed!")
    lf.close()


if __name__ == "__main__":
    main()
