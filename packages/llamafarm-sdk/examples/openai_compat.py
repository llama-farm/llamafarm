#!/usr/bin/env python3
"""OpenAI compatibility demo — use LlamaFarm as an OpenAI drop-in replacement."""

from openai import OpenAI
from llamafarm import LlamaFarm

MODEL = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M"


def main():
    lf = LlamaFarm()
    print(f"OpenAI-compatible base URL: {lf.openai_base_url}")

    # Create an OpenAI client pointing at LlamaFarm runtime
    client = OpenAI(base_url=lf.openai_base_url, api_key="not-needed")

    # List models
    models = client.models.list()
    print(f"\nAvailable models: {[m.id for m in models.data]}")

    # Chat completion
    print("\n--- Chat Completion ---")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ],
        max_tokens=32,
        temperature=0.1,
    )
    print(f"Response: {resp.choices[0].message.content}")

    # Streaming chat
    print("\n--- Streaming Chat ---")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_tokens=64,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    # Embeddings
    print("\n--- Embeddings ---")
    try:
        emb = client.embeddings.create(model=MODEL, input="Hello world")
        print(f"Embedding dim: {len(emb.data[0].embedding)}")
    except Exception as e:
        print(f"Embeddings not available: {e}")

    print("\n✅ OpenAI compatibility demo complete!")


if __name__ == "__main__":
    main()
