# 🦙 LlamaFarm Python SDK

> **Alpha** — API may change between releases.

Python client for [LlamaFarm](https://llamafarm.com) — local AI infrastructure.

## Install

```bash
pip install llamafarm
```

## Quick Start

```python
from llamafarm import LlamaFarm

# Auto-discovers local server (localhost:14345)
lf = LlamaFarm()

# Create a project
project = lf.create_project("my-app", model="Qwen/Qwen3-8B")

# Chat
response = project.chat("What is the meaning of life?")
print(response.text)

# Stream
for chunk in project.chat_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## Async

```python
import asyncio
from llamafarm import LlamaFarm

async def main():
    lf = LlamaFarm()
    project = lf.project("my-app")
    
    response = await project.achat("Hello!")
    print(response.text)
    
    async for chunk in project.achat_stream("Tell me a story"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

## Vision

```python
import base64

# Detect objects
result = project.vision.detect(
    base64.b64encode(open("photo.jpg", "rb").read()),
    model="yolov8n",
    confidence=0.5,
)
for det in result.detections:
    print(f"{det.class_name}: {det.confidence:.2f}")

# Classify
result = project.vision.classify(
    base64.b64encode(open("photo.jpg", "rb").read()),
    classes=["cat", "dog", "bird"],
)
print(result.classification.class_name)
```

## Fine-Tuning

```python
# SFT
job = project.finetune.sft(
    model="google/gemma-3-4b-it",
    dataset=[
        {"conversations": [
            {"from": "human", "value": "What is LlamaFarm?"},
            {"from": "gpt", "value": "A local AI platform."},
        ]}
    ],
    epochs=3,
)
print(f"Job {job.job_id}: {job.status}")

# Check status
status = project.finetune.status(job.job_id)
print(f"Progress: {status.progress:.0%}")
```

## KV Cache

```python
# Pre-warm cache for a system prompt
result = project.cache.prepare(
    messages=[{"role": "system", "content": "You are a medical assistant..."}],
    warm=True,
)

# Use cache key in chat
response = project.chat(
    "What are the symptoms of diabetes?",
    cache_key=result["cache_key"],
    return_cache_key=True,
)
# Next turn reuses cache
next_response = project.chat(
    "What about treatment options?",
    cache_key=response.x_cache["new_cache_key"],
)
```

## Configuration

```python
# Explicit URL
lf = LlamaFarm(url="http://my-server:14345")

# With API key
lf = LlamaFarm(api_key="sk-...")

# Environment variables
# LLAMAFARM_URL=http://localhost:14345
# LLAMAFARM_API_KEY=sk-...
```

## Requirements

- Python 3.10+
- `httpx` and `pydantic` (installed automatically)
- A running LlamaFarm server (`lf start`)
