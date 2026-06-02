# 🦙 LlamaFarm TypeScript SDK

> **Alpha** — API may change between releases.

TypeScript client for [LlamaFarm](https://llamafarm.com) — local AI infrastructure.

## Install

```bash
npm install llamafarm
```

## Quick Start

```typescript
import { LlamaFarm } from 'llamafarm'

const lf = new LlamaFarm()
const project = lf.project('default', 'my-app')

// Chat
const response = await project.chat('What is the meaning of life?')
console.log(response.text)

// Stream
const stream = await project.chatStream('Tell me a story')
const reader = stream.getReader()
while (true) {
  const { done, value } = await reader.read()
  if (done) break
  process.stdout.write(value.content)
}
```

## Features

- **Chat** — sync and streaming with OpenAI-compatible API
- **RAG** — query, health, stats, database management
- **Vision** — detect, classify (YOLO + CLIP)
- **Fine-Tuning** — SFT, CPT, job management
- **KV Cache** — prepare, stats, garbage collection
- **Zero dependencies** — uses native `fetch`

## Requirements

- Node.js 18+ (for native `fetch`)
- A running LlamaFarm server (`lf start`)
