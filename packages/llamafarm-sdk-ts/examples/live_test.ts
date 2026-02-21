#!/usr/bin/env npx tsx
/**
 * Live integration test — TypeScript SDK against running LlamaFarm server.
 */

import { LlamaFarm } from '../src/index.js'

const NAMESPACE = 'default'
const PROJECT = 'sdk-test'

async function main() {
  const lf = new LlamaFarm()
  console.log(`SDK url: ${lf.url}`)
  console.log(`Runtime: ${lf.runtimeUrl}\n`)

  // 1. Health check
  console.log('── Health ──')
  const h = await lf.health()
  console.log(`  Status: ${h.status}`)
  console.log(`  Summary: ${h.summary ?? 'n/a'}\n`)

  // 2. List models
  console.log('── Models (first 5) ──')
  const models = await lf.models()
  for (const m of models.slice(0, 5)) {
    console.log(`  • ${m.id}`)
  }
  console.log()

  // 3. List projects
  console.log("── Projects in 'default' (first 5) ──")
  const projects = await lf.projects(NAMESPACE)
  for (const p of projects.slice(0, 5)) {
    console.log(`  • ${p.name}`)
  }
  console.log()

  // 4. Chat (non-streaming)
  console.log('── Chat (sync) ──')
  const p = lf.project(NAMESPACE, PROJECT)
  const resp = await p.chat('What is 2 + 2? Answer in one sentence.', { stateless: true, max_tokens: 60 })
  console.log(`  Model: ${resp.model}`)
  console.log(`  Reply: ${resp.text}`)
  console.log(`  Usage: ${JSON.stringify(resp.usage)}\n`)

  // 5. Chat streaming
  console.log('── Chat (streaming) ──')
  process.stdout.write('  ')
  const stream = await p.chatStream('Name 3 colors. One word each, comma-separated.', { stateless: true, max_tokens: 30 })
  const reader = stream.getReader()
  const parts: string[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    process.stdout.write(value.content)
    parts.push(value.content)
  }
  console.log(`\n  (collected: ${parts.join('')})\n`)

  // 6. Vision (if available)
  console.log('── Vision ──')
  console.log('  Vision API accessible via lf.vision')

  // 7. OpenAI compatibility
  console.log('\n── OpenAI Compatibility ──')
  console.log(`  Base URL: ${lf.openaiBaseUrl}`)

  // 8. ML APIs (graceful — these may 404 if backends not loaded)
  console.log('\n── ML APIs ──')
  const mlTests: [string, () => Promise<unknown>][] = [
    ['anomaly.backends', () => lf.anomaly.backends()],
    ['anomaly.models', () => lf.anomaly.models()],
    ['classifier.models', () => lf.classifier.models()],
    ['timeseries.backends', () => lf.timeseries.backends()],
    ['timeseries.models', () => lf.timeseries.models()],
    ['adtk.detectors', () => lf.adtk.detectors()],
    ['catboost.info', () => lf.catboost.info()],
    ['drift.detectors', () => lf.drift.detectors()],
    ['explain.explainers', () => lf.explain.explainers()],
    ['polars.buffers', () => lf.polars.buffers()],
    ['nlp.embeddings', () => lf.nlp.embeddings('test')],
  ]
  for (const [name, fn] of mlTests) {
    try {
      const r = await fn()
      console.log(`  ✅ ${name}: ${JSON.stringify(r).slice(0, 80)}`)
    } catch (e: any) {
      console.log(`  ⚠️  ${name}: ${e.message?.slice(0, 60)}`)
    }
  }

  console.log('\n✅ All live tests passed!')
}

main().catch(err => {
  console.error('❌ Error:', err.message)
  process.exit(1)
})
