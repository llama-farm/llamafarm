/**
 * OpenAI SDK compatibility demo.
 *
 * Shows how to use LlamaFarm as an OpenAI-compatible backend.
 * Requires: npm install openai (dev dependency only for this example)
 *
 * Usage: npx tsx examples/openai_compat.ts
 */

import { LlamaFarm } from '../src/index.js'

async function main() {
  const lf = new LlamaFarm()

  console.log('=== OpenAI Compatibility Demo ===\n')
  console.log(`OpenAI Base URL: ${lf.openaiBaseUrl}`)
  console.log()

  // You can use this URL with the OpenAI SDK:
  //
  //   import OpenAI from 'openai'
  //   const client = new OpenAI({ baseURL: lf.openaiBaseUrl, apiKey: 'not-needed' })
  //   const completion = await client.chat.completions.create({
  //     model: 'Qwen/Qwen3-8B',
  //     messages: [{ role: 'user', content: 'Hello!' }],
  //   })
  //   console.log(completion.choices[0].message.content)
  //

  // Direct test via fetch to the OpenAI-compatible endpoint
  try {
    const res = await fetch(`${lf.openaiBaseUrl}/models`)
    if (res.ok) {
      const data = await res.json() as { data?: unknown[] }
      console.log(`Models available via OpenAI-compat endpoint: ${(data.data ?? []).length}`)
    }
  } catch (e) {
    console.log('Runtime not available — skipping direct fetch test')
  }

  // Also works with LlamaFarm native SDK
  try {
    const models = await lf.models()
    console.log(`\nModels via LlamaFarm SDK: ${models.length}`)
    for (const m of models.slice(0, 5)) {
      console.log(`  - ${m.id} (loaded: ${m.loaded})`)
    }
  } catch (e) {
    console.log('Server not available — skipping models list')
  }

  console.log('\n✅ OpenAI compatibility ready')
}

main().catch(console.error)
