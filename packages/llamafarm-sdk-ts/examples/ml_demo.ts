/**
 * ML APIs demo — exercises all ML/AI endpoints.
 *
 * Usage: npx tsx examples/ml_demo.ts
 */

import { LlamaFarm } from '../src/index.js'

const lf = new LlamaFarm()

async function tryApi(name: string, fn: () => Promise<unknown>) {
  try {
    const result = await fn()
    console.log(`✅ ${name}:`, JSON.stringify(result).slice(0, 200))
  } catch (e: any) {
    console.log(`⚠️  ${name}: ${e.message?.slice(0, 100)}`)
  }
}

async function main() {
  console.log('=== LlamaFarm ML APIs Demo ===\n')

  // NLP
  console.log('--- NLP ---')
  await tryApi('nlp.embeddings', () => lf.nlp.embeddings('hello world'))
  await tryApi('nlp.classify', () => lf.nlp.classify('I love this!', ['positive', 'negative']))
  await tryApi('nlp.ner', () => lf.nlp.ner('John works at Google in NYC'))
  await tryApi('nlp.rerank', () => lf.nlp.rerank('AI models', ['deep learning paper', 'cooking recipe', 'neural networks']))

  // Anomaly Detection
  console.log('\n--- Anomaly Detection ---')
  const normalData = Array.from({ length: 50 }, () => [Math.random(), Math.random()])
  await tryApi('anomaly.fit', () => lf.anomaly.fit(normalData, { model_name: 'demo-iforest' }))
  await tryApi('anomaly.detect', () => lf.anomaly.detect([[0.5, 0.5], [99, 99]], { explain: true }))
  await tryApi('anomaly.models', () => lf.anomaly.models())
  await tryApi('anomaly.backends', () => lf.anomaly.backends())

  // Classifier
  console.log('\n--- Classifier ---')
  const X = [[1, 0], [0, 1], [1, 1], [0, 0]]
  const y = [1, 1, 0, 0]
  await tryApi('classifier.fit', () => lf.classifier.fit(X, y, { model_name: 'demo-clf' }))
  await tryApi('classifier.predict', () => lf.classifier.predict([[0.5, 0.5]]))
  await tryApi('classifier.models', () => lf.classifier.models())

  // Time Series
  console.log('\n--- Time Series ---')
  const ts = Array.from({ length: 100 }, (_, i) => Math.sin(i / 10) + Math.random() * 0.1)
  await tryApi('timeseries.fit', () => lf.timeseries.fit(ts, { model_name: 'demo-ts' }))
  await tryApi('timeseries.predict', () => lf.timeseries.predict(10))
  await tryApi('timeseries.models', () => lf.timeseries.models())
  await tryApi('timeseries.backends', () => lf.timeseries.backends())

  // ADTK
  console.log('\n--- ADTK ---')
  await tryApi('adtk.detectors', () => lf.adtk.detectors())
  await tryApi('adtk.models', () => lf.adtk.models())

  // CatBoost
  console.log('\n--- CatBoost ---')
  await tryApi('catboost.info', () => lf.catboost.info())
  await tryApi('catboost.models', () => lf.catboost.models())
  await tryApi('catboost.fit', () => lf.catboost.fit(X, y, { model_name: 'demo-cb' }))

  // Drift Detection
  console.log('\n--- Drift ---')
  await tryApi('drift.detectors', () => lf.drift.detectors())
  await tryApi('drift.models', () => lf.drift.models())

  // Explainability
  console.log('\n--- Explain ---')
  await tryApi('explain.explainers', () => lf.explain.explainers())
  await tryApi('explain.shap', () => lf.explain.shap([[1, 2, 3]]))

  // Polars
  console.log('\n--- Polars ---')
  await tryApi('polars.buffers', () => lf.polars.buffers())
  await tryApi('polars.createBuffer', () => lf.polars.createBuffer({ name: 'demo', schema: { col1: 'f64', col2: 'f64' } }))

  console.log('\n=== Done ===')
}

main().catch(console.error)
