import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { LlamaFarm } from '../src/client.js'
import { LlamaFarmError, ConnectionError, NotFoundError, ValidationError, ServerError } from '../src/errors.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockFetch(status: number, body: unknown, headers?: Record<string, string>) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: `Status ${status}`,
    headers: new Headers(headers),
    json: () => Promise.resolve(body),
    body: null,
  })
}

function mockStream(chunks: string[]) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    body: new ReadableStream({
      start(controller) {
        for (const c of chunks) controller.enqueue(new TextEncoder().encode(c))
        controller.close()
      },
    }),
  })
}

// ---------------------------------------------------------------------------
// Constructor & Discovery
// ---------------------------------------------------------------------------

describe('LlamaFarm constructor', () => {
  const orig = process.env

  beforeEach(() => { process.env = { ...orig } })
  afterEach(() => { process.env = orig })

  it('uses defaults', () => {
    const lf = new LlamaFarm()
    expect(lf.url).toBe('http://localhost:14345')
    expect(lf.runtimeUrl).toBe('http://localhost:11540')
  })

  it('reads env vars', () => {
    process.env.LLAMAFARM_URL = 'http://my-server:9000/'
    process.env.LLAMAFARM_RUNTIME_URL = 'http://my-runtime:9001/'
    const lf = new LlamaFarm()
    expect(lf.url).toBe('http://my-server:9000')
    expect(lf.runtimeUrl).toBe('http://my-runtime:9001')
  })

  it('explicit options override env', () => {
    process.env.LLAMAFARM_URL = 'http://env:1'
    const lf = new LlamaFarm({ url: 'http://opt:2' })
    expect(lf.url).toBe('http://opt:2')
  })

  it('strips trailing slash', () => {
    const lf = new LlamaFarm({ url: 'http://host:1234/' })
    expect(lf.url).toBe('http://host:1234')
  })
})

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe('error handling', () => {
  it('throws NotFoundError on 404', async () => {
    vi.stubGlobal('fetch', mockFetch(404, { detail: 'gone' }))
    const lf = new LlamaFarm()
    await expect(lf.health()).rejects.toThrow(NotFoundError)
    vi.unstubAllGlobals()
  })

  it('throws ValidationError on 422', async () => {
    vi.stubGlobal('fetch', mockFetch(422, { detail: 'bad input' }))
    const lf = new LlamaFarm()
    await expect(lf.health()).rejects.toThrow(ValidationError)
    vi.unstubAllGlobals()
  })

  it('throws ServerError on 500', async () => {
    vi.stubGlobal('fetch', mockFetch(500, { detail: 'boom' }))
    const lf = new LlamaFarm()
    await expect(lf.health()).rejects.toThrow(ServerError)
    vi.unstubAllGlobals()
  })

  it('error hierarchy extends LlamaFarmError', () => {
    expect(new NotFoundError('x')).toBeInstanceOf(LlamaFarmError)
    expect(new ValidationError('x')).toBeInstanceOf(LlamaFarmError)
    expect(new ServerError('x')).toBeInstanceOf(LlamaFarmError)
    expect(new ConnectionError('x')).toBeInstanceOf(LlamaFarmError)
  })
})

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

describe('health', () => {
  it('returns health data', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { status: 'ok', version: '0.0.27' }))
    const lf = new LlamaFarm()
    const h = await lf.health()
    expect(h.status).toBe('ok')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

describe('models', () => {
  it('parses models list (data key)', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { data: [{ id: 'qwen3-8b', object: 'model', owned_by: 'local', loaded: true }] }))
    const lf = new LlamaFarm()
    const models = await lf.models()
    expect(models).toHaveLength(1)
    expect(models[0].id).toBe('qwen3-8b')
    vi.unstubAllGlobals()
  })

  it('handles models key fallback', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { models: [{ id: 'phi-4' }] }))
    const lf = new LlamaFarm()
    const models = await lf.models()
    expect(models[0].id).toBe('phi-4')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

describe('projects', () => {
  it('creates project and returns handle', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { name: 'my-app' }))
    const lf = new LlamaFarm()
    const p = await lf.createProject('default', 'my-app')
    expect(p.namespace).toBe('default')
    expect(p.projectId).toBe('my-app')
    vi.unstubAllGlobals()
  })

  it('project() returns lazy handle (no API call)', () => {
    const lf = new LlamaFarm()
    const p = lf.project('ns', 'pid')
    expect(p.namespace).toBe('ns')
    expect(p.projectId).toBe('pid')
  })

  it('lists projects', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { projects: [{ name: 'a' }, { name: 'b' }] }))
    const lf = new LlamaFarm()
    const ps = await lf.projects('default')
    expect(ps).toHaveLength(2)
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

describe('chat', () => {
  it('sends string message and gets text shortcut', async () => {
    const mockResp = {
      id: 'chat-1',
      object: 'chat.completion',
      model: 'qwen3-8b',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Hello back!' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    }
    vi.stubGlobal('fetch', mockFetch(200, mockResp))
    const lf = new LlamaFarm()
    const p = lf.project('default', 'test')
    const resp = await p.chat('Hello!')
    expect(resp.text).toBe('Hello back!')
    expect(resp.model).toBe('qwen3-8b')
    vi.unstubAllGlobals()
  })

  it('sends message array', async () => {
    const f = mockFetch(200, { choices: [{ message: { content: 'ok' } }] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    const p = lf.project('ns', 'pid')
    await p.chat([{ role: 'system', content: 'be helpful' }, { role: 'user', content: 'hi' }])
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.messages).toHaveLength(2)
    expect(body.messages[0].role).toBe('system')
    vi.unstubAllGlobals()
  })

  it('includes optional chat params', async () => {
    const f = mockFetch(200, { choices: [{ message: { content: '' } }] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.project('ns', 'p').chat('hi', {
      model: 'phi-4',
      temperature: 0.7,
      max_tokens: 100,
      cache_key: 'abc123',
      return_cache_key: true,
    })
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.model).toBe('phi-4')
    expect(body.temperature).toBe(0.7)
    expect(body.max_tokens).toBe(100)
    expect(body.cache_key).toBe('abc123')
    expect(body.return_cache_key).toBe(true)
    vi.unstubAllGlobals()
  })

  it('builds correct URL path', async () => {
    const f = mockFetch(200, { choices: [{ message: { content: '' } }] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.project('acme', 'agent').chat('hi')
    const url = f.mock.calls[0][0]
    expect(url).toBe('http://localhost:14345/v1/projects/acme/agent/chat/completions')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe('streaming', () => {
  it('parses SSE chunks', async () => {
    const sseData = [
      'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}],"model":"q3"}\n\n',
      'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}],"model":"q3"}\n\n',
      'data: [DONE]\n\n',
    ]
    vi.stubGlobal('fetch', mockStream(sseData))
    const lf = new LlamaFarm()
    const stream = await lf.project('ns', 'p').chatStream('hi')
    const reader = stream.getReader()
    const chunks: string[] = []
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value.content)
    }
    expect(chunks.join('')).toBe('Hello world')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Vision
// ---------------------------------------------------------------------------

describe('vision', () => {
  it('detect returns detections', async () => {
    vi.stubGlobal('fetch', mockFetch(200, {
      detections: [{ x1: 0, y1: 0, x2: 100, y2: 100, class_name: 'person', class_id: 0, confidence: 0.95 }],
      model: 'yolov8n',
      time_ms: 42,
    }))
    const lf = new LlamaFarm()
    const r = await lf.vision.detect('base64data')
    expect(r.detections).toHaveLength(1)
    expect(r.detections[0].class_name).toBe('person')
    vi.unstubAllGlobals()
  })

  it('classify returns classification', async () => {
    vi.stubGlobal('fetch', mockFetch(200, {
      class_name: 'cat',
      confidence: 0.92,
      all_scores: { cat: 0.92, dog: 0.08 },
      model: 'clip-vit-base',
      time_ms: 15,
    }))
    const lf = new LlamaFarm()
    const r = await lf.vision.classify('base64data', ['cat', 'dog'])
    expect(r.classification?.class_name).toBe('cat')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Fine-Tuning
// ---------------------------------------------------------------------------

describe('finetune', () => {
  it('submits SFT job', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { job_id: 'ft-1', status: 'queued', type: 'sft' }))
    const lf = new LlamaFarm()
    const job = await lf.finetune.sft('model-id', [{ instruction: 'hi', output: 'hello' }])
    expect(job.job_id).toBe('ft-1')
    // Verify it hits runtime URL
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(url).toContain('localhost:11540')
    vi.unstubAllGlobals()
  })

  it('lists jobs', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { jobs: [{ job_id: 'ft-1' }, { job_id: 'ft-2' }] }))
    const lf = new LlamaFarm()
    const jobs = await lf.finetune.jobs()
    expect(jobs).toHaveLength(2)
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

describe('cache', () => {
  it('returns stats', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { entries: 5, total_bytes: 1024, hit_rate: 0.8, ram_bytes: 512, disk_bytes: 512 }))
    const lf = new LlamaFarm()
    const s = await lf.cache.stats()
    expect(s.entries).toBe(5)
    expect(s.hit_rate).toBe(0.8)
    vi.unstubAllGlobals()
  })

  it('prepare hits runtime URL', async () => {
    const f = mockFetch(200, { cache_key: 'abc', cached_tokens: 100 })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.cache.prepare([{ role: 'system', content: 'be helpful' }])
    const url = f.mock.calls[0][0]
    expect(url).toContain('localhost:11540')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// RAG
// ---------------------------------------------------------------------------

describe('rag', () => {
  it('queries RAG via project', async () => {
    vi.stubGlobal('fetch', mockFetch(200, {
      query: 'test', results: [{ content: 'answer', score: 0.9 }],
      total_results: 1, retrieval_strategy_used: 'hybrid', database_used: 'default',
    }))
    const lf = new LlamaFarm()
    const r = await lf.project('ns', 'p').rag.query({ query: 'test' })
    expect(r.results).toHaveLength(1)
    // Verify URL path
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(url).toContain('/v1/projects/ns/p/rag/query')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

describe('datasets', () => {
  it('lists datasets via project', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { datasets: [{ name: 'ds1' }] }))
    const lf = new LlamaFarm()
    const ds = await lf.project('ns', 'p').datasets.list()
    expect(ds).toHaveLength(1)
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// OpenAI Compatibility
// ---------------------------------------------------------------------------

describe('openaiBaseUrl', () => {
  it('returns runtime URL with /v1', () => {
    const lf = new LlamaFarm()
    expect(lf.openaiBaseUrl).toBe('http://localhost:11540/v1')
  })

  it('uses custom runtime URL', () => {
    const lf = new LlamaFarm({ runtimeUrl: 'http://gpu:8080' })
    expect(lf.openaiBaseUrl).toBe('http://gpu:8080/v1')
  })
})

// ---------------------------------------------------------------------------
// NLP API
// ---------------------------------------------------------------------------

describe('nlp', () => {
  it('embeddings sends correct request', async () => {
    const f = mockFetch(200, { embeddings: [[0.1, 0.2]], model: 'bge-small' })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    const r = await lf.nlp.embeddings('hello')
    expect(r.embeddings).toBeDefined()
    const url = f.mock.calls[0][0]
    expect(url).toContain('/v1/nlp/embeddings')
    vi.unstubAllGlobals()
  })

  it('classify sends text and labels', async () => {
    const f = mockFetch(200, { label: 'positive', scores: { positive: 0.9, negative: 0.1 } })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.nlp.classify('great movie', ['positive', 'negative'])
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.text).toBe('great movie')
    expect(body.labels).toEqual(['positive', 'negative'])
    vi.unstubAllGlobals()
  })

  it('ner extracts entities', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { entities: [{ text: 'Paris', label: 'LOC' }] }))
    const lf = new LlamaFarm()
    const r = await lf.nlp.ner('I live in Paris')
    expect(r.entities).toBeDefined()
    vi.unstubAllGlobals()
  })

  it('rerank sends query and documents', async () => {
    const f = mockFetch(200, { results: [{ index: 0, score: 0.95 }] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.nlp.rerank('query', ['doc1', 'doc2'])
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.query).toBe('query')
    expect(body.documents).toEqual(['doc1', 'doc2'])
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Anomaly Detection API
// ---------------------------------------------------------------------------

describe('anomaly', () => {
  it('detect sends data with options', async () => {
    const f = mockFetch(200, { anomalies: [0, 1, 0], scores: [-0.1, 2.5, -0.2] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.anomaly.detect([[1, 2], [99, 99], [1, 3]], { explain: true })
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.data).toHaveLength(3)
    expect(body.explain).toBe(true)
    vi.unstubAllGlobals()
  })

  it('models returns list', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { models: [{ id: 'iforest-1' }] }))
    const lf = new LlamaFarm()
    const m = await lf.anomaly.models()
    expect(m).toHaveLength(1)
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Classifier API
// ---------------------------------------------------------------------------

describe('classifier', () => {
  it('fit sends X and y', async () => {
    const f = mockFetch(200, { model_id: 'clf-1', accuracy: 0.95 })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.classifier.fit([[1], [2], [3]], [0, 1, 0])
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.X).toHaveLength(3)
    expect(body.y).toEqual([0, 1, 0])
    vi.unstubAllGlobals()
  })

  it('predict returns predictions', async () => {
    vi.stubGlobal('fetch', mockFetch(200, { predictions: [0, 1] }))
    const lf = new LlamaFarm()
    const r = await lf.classifier.predict([[1], [2]])
    expect(r.predictions).toBeDefined()
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Time Series API
// ---------------------------------------------------------------------------

describe('timeseries', () => {
  it('predict sends steps', async () => {
    const f = mockFetch(200, { forecast: [1.1, 1.2, 1.3] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.timeseries.predict(3)
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.steps).toBe(3)
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// CatBoost API
// ---------------------------------------------------------------------------

describe('catboost', () => {
  it('importance hits correct URL', async () => {
    const f = mockFetch(200, { importance: [0.5, 0.3, 0.2] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.catboost.importance('my-model')
    const url = f.mock.calls[0][0]
    expect(url).toContain('/v1/catboost/my-model/importance')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Drift Detection API
// ---------------------------------------------------------------------------

describe('drift', () => {
  it('status hits correct URL', async () => {
    const f = mockFetch(200, { drifted: false, p_value: 0.45 })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.drift.status('prod-model')
    const url = f.mock.calls[0][0]
    expect(url).toContain('/v1/drift/status/prod-model')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Explainability API
// ---------------------------------------------------------------------------

describe('explain', () => {
  it('shap sends data', async () => {
    const f = mockFetch(200, { shap_values: [[0.1, -0.2]] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.explain.shap([[1, 2]])
    const body = JSON.parse(f.mock.calls[0][1].body)
    expect(body.data).toEqual([[1, 2]])
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Polars API
// ---------------------------------------------------------------------------

describe('polars', () => {
  it('bufferData hits correct URL', async () => {
    const f = mockFetch(200, { data: [{ col1: 1 }] })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.polars.bufferData('buf-123')
    const url = f.mock.calls[0][0]
    expect(url).toContain('/v1/ml/polars/buffers/buf-123/data')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Auth header
// ---------------------------------------------------------------------------

describe('auth', () => {
  it('sends Bearer token when apiKey set', async () => {
    const f = mockFetch(200, { status: 'ok' })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm({ apiKey: 'sk-test-123' })
    await lf.health()
    const headers = f.mock.calls[0][1].headers
    expect(headers['Authorization']).toBe('Bearer sk-test-123')
    vi.unstubAllGlobals()
  })

  it('no auth header without apiKey', async () => {
    const f = mockFetch(200, { status: 'ok' })
    vi.stubGlobal('fetch', f)
    const lf = new LlamaFarm()
    await lf.health()
    const headers = f.mock.calls[0][1].headers
    expect(headers['Authorization']).toBeUndefined()
    vi.unstubAllGlobals()
  })
})
