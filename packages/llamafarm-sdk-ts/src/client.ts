/**
 * LlamaFarm TypeScript client.
 *
 * @alpha — API may change between releases.
 */

import { ConnectionError, LlamaFarmError, NotFoundError, ServerError, ValidationError } from './errors.js'
import type {
  LlamaFarmOptions,
  ChatMessage,
  ChatOptions,
  ChatResponse,
  StreamChunk,
  Model,
  Project,
  CreateProjectOptions,
  Detection,
  DetectResponse,
  ClassifyResponse,
  FineTuneJob,
  CacheStats,
  RAGQueryRequest,
  RAGQueryResponse,
} from './types.js'

const DEFAULT_URL = 'http://localhost:14345'
const DEFAULT_RUNTIME_URL = 'http://localhost:11540'
const DEFAULT_TIMEOUT = 120_000

// ======================================================================
// LlamaFarm Client
// ======================================================================

export class LlamaFarm {
  readonly url: string
  readonly runtimeUrl: string
  private readonly apiKey?: string
  private readonly timeoutMs: number

  constructor(options: LlamaFarmOptions = {}) {
    this.url = (options.url ?? process.env.LLAMAFARM_URL ?? DEFAULT_URL).replace(/\/$/, '')
    this.runtimeUrl = (options.runtimeUrl ?? process.env.LLAMAFARM_RUNTIME_URL ?? DEFAULT_RUNTIME_URL).replace(/\/$/, '')
    this.apiKey = options.apiKey ?? process.env.LLAMAFARM_API_KEY
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT
  }

  // ------------------------------------------------------------------
  // HTTP helpers
  // ------------------------------------------------------------------

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' }
    if (this.apiKey) h['Authorization'] = `Bearer ${this.apiKey}`
    return h
  }

  private async request(base: string, method: string, path: string, body?: unknown): Promise<unknown> {
    const url = `${base}${path}`
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)

    try {
      const res = await fetch(url, {
        method,
        headers: this.headers(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      })
      clearTimeout(timer)

      if (!res.ok) {
        let detail: string
        try {
          const err = await res.json() as { detail?: string }
          detail = err.detail ?? res.statusText
        } catch {
          detail = res.statusText
        }
        if (res.status === 404) throw new NotFoundError(`Not found: ${detail}`, detail)
        if (res.status === 422) throw new ValidationError(`Validation error: ${detail}`, detail)
        if (res.status >= 500) throw new ServerError(`Server error: ${detail}`, res.status, detail)
        throw new LlamaFarmError(`HTTP ${res.status}: ${detail}`, res.status, detail)
      }

      return await res.json()
    } catch (e) {
      clearTimeout(timer)
      if (e instanceof LlamaFarmError) throw e
      if (e instanceof TypeError && (e.message.includes('fetch') || e.message.includes('connect'))) {
        throw new ConnectionError(`Cannot connect to LlamaFarm at ${base}. Is the server running?`)
      }
      throw e
    }
  }

  private req(method: string, path: string, body?: unknown) {
    return this.request(this.url, method, path, body)
  }

  private runtimeReq(method: string, path: string, body?: unknown) {
    return this.request(this.runtimeUrl, method, path, body)
  }

  // ------------------------------------------------------------------
  // Health
  // ------------------------------------------------------------------

  async health(): Promise<Record<string, unknown>> {
    return this.req('GET', '/health') as Promise<Record<string, unknown>>
  }

  // ------------------------------------------------------------------
  // Models
  // ------------------------------------------------------------------

  async models(): Promise<Model[]> {
    const data = await this.req('GET', '/v1/models') as { data?: Model[]; models?: Model[] }
    return data.data ?? data.models ?? []
  }

  // ------------------------------------------------------------------
  // Projects
  // ------------------------------------------------------------------

  async createProject(namespace: string, name: string, options: CreateProjectOptions = {}): Promise<ProjectHandle> {
    await this.req('POST', `/v1/projects/${namespace}`, {
      name,
      config_template: options.configTemplate ?? null,
    })
    return new ProjectHandle(this, namespace, name)
  }

  project(namespace: string, projectId: string): ProjectHandle {
    return new ProjectHandle(this, namespace, projectId)
  }

  async projects(namespace: string): Promise<Project[]> {
    const data = await this.req('GET', `/v1/projects/${namespace}`) as { projects?: Project[] }
    return data.projects ?? []
  }

  // ------------------------------------------------------------------
  // Vision (server-level)
  // ------------------------------------------------------------------

  get vision(): VisionAPI {
    return new VisionAPI(this)
  }

  // ------------------------------------------------------------------
  // Fine-Tuning (runtime-level)
  // ------------------------------------------------------------------

  get finetune(): FineTuneAPI {
    return new FineTuneAPI(this)
  }

  // ------------------------------------------------------------------
  // Cache (runtime-level)
  // ------------------------------------------------------------------

  get cache(): CacheAPI {
    return new CacheAPI(this)
  }

  // ------------------------------------------------------------------
  // OpenAI Compatibility
  // ------------------------------------------------------------------

  /** Base URL for use with the OpenAI SDK. */
  get openaiBaseUrl(): string { return `${this.runtimeUrl}/v1` }

  // ------------------------------------------------------------------
  // NLP
  // ------------------------------------------------------------------

  get nlp(): NLPAPI { return new NLPAPI(this) }

  // ------------------------------------------------------------------
  // Anomaly Detection
  // ------------------------------------------------------------------

  get anomaly(): AnomalyAPI { return new AnomalyAPI(this) }

  // ------------------------------------------------------------------
  // Classifier
  // ------------------------------------------------------------------

  get classifier(): ClassifierAPI { return new ClassifierAPI(this) }

  // ------------------------------------------------------------------
  // Time Series
  // ------------------------------------------------------------------

  get timeseries(): TimeSeriesAPI { return new TimeSeriesAPI(this) }

  // ------------------------------------------------------------------
  // ADTK
  // ------------------------------------------------------------------

  get adtk(): ADTKAPI { return new ADTKAPI(this) }

  // ------------------------------------------------------------------
  // CatBoost
  // ------------------------------------------------------------------

  get catboost(): CatBoostAPI { return new CatBoostAPI(this) }

  // ------------------------------------------------------------------
  // Drift Detection
  // ------------------------------------------------------------------

  get drift(): DriftAPI { return new DriftAPI(this) }

  // ------------------------------------------------------------------
  // Explainability
  // ------------------------------------------------------------------

  get explain(): ExplainAPI { return new ExplainAPI(this) }

  // ------------------------------------------------------------------
  // Polars Data Processing
  // ------------------------------------------------------------------

  get polars(): PolarsAPI { return new PolarsAPI(this) }

  /** @internal */
  _req(method: string, path: string, body?: unknown) { return this.req(method, path, body) }
  /** @internal */
  _runtimeReq(method: string, path: string, body?: unknown) { return this.runtimeReq(method, path, body) }
  /** @internal */
  async _stream(path: string, body: unknown): Promise<ReadableStream<StreamChunk>> {
    const url = `${this.url}${path}`
    const res = await fetch(url, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
    })
    if (!res.ok || !res.body) throw new ServerError('Stream failed', res.status)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    return new ReadableStream<StreamChunk>({
      async pull(controller) {
        const { done, value } = await reader.read()
        if (done) { controller.close(); return }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data: ')) continue
          const data = trimmed.slice(6)
          if (data === '[DONE]') { controller.close(); return }
          try {
            const parsed = JSON.parse(data)
            const choices = parsed.choices ?? []
            if (choices.length > 0) {
              controller.enqueue({
                content: choices[0].delta?.content ?? '',
                finish_reason: choices[0].finish_reason ?? null,
                model: parsed.model ?? '',
                x_cache: parsed.x_cache,
              })
            }
          } catch { /* skip malformed */ }
        }
      }
    })
  }
}

// ======================================================================
// ProjectHandle
// ======================================================================

export class ProjectHandle {
  constructor(
    private readonly client: LlamaFarm,
    readonly namespace: string,
    readonly projectId: string,
  ) {}

  private path(suffix: string): string {
    return `/v1/projects/${this.namespace}/${this.projectId}${suffix}`
  }

  // Chat
  async chat(message: string | ChatMessage[], options: ChatOptions = {}): Promise<ChatResponse> {
    const messages = typeof message === 'string'
      ? [{ role: 'user' as const, content: message }]
      : message

    const body: Record<string, unknown> = { messages, stream: false }
    if (options.model) body.model = options.model
    if (options.temperature != null) body.temperature = options.temperature
    if (options.max_tokens != null) body.max_tokens = options.max_tokens
    if (options.tools) body.tools = options.tools
    if (options.cache_key) body.cache_key = options.cache_key
    if (options.return_cache_key) body.return_cache_key = true

    const headers: Record<string, string> = {}
    if (options.session_id) headers['X-Session-ID'] = options.session_id
    if (options.stateless) headers['X-No-Session'] = 'true'

    const data = await this.client._req('POST', this.path('/chat/completions'), body) as ChatResponse
    return {
      ...data,
      get text() { return data.choices?.[0]?.message?.content ?? '' },
    }
  }

  async chatStream(message: string | ChatMessage[], options: ChatOptions = {}): Promise<ReadableStream<StreamChunk>> {
    const messages = typeof message === 'string'
      ? [{ role: 'user' as const, content: message }]
      : message

    const body: Record<string, unknown> = { messages, stream: true }
    if (options.model) body.model = options.model
    if (options.temperature != null) body.temperature = options.temperature
    if (options.max_tokens != null) body.max_tokens = options.max_tokens

    return this.client._stream(this.path('/chat/completions'), body)
  }

  // RAG
  get rag(): RAGAPI {
    return new RAGAPI(this.client, this.namespace, this.projectId)
  }

  // Datasets
  get datasets(): DatasetsAPI {
    return new DatasetsAPI(this.client, this.namespace, this.projectId)
  }

  // Delete
  async delete(): Promise<void> {
    await this.client._req('DELETE', this.path(''))
  }
}

// ======================================================================
// Sub-APIs
// ======================================================================

class VisionAPI {
  constructor(private readonly client: LlamaFarm) {}

  async detect(imageBase64: string, options: { model?: string; confidence?: number; classes?: string[] } = {}): Promise<DetectResponse> {
    const body: Record<string, unknown> = { image: imageBase64, model: options.model ?? 'yolov8n', confidence_threshold: options.confidence ?? 0.5 }
    if (options.classes) body.classes = options.classes
    const data = await this.client._req('POST', '/v1/vision/detect', body) as Record<string, unknown>
    return {
      detections: (data.detections ?? data.boxes ?? []) as Detection[],
      model: (data.model ?? options.model ?? 'yolov8n') as string,
      time_ms: (data.time_ms ?? data.detection_time_ms ?? 0) as number,
    }
  }

  async classify(imageBase64: string, classes: string[], options: { model?: string; topK?: number } = {}): Promise<ClassifyResponse> {
    const data = await this.client._req('POST', '/v1/vision/classify', {
      image: imageBase64, model: options.model ?? 'clip-vit-base', classes, top_k: options.topK ?? 3,
    }) as Record<string, unknown>
    return {
      classification: (data.class_name ? data : null) as ClassifyResponse['classification'],
      model: (data.model ?? '') as string,
      time_ms: (data.time_ms ?? 0) as number,
    }
  }
}

class FineTuneAPI {
  constructor(private readonly client: LlamaFarm) {}

  async sft(model: string, dataset: Record<string, unknown>[], options: Record<string, unknown> = {}): Promise<FineTuneJob> {
    return this.client._runtimeReq('POST', '/v1/finetune/sft', { model, dataset, ...options }) as Promise<FineTuneJob>
  }

  async cpt(model: string, dataset: Record<string, unknown>[], options: Record<string, unknown> = {}): Promise<FineTuneJob> {
    return this.client._runtimeReq('POST', '/v1/finetune/cpt', { model, dataset, ...options }) as Promise<FineTuneJob>
  }

  async status(jobId: string): Promise<FineTuneJob> {
    return this.client._runtimeReq('GET', `/v1/finetune/jobs/${jobId}`) as Promise<FineTuneJob>
  }

  async jobs(): Promise<FineTuneJob[]> {
    const data = await this.client._runtimeReq('GET', '/v1/finetune/jobs') as { jobs?: FineTuneJob[] }
    return data.jobs ?? []
  }

  async cancel(jobId: string): Promise<FineTuneJob> {
    return this.client._runtimeReq('DELETE', `/v1/finetune/jobs/${jobId}`) as Promise<FineTuneJob>
  }

  async validate(dataset: Record<string, unknown>[], format?: string): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { dataset }
    if (format) body.dataset_format = format
    return this.client._runtimeReq('POST', '/v1/finetune/validate', body) as Promise<Record<string, unknown>>
  }

  async templates(): Promise<string[]> {
    const data = await this.client._runtimeReq('GET', '/v1/finetune/templates') as { templates?: string[] }
    return data.templates ?? []
  }
}

class CacheAPI {
  constructor(private readonly client: LlamaFarm) {}

  async prepare(messages: ChatMessage[], options: { model?: string; tools?: Record<string, unknown>[]; warm?: boolean } = {}): Promise<Record<string, unknown>> {
    return this.client._runtimeReq('POST', '/v1/cache/prepare', { messages, warm: options.warm ?? true, ...options }) as Promise<Record<string, unknown>>
  }

  async stats(): Promise<CacheStats> {
    return this.client._runtimeReq('GET', '/v1/cache/stats') as Promise<CacheStats>
  }

  async gc(): Promise<Record<string, unknown>> {
    return this.client._runtimeReq('POST', '/v1/cache/gc') as Promise<Record<string, unknown>>
  }

  async clear(key: string): Promise<Record<string, unknown>> {
    return this.client._runtimeReq('DELETE', `/v1/cache/${key}`) as Promise<Record<string, unknown>>
  }
}

class RAGAPI {
  constructor(
    private readonly client: LlamaFarm,
    private readonly namespace: string,
    private readonly project: string,
  ) {}

  private path(suffix: string): string {
    return `/v1/projects/${this.namespace}/${this.project}/rag${suffix}`
  }

  async query(request: RAGQueryRequest): Promise<RAGQueryResponse> {
    return this.client._req('POST', this.path('/query'), request) as Promise<RAGQueryResponse>
  }

  async health(): Promise<Record<string, unknown>> {
    return this.client._req('GET', this.path('/health')) as Promise<Record<string, unknown>>
  }

  async stats(): Promise<Record<string, unknown>> {
    return this.client._req('GET', this.path('/stats')) as Promise<Record<string, unknown>>
  }

  async databases(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', this.path('/databases')) as { databases?: Record<string, unknown>[] }
    return data.databases ?? []
  }
}

// ======================================================================
// NLP API
// ======================================================================

class NLPAPI {
  constructor(private readonly client: LlamaFarm) {}

  async embeddings(input: string | string[], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/nlp/embeddings', { input, ...options }) as Promise<Record<string, unknown>>
  }

  async classify(text: string, labels: string[], options: { model?: string; multi_label?: boolean } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/nlp/classify', { text, labels, ...options }) as Promise<Record<string, unknown>>
  }

  async ner(text: string, options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/nlp/ner', { text, ...options }) as Promise<Record<string, unknown>>
  }

  async rerank(query: string, documents: string[], options: { model?: string; top_k?: number } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/nlp/rerank', { query, documents, ...options }) as Promise<Record<string, unknown>>
  }
}

// ======================================================================
// Anomaly Detection API
// ======================================================================

class AnomalyAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(data: number[][], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/anomaly/fit', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async detect(data: number[][], options: { model?: string; explain?: boolean } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/anomaly/detect', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async score(data: number[][], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/anomaly/score', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/anomaly/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/ml/anomaly/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }

  async backends(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/ml/anomaly/backends') as Record<string, unknown>
    return (data.backends ?? data) as Record<string, unknown>[]
  }
}

// ======================================================================
// Classifier API
// ======================================================================

class ClassifierAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(X: number[][], y: (string | number)[], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/classifier/fit', { X, y, ...options }) as Promise<Record<string, unknown>>
  }

  async predict(X: number[][], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/classifier/predict', { X, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/classifier/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/ml/classifier/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }
}

// ======================================================================
// Time Series API
// ======================================================================

class TimeSeriesAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(data: number[], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/timeseries/fit', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async predict(steps: number, options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/timeseries/predict', { steps, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/timeseries/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/timeseries/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }

  async backends(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/timeseries/backends') as Record<string, unknown>
    return (data.backends ?? data) as Record<string, unknown>[]
  }
}

// ======================================================================
// ADTK API
// ======================================================================

class ADTKAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(data: number[], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/adtk/fit', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async detect(data: number[], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/adtk/detect', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/adtk/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/adtk/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }

  async detectors(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/adtk/detectors') as Record<string, unknown>
    return (data.detectors ?? data) as Record<string, unknown>[]
  }
}

// ======================================================================
// CatBoost API
// ======================================================================

class CatBoostAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(X: number[][], y: (string | number)[], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/catboost/fit', { X, y, ...options }) as Promise<Record<string, unknown>>
  }

  async predict(X: number[][], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/catboost/predict', { X, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/catboost/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async update(X: number[][], y: (string | number)[], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/catboost/update', { X, y, ...options }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/catboost/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }

  async info(): Promise<Record<string, unknown>> {
    return this.client._req('GET', '/v1/catboost/info') as Promise<Record<string, unknown>>
  }

  async importance(modelId: string): Promise<Record<string, unknown>> {
    return this.client._req('GET', `/v1/catboost/${modelId}/importance`) as Promise<Record<string, unknown>>
  }
}

// ======================================================================
// Drift Detection API
// ======================================================================

class DriftAPI {
  constructor(private readonly client: LlamaFarm) {}

  async fit(referenceData: number[][], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/drift/fit', { reference_data: referenceData, ...options }) as Promise<Record<string, unknown>>
  }

  async detect(data: number[][], options: { model?: string } = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/drift/detect', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async load(modelPath: string): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/drift/load', { model_path: modelPath }) as Promise<Record<string, unknown>>
  }

  async models(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/drift/models') as Record<string, unknown>
    return (data.models ?? data) as Record<string, unknown>[]
  }

  async detectors(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/drift/detectors') as Record<string, unknown>
    return (data.detectors ?? data) as Record<string, unknown>[]
  }

  async status(modelName: string): Promise<Record<string, unknown>> {
    return this.client._req('GET', `/v1/drift/status/${modelName}`) as Promise<Record<string, unknown>>
  }
}

// ======================================================================
// Explainability API
// ======================================================================

class ExplainAPI {
  constructor(private readonly client: LlamaFarm) {}

  async shap(data: number[][], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/explain/shap', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async importance(data: number[][], options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/explain/importance', { data, ...options }) as Promise<Record<string, unknown>>
  }

  async explainers(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/explain/explainers') as Record<string, unknown>
    return (data.explainers ?? data) as Record<string, unknown>[]
  }
}

// ======================================================================
// Polars Data Processing API
// ======================================================================

class PolarsAPI {
  constructor(private readonly client: LlamaFarm) {}

  async createBuffer(options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/polars/buffers', options) as Promise<Record<string, unknown>>
  }

  async append(bufferId: string, data: Record<string, unknown>[]): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/polars/append', { buffer_id: bufferId, data }) as Promise<Record<string, unknown>>
  }

  async features(bufferId: string, options: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    return this.client._req('POST', '/v1/ml/polars/features', { buffer_id: bufferId, ...options }) as Promise<Record<string, unknown>>
  }

  async buffers(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', '/v1/ml/polars/buffers') as Record<string, unknown>
    return (data.buffers ?? data) as Record<string, unknown>[]
  }

  async buffer(bufferId: string): Promise<Record<string, unknown>> {
    return this.client._req('GET', `/v1/ml/polars/buffers/${bufferId}`) as Promise<Record<string, unknown>>
  }

  async bufferData(bufferId: string): Promise<Record<string, unknown>> {
    return this.client._req('GET', `/v1/ml/polars/buffers/${bufferId}/data`) as Promise<Record<string, unknown>>
  }
}

class DatasetsAPI {
  constructor(
    private readonly client: LlamaFarm,
    private readonly namespace: string,
    private readonly project: string,
  ) {}

  private path(suffix: string): string {
    return `/v1/projects/${this.namespace}/${this.project}/datasets${suffix}`
  }

  async list(): Promise<Record<string, unknown>[]> {
    const data = await this.client._req('GET', this.path('')) as { datasets?: Record<string, unknown>[] }
    return data.datasets ?? (Array.isArray(data) ? data : []) as Record<string, unknown>[]
  }
}
