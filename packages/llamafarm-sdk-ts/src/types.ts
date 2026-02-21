// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/** Options for creating a LlamaFarm client. */
export interface LlamaFarmOptions {
  /** Server URL (auto-discovered if not provided). */
  url?: string
  /** Universal Runtime URL for direct access (fine-tuning, cache). */
  runtimeUrl?: string
  /** API key for authentication. */
  apiKey?: string
  /** Request timeout in milliseconds. */
  timeoutMs?: number
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  name?: string
  tool_calls?: Record<string, unknown>[]
  tool_call_id?: string
}

export interface ChatChoice {
  index: number
  message: ChatMessage
  finish_reason: string | null
}

export interface ChatUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatResponse {
  id: string
  object: string
  model: string
  choices: ChatChoice[]
  usage: ChatUsage
  /** LlamaFarm KV cache info. */
  x_cache?: Record<string, unknown>
  /** Shortcut: first choice content. */
  readonly text: string
}

export interface StreamChunk {
  content: string
  finish_reason: string | null
  model: string
  x_cache?: Record<string, unknown>
}

export interface ChatOptions {
  model?: string
  temperature?: number
  max_tokens?: number
  tools?: Record<string, unknown>[]
  /** KV cache key from a previous response. */
  cache_key?: string
  /** Request a cache key in the response. */
  return_cache_key?: boolean
  /** Session ID for stateful conversations. */
  session_id?: string
  /** Stateless mode (no session persistence). */
  stateless?: boolean
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface Model {
  id: string
  object: string
  owned_by: string
  type?: string
  size_bytes?: number
  context_length?: number
  loaded: boolean
}

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

export interface ProjectConfig {
  model: string
  provider: string
  system_prompt: string
  temperature: number
  max_tokens?: number
  tools: Record<string, unknown>[]
  rag?: Record<string, unknown>
}

export interface Project {
  namespace: string
  name: string
  config: ProjectConfig | Record<string, unknown>
  validation_error?: string
  last_modified?: string
}

export interface CreateProjectOptions {
  /** Model name (e.g., "Qwen/Qwen3-8B"). */
  model?: string
  /** System prompt. */
  systemPrompt?: string
  /** Config template name. */
  configTemplate?: string
  /** Sampling temperature. */
  temperature?: number
}

// ---------------------------------------------------------------------------
// Vision
// ---------------------------------------------------------------------------

export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  class_name: string
  class_id: number
  confidence: number
}

export interface Classification {
  class_name: string
  confidence: number
  all_scores: Record<string, number>
}

export interface DetectResponse {
  detections: Detection[]
  model: string
  time_ms: number
}

export interface ClassifyResponse {
  classification: Classification | null
  model: string
  time_ms: number
}

// ---------------------------------------------------------------------------
// Fine-Tuning
// ---------------------------------------------------------------------------

export interface FineTuneJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  type: 'sft' | 'cpt'
  model: string
  progress: number
  metrics?: Record<string, unknown>
  output_dir?: string
  error?: string
  created_at: number
  started_at?: number
  completed_at?: number
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

export interface CacheStats {
  entries: number
  total_bytes: number
  hit_rate: number
  ram_bytes: number
  disk_bytes: number
}

// ---------------------------------------------------------------------------
// RAG
// ---------------------------------------------------------------------------

export interface RAGQueryRequest {
  query: string
  database?: string
  retrieval_strategy?: string
  top_k?: number
  score_threshold?: number
  metadata_filters?: Record<string, unknown>
  hybrid_alpha?: number
  rerank_model?: string
  query_expansion?: boolean
}

export interface RAGQueryResult {
  content: string
  score: number
  metadata: Record<string, unknown>
  chunk_id?: string
  document_id?: string
}

export interface RAGQueryResponse {
  query: string
  results: RAGQueryResult[]
  total_results: number
  processing_time_ms?: number
  retrieval_strategy_used: string
  database_used: string
}
