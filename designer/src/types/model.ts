/**
 * Model types for LlamaFarm Designer
 */

export interface Model {
  name: string // Internal name (e.g., "fast", "powerful")
  model: string // Actual model ID (e.g., "gemma3:1b")
  provider: string // Provider (e.g., "ollama", "lemonade")
  description?: string // Optional description
  default: boolean // Whether this is the default model
  base_url?: string // Base URL for the provider
  prompt_format?: string // Prompt format
  // Lemonade-specific fields
  lemonade?: {
    backend?: string
    port?: number
    context_size?: number
  }
}

export interface ListModelsResponse {
  total: number
  models: Model[]
}
