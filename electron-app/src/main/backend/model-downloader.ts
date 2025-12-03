/**
 * Model Downloader - Handles checking and downloading required ML models
 * Communicates directly with the server's SSE endpoint for reliable progress tracking
 */

import { app } from 'electron'
import * as fs from 'fs'
import * as path from 'path'
import * as yaml from 'js-yaml'
import { promises as fsPromises } from 'fs'

// Server URL - defaults to localhost:8000
const DEFAULT_SERVER_URL = 'http://127.0.0.1:8000'

/**
 * SSE event structure from the server's download endpoint
 */
interface DownloadEvent {
  event: 'start' | 'progress' | 'end' | 'done' | 'error'
  desc?: string
  total?: number
  n?: number
  message?: string
}

/**
 * Model info from the server's list endpoint
 */
interface CachedModel {
  id: string
  name: string
  path?: string
  size_on_disk?: number
}

/**
 * Parse SSE event from a data line
 */
function parseSseEvent(line: string): DownloadEvent | null {
  const trimmed = line.trim()
  if (!trimmed.startsWith('data: ')) {
    return null
  }
  try {
    return JSON.parse(trimmed.slice(6)) as DownloadEvent
  } catch {
    return null
  }
}

export interface ModelConfig {
  id: string
  quantization?: string
  display_name: string
  type: 'language' | 'embedding'
  required: boolean
  size_estimate_mb: number
}

export interface RequiredModelsConfig {
  version: string
  models: ModelConfig[]
}

export interface ModelStatus {
  id: string
  display_name: string
  status: 'checking' | 'present' | 'downloading' | 'error'
  progress?: number
  error?: string
  size_estimate_mb: number
}

export interface ModelDownloadProgress {
  models: ModelStatus[]
  overall_progress: number
  current_model?: string
  message: string
}

export class ModelDownloader {
  private config: RequiredModelsConfig | null = null
  private configPath: string
  private serverUrl: string

  constructor(serverUrl?: string) {
    // Server URL for direct API communication
    this.serverUrl = serverUrl || DEFAULT_SERVER_URL

    // Config file location - check multiple paths
    const possiblePaths = [
      path.join(__dirname, '../../required-models.yaml'),
      path.join(__dirname, '../../../required-models.yaml'),
      path.join(app.getAppPath(), 'required-models.yaml'),
      path.join(process.cwd(), 'required-models.yaml')
    ]

    this.configPath = possiblePaths.find(p => fs.existsSync(p)) || possiblePaths[0]
  }

  /**
   * Load configuration from YAML file
   */
  async loadConfig(): Promise<RequiredModelsConfig> {
    if (this.config) return this.config

    try {
      const content = await fsPromises.readFile(this.configPath, 'utf8')
      this.config = yaml.load(content) as RequiredModelsConfig
      console.log('Loaded model config from:', this.configPath)
      return this.config
    } catch (error) {
      console.error('Failed to load model config:', error)
      // Return a default config with models from config templates
      this.config = {
        version: '1',
        models: [
          {
            id: 'unsloth/gemma-3-1b-it-gguf',
            quantization: 'Q4_K_M',
            display_name: 'Gemma 3 1B',
            type: 'language',
            required: true,
            size_estimate_mb: 700
          },
          {
            id: 'unsloth/Qwen3-1.7B-GGUF',
            quantization: 'Q4_K_M',
            display_name: 'Qwen3 1.7B',
            type: 'language',
            required: true,
            size_estimate_mb: 1200
          },
          {
            id: 'nomic-ai/nomic-embed-text-v1.5',
            display_name: 'Nomic Embed v1.5',
            type: 'embedding',
            required: true,
            size_estimate_mb: 550
          }
        ]
      }
      return this.config
    }
  }

  /**
   * Get full model ID with quantization
   */
  private getFullModelId(model: ModelConfig): string {
    return model.quantization ? `${model.id}:${model.quantization}` : model.id
  }

  /**
   * Fetch the list of cached models from the server
   */
  private async fetchCachedModels(): Promise<CachedModel[]> {
    try {
      const response = await fetch(`${this.serverUrl}/v1/models`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: AbortSignal.timeout(30000)
      })

      if (!response.ok) {
        console.error(`Failed to fetch model list: HTTP ${response.status}`)
        return []
      }

      const result = await response.json() as { data: CachedModel[] }
      return result.data
    } catch (error) {
      console.error('Error fetching model list:', error)
      return []
    }
  }

  /**
   * Check if a model is cached against a provided list of cached models
   */
  private isModelInCache(model: ModelConfig, cachedModels: CachedModel[]): boolean {
    const modelId = this.getFullModelId(model)
    // Strip quantization suffix for comparison (e.g., "org/repo:Q4_K_M" -> "org/repo")
    const baseModelId = modelId.includes(':') ? modelId.split(':')[0] : modelId

    // Check if model is in the cache (compare with or without quantization)
    return cachedModels.some(
      cached => cached.id === baseModelId || cached.id === modelId
    )
  }

  /**
   * Check if a model is cached using the server's models API
   * @deprecated Use checkModels() or getModelStatus() instead to avoid redundant API calls
   */
  async isModelCached(model: ModelConfig): Promise<boolean> {
    const cachedModels = await this.fetchCachedModels()
    return this.isModelInCache(model, cachedModels)
  }

  /**
   * Check all required models and return their status
   */
  async checkModels(onProgress?: (progress: ModelDownloadProgress) => void): Promise<ModelStatus[]> {
    const config = await this.loadConfig()
    const statuses: ModelStatus[] = []

    // Fetch the cached models list ONCE for all checks
    const cachedModels = await this.fetchCachedModels()

    for (let i = 0; i < config.models.length; i++) {
      const model = config.models[i]
      const status: ModelStatus = {
        id: model.id,
        display_name: model.display_name,
        status: 'checking',
        size_estimate_mb: model.size_estimate_mb
      }
      statuses.push(status)

      onProgress?.({
        models: [...statuses],
        overall_progress: ((i + 0.5) / config.models.length) * 30,
        current_model: model.display_name,
        message: `Checking ${model.display_name}...`
      })

      const isCached = this.isModelInCache(model, cachedModels)
      status.status = isCached ? 'present' : 'downloading'

      onProgress?.({
        models: [...statuses],
        overall_progress: ((i + 1) / config.models.length) * 30,
        current_model: model.display_name,
        message: isCached ? `${model.display_name} ✓` : `${model.display_name} needs download`
      })
    }

    return statuses
  }

  /**
   * Download a model using the server's SSE endpoint for reliable progress tracking
   */
  async downloadModel(
    model: ModelConfig,
    onProgress?: (progress: number, message: string) => void
  ): Promise<void> {
    const modelId = this.getFullModelId(model)

    console.log(`Starting download for ${modelId} via server SSE...`)
    onProgress?.(0, `Starting ${model.display_name}...`)

    // Create abort controller for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 30 * 60 * 1000) // 30 min timeout

    try {
      const response = await fetch(`${this.serverUrl}/v1/models/download`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          provider: 'universal',
          model_name: modelId
        }),
        signal: controller.signal
      })

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`)
      }

      if (!response.body) {
        throw new Error('No response body received')
      }

      // Read and parse the SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let currentDesc = ''
      let currentTotal = 0
      let receivedDone = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Append new data to buffer and process complete lines
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          const event = parseSseEvent(line)
          if (!event) continue

          switch (event.event) {
            case 'start':
              currentDesc = event.desc || model.display_name
              currentTotal = event.total || 0
              if (currentTotal > 1024 * 1024) {
                const totalMB = (currentTotal / 1024 / 1024).toFixed(1)
                onProgress?.(0, `${currentDesc} (${totalMB} MB)...`)
              } else {
                onProgress?.(0, `${currentDesc}...`)
              }
              break

            case 'progress':
              if (currentTotal > 0 && event.n !== undefined) {
                const progress = Math.round((event.n / currentTotal) * 100)
                onProgress?.(progress, `Downloading ${model.display_name}... ${progress}%`)
              }
              break

            case 'end':
              // File completed, progress continues with next file
              break

            case 'done':
              receivedDone = true
              onProgress?.(100, 'Complete')
              clearTimeout(timeoutId)
              return

            case 'error':
              throw new Error(event.message || 'Download failed')
          }
        }
      }

      // If we reach here without 'done', the download was incomplete
      if (!receivedDone) {
        throw new Error('Download incomplete: stream ended without completion signal')
      }
      onProgress?.(100, 'Complete')
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Download timed out after 30 minutes')
      }
      throw error
    } finally {
      clearTimeout(timeoutId)
    }
  }

  /**
   * Ensure all required models are downloaded
   */
  async ensureModels(
    onProgress?: (progress: ModelDownloadProgress) => void
  ): Promise<{ success: boolean; models: ModelStatus[] }> {
    const config = await this.loadConfig()
    const statuses = await this.checkModels(onProgress)

    // Find models that need downloading
    const modelsToDownload = statuses.filter(s => s.status === 'downloading')

    if (modelsToDownload.length === 0) {
      onProgress?.({
        models: statuses,
        overall_progress: 100,
        message: 'All models ready!'
      })
      return { success: true, models: statuses }
    }

    // Calculate total size to download for better progress estimates
    const totalSizeToDownload = modelsToDownload.reduce((sum, m) => sum + m.size_estimate_mb, 0)
    let downloadedSize = 0

    // Download missing models
    for (const modelStatus of modelsToDownload) {
      const modelConfig = config.models.find(m => m.id === modelStatus.id)
      if (!modelConfig) continue

      const modelStartProgress = 30 + (downloadedSize / totalSizeToDownload) * 70

      onProgress?.({
        models: statuses,
        overall_progress: modelStartProgress,
        current_model: modelStatus.display_name,
        message: `Downloading ${modelStatus.display_name}...`
      })

      try {
        await this.downloadModel(modelConfig, (progress, message) => {
          modelStatus.progress = progress

          // Calculate overall progress based on model size
          const modelProgress = (progress / 100) * modelConfig.size_estimate_mb
          const overallDownloaded = downloadedSize + modelProgress
          const overallProgress = 30 + (overallDownloaded / totalSizeToDownload) * 70

          onProgress?.({
            models: [...statuses],
            overall_progress: Math.min(99, overallProgress),
            current_model: modelStatus.display_name,
            message: `${modelStatus.display_name}: ${message}`
          })
        })

        modelStatus.status = 'present'
        modelStatus.progress = 100
        downloadedSize += modelConfig.size_estimate_mb
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Download failed'
        modelStatus.status = 'error'
        modelStatus.error = errorMsg
        console.error(`Failed to download ${modelStatus.display_name}:`, error)
      }
    }

    const allSuccess = statuses.every(s => s.status === 'present')

    onProgress?.({
      models: statuses,
      overall_progress: 100,
      message: allSuccess ? 'All models ready!' : 'Some models failed to download'
    })

    return { success: allSuccess, models: statuses }
  }

  /**
   * Get current model status without downloading
   */
  async getModelStatus(): Promise<ModelStatus[]> {
    const config = await this.loadConfig()
    const statuses: ModelStatus[] = []

    // Fetch the cached models list ONCE for all checks
    const cachedModels = await this.fetchCachedModels()

    for (const model of config.models) {
      const isCached = this.isModelInCache(model, cachedModels)
      statuses.push({
        id: model.id,
        display_name: model.display_name,
        status: isCached ? 'present' : 'downloading',
        size_estimate_mb: model.size_estimate_mb
      })
    }

    return statuses
  }
}
