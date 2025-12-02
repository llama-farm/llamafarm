/**
 * Model Downloader - Handles checking and downloading required ML models
 * Uses the LlamaFarm CLI for all model operations with periodic status checking
 */

import { app } from 'electron'
import * as fs from 'fs'
import * as path from 'path'
import * as yaml from 'js-yaml'
import { exec, spawn, ChildProcess } from 'child_process'
import { promisify } from 'util'
import { promises as fsPromises } from 'fs'

const execAsync = promisify(exec)

/**
 * Validate and sanitize model ID to prevent command injection.
 * Model IDs should only contain alphanumeric characters, hyphens, underscores,
 * forward slashes (for org/repo format), colons (for quantization), and periods.
 */
function validateModelId(modelId: string): string {
  // Only allow safe characters for model IDs
  const safePattern = /^[a-zA-Z0-9\-_\/\.:]+$/
  if (!safePattern.test(modelId)) {
    throw new Error(`Invalid model ID: contains unsafe characters: ${modelId}`)
  }
  // Additional check: no shell metacharacters or sequences
  const dangerousPatterns = ['..', '$(', '`', '|', ';', '&', '>', '<', '\n', '\r']
  for (const pattern of dangerousPatterns) {
    if (modelId.includes(pattern)) {
      throw new Error(`Invalid model ID: contains forbidden sequence: ${modelId}`)
    }
  }
  return modelId
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
  private cliPath: string
  private statusCheckInterval: NodeJS.Timeout | null = null

  constructor(cliPath?: string) {
    // CLI path - use provided or find it
    this.cliPath = cliPath || this.findCLIPath()

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
   * Find the CLI path
   */
  private findCLIPath(): string {
    // Check local installation first
    const userDataPath = app.getPath('userData')
    const localPath = path.join(userDataPath, 'bin', 'lf')

    if (fs.existsSync(localPath)) {
      return localPath
    }

    // Fallback to system PATH
    return 'lf'
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
   * Check if a model is cached using CLI
   */
  async isModelCached(model: ModelConfig): Promise<boolean> {
    const modelId = this.getFullModelId(model)

    try {
      // Validate model ID to prevent command injection
      const safeModelId = validateModelId(modelId)

      // Use lf models status command
      await execAsync(`"${this.cliPath}" models status "${safeModelId}"`, {
        timeout: 30000
      })
      // Exit code 0 means model is cached
      return true
    } catch (error) {
      // Exit code 1 means model is not cached, or validation failed
      return false
    }
  }

  /**
   * Check all required models and return their status
   */
  async checkModels(onProgress?: (progress: ModelDownloadProgress) => void): Promise<ModelStatus[]> {
    const config = await this.loadConfig()
    const statuses: ModelStatus[] = []

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

      const isCached = await this.isModelCached(model)
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
   * Download a model using CLI with periodic status checking
   */
  async downloadModel(
    model: ModelConfig,
    onProgress?: (progress: number, message: string) => void
  ): Promise<void> {
    const modelId = this.getFullModelId(model)

    // Validate model ID to prevent command injection
    const safeModelId = validateModelId(modelId)

    return new Promise((resolve, reject) => {
      console.log(`Starting download for ${safeModelId} via CLI...`)
      onProgress?.(0, `Starting ${model.display_name}...`)

      let downloadProcess: ChildProcess | null = null
      let isComplete = false
      let lastProgressUpdate = Date.now()
      let estimatedProgress = 0

      // Start periodic status checking as backup
      const statusCheckInterval = setInterval(async () => {
        if (isComplete) {
          clearInterval(statusCheckInterval)
          return
        }

        // Check if model is now cached (download completed)
        try {
          const cached = await this.isModelCached(model)
          if (cached && !isComplete) {
            console.log(`Model ${modelId} detected as cached via status check`)
            isComplete = true
            clearInterval(statusCheckInterval)
            onProgress?.(100, 'Complete')

            // Kill the download process if still running
            if (downloadProcess && !downloadProcess.killed) {
              downloadProcess.kill()
            }
            resolve()
            return
          }
        } catch (e) {
          // Ignore status check errors
        }

        // Update estimated progress if no updates from CLI
        const timeSinceUpdate = Date.now() - lastProgressUpdate
        if (timeSinceUpdate > 3000) {
          // Slowly increment progress to show activity (max 95%)
          estimatedProgress = Math.min(95, estimatedProgress + 2)
          onProgress?.(estimatedProgress, `Downloading ${model.display_name}...`)
        }
      }, 2000) // Check every 2 seconds

      // Use spawn to get real-time output
      // Note: safeModelId has been validated to prevent command injection
      downloadProcess = spawn(this.cliPath, ['models', 'pull', safeModelId], {
        shell: true
      })

      downloadProcess.stdout?.on('data', (data: Buffer) => {
        const output = data.toString()
        console.log('CLI output:', output)
        lastProgressUpdate = Date.now()

        // Parse progress from output (e.g., "Progress: 45%")
        const progressMatch = output.match(/Progress:\s*(\d+)%/)
        if (progressMatch) {
          const progress = parseInt(progressMatch[1], 10)
          estimatedProgress = progress
          onProgress?.(progress, `Downloading ${model.display_name}... ${progress}%`)
        }

        // Check for completion indicators
        if (output.includes('Download complete') || output.includes('✓')) {
          estimatedProgress = 100
          onProgress?.(100, 'Complete')
        }
      })

      downloadProcess.stderr?.on('data', (data: Buffer) => {
        console.error('CLI stderr:', data.toString())
      })

      downloadProcess.on('close', (code) => {
        if (isComplete) return // Already resolved via status check

        isComplete = true
        clearInterval(statusCheckInterval)

        if (code === 0) {
          onProgress?.(100, 'Complete')
          resolve()
        } else {
          reject(new Error(`Download failed with exit code ${code}`))
        }
      })

      downloadProcess.on('error', (error) => {
        if (isComplete) return

        isComplete = true
        clearInterval(statusCheckInterval)
        reject(error)
      })

      // Timeout after 30 minutes
      setTimeout(() => {
        if (!isComplete) {
          isComplete = true
          clearInterval(statusCheckInterval)
          if (downloadProcess && !downloadProcess.killed) {
            downloadProcess.kill()
          }
          reject(new Error('Download timed out after 30 minutes'))
        }
      }, 30 * 60 * 1000)
    })
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

    for (const model of config.models) {
      const isCached = await this.isModelCached(model)
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
