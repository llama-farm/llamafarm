/**
 * Backend Manager - Handles LlamaFarm backend lifecycle
 * Manages starting, stopping, monitoring, and recovery of backend services
 */

import { ChildProcess, spawn } from 'child_process'
import * as path from 'path'
import { app } from 'electron'
import { HealthChecker, HealthStatus } from './health-checker'

export interface BackendStatus {
  state: 'stopped' | 'starting' | 'running' | 'degraded' | 'error' | 'stopping'
  message: string
  health?: HealthStatus
  error?: string
  port?: number
}

export interface BackendConfig {
  cliPath: string
  projectPath?: string
  autoRestart?: boolean
  maxRestartAttempts?: number
}

export class BackendManager {
  private process: ChildProcess | null = null
  private config: BackendConfig
  private healthChecker: HealthChecker
  private status: BackendStatus
  private restartAttempts = 0
  private readonly MAX_RESTART_ATTEMPTS = 3
  private restartTimeout?: NodeJS.Timeout
  private onStatusChange?: (status: BackendStatus) => void
  private startupTimeout?: NodeJS.Timeout
  private readonly STARTUP_TIMEOUT = 180000 // 3 minutes

  constructor(config: BackendConfig) {
    this.config = {
      autoRestart: true,
      maxRestartAttempts: 3,
      ...config
    }
    this.healthChecker = new HealthChecker('http://localhost:8000')
    this.status = {
      state: 'stopped',
      message: 'Backend not started'
    }
  }

  /**
   * Set status change callback
   */
  onStatus(callback: (status: BackendStatus) => void): void {
    this.onStatusChange = callback
  }

  /**
   * Update and emit status
   */
  private updateStatus(newStatus: Partial<BackendStatus>): void {
    this.status = { ...this.status, ...newStatus }
    this.onStatusChange?.({ ...this.status })
  }

  /**
   * Get current status
   */
  getStatus(): BackendStatus {
    return { ...this.status }
  }

  /**
   * Start the backend
   */
  async start(): Promise<void> {
    if (this.process) {
      console.log('Backend already running')
      return
    }

    this.updateStatus({
      state: 'starting',
      message: 'Starting LlamaFarm backend...'
    })

    try {
      await this.startBackendProcess()
      await this.waitForHealthy()

      this.updateStatus({
        state: 'running',
        message: 'Backend running',
        port: 8000
      })

      // Reset restart attempts on successful start
      this.restartAttempts = 0

      // Start health monitoring
      this.startHealthMonitoring()
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.updateStatus({
        state: 'error',
        message: 'Failed to start backend',
        error: errorMessage
      })

      // Cleanup
      await this.stop()

      // Auto-restart if enabled
      if (this.config.autoRestart && this.shouldAttemptRestart()) {
        this.scheduleRestart()
      }

      throw error
    }
  }

  /**
   * Start the backend process (lf start equivalent)
   */
  private async startBackendProcess(): Promise<void> {
    return new Promise((resolve, reject) => {
      const cwd = this.config.projectPath || app.getPath('home')

      console.log('Starting backend process:', this.config.cliPath)
      console.log('Working directory:', cwd)

      // Spawn the lf start process
      // We'll use a custom approach similar to lf start but more controlled
      this.process = spawn(this.config.cliPath, ['start', '--no-tui'], {
        cwd,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          LF_DESKTOP_MODE: '1', // Signal we're running in desktop mode
          ELECTRON_RUN_AS_NODE: undefined, // Prevent Electron env pollution
        },
        detached: false
      })

      // Handle stdout
      this.process.stdout?.on('data', (data) => {
        const output = data.toString()
        console.log('[Backend]', output)

        // Look for startup indicators
        if (output.includes('Application startup complete') ||
            output.includes('Uvicorn running on')) {
          this.updateStatus({
            message: 'Backend services started, waiting for health check...'
          })
        }
      })

      // Handle stderr
      this.process.stderr?.on('data', (data) => {
        const error = data.toString()
        console.error('[Backend Error]', error)
      })

      // Handle process exit
      this.process.on('exit', (code, signal) => {
        console.log(`Backend process exited with code ${code}, signal ${signal}`)
        this.process = null

        if (this.startupTimeout) {
          clearTimeout(this.startupTimeout)
        }

        if (this.status.state === 'starting') {
          reject(new Error(`Backend process exited during startup (code: ${code})`))
        } else if (this.status.state === 'running' && this.config.autoRestart) {
          this.updateStatus({
            state: 'error',
            message: 'Backend crashed unexpectedly',
            error: `Process exited with code ${code}`
          })

          if (this.shouldAttemptRestart()) {
            this.scheduleRestart()
          }
        }
      })

      // Handle process errors
      this.process.on('error', (error) => {
        console.error('Backend process error:', error)
        reject(error)
      })

      // Give the process a moment to start
      setTimeout(() => {
        if (this.process && !this.process.killed) {
          resolve()
        } else {
          reject(new Error('Process failed to start'))
        }
      }, 2000)

      // Set startup timeout
      this.startupTimeout = setTimeout(() => {
        if (this.status.state === 'starting') {
          reject(new Error('Backend startup timeout'))
        }
      }, this.STARTUP_TIMEOUT)
    })
  }

  /**
   * Wait for backend to become healthy
   */
  private async waitForHealthy(): Promise<void> {
    const maxAttempts = 60 // 60 attempts * 3 seconds = 3 minutes
    let attempts = 0

    while (attempts < maxAttempts) {
      try {
        const health = await this.healthChecker.check()

        if (health.status === 'healthy') {
          this.updateStatus({ health })
          return
        }

        this.updateStatus({
          message: `Waiting for services to be ready... (${health.readyCount}/${health.totalCount})`,
          health
        })
      } catch (error) {
        // Health check failed, continue waiting
        console.log('Health check attempt', attempts + 1, 'failed')
      }

      await new Promise(resolve => setTimeout(resolve, 3000))
      attempts++
    }

    throw new Error('Backend health check timeout')
  }

  /**
   * Start health monitoring
   */
  private startHealthMonitoring(): void {
    const checkInterval = setInterval(async () => {
      if (!this.process || this.status.state !== 'running') {
        clearInterval(checkInterval)
        return
      }

      try {
        const health = await this.healthChecker.check()

        if (health.status === 'healthy') {
          this.updateStatus({
            state: 'running',
            health
          })
        } else if (health.status === 'degraded') {
          this.updateStatus({
            state: 'degraded',
            message: 'Some services are degraded',
            health
          })
        } else {
          this.updateStatus({
            state: 'error',
            message: 'Backend unhealthy',
            health
          })
        }
      } catch (error) {
        console.error('Health check failed:', error)
        this.updateStatus({
          state: 'error',
          message: 'Health check failed',
          error: error instanceof Error ? error.message : 'Unknown error'
        })
      }
    }, 10000) // Check every 10 seconds
  }

  /**
   * Stop the backend
   */
  async stop(): Promise<void> {
    if (this.startupTimeout) {
      clearTimeout(this.startupTimeout)
    }

    if (this.restartTimeout) {
      clearTimeout(this.restartTimeout)
    }

    if (!this.process) {
      this.updateStatus({
        state: 'stopped',
        message: 'Backend not running'
      })
      return
    }

    this.updateStatus({
      state: 'stopping',
      message: 'Stopping backend...'
    })

    return new Promise((resolve) => {
      if (!this.process) {
        resolve()
        return
      }

      // Set up exit handler
      this.process.once('exit', () => {
        this.process = null
        this.updateStatus({
          state: 'stopped',
          message: 'Backend stopped'
        })
        resolve()
      })

      // Try graceful shutdown first (SIGTERM)
      this.process.kill('SIGTERM')

      // Force kill after timeout
      setTimeout(() => {
        if (this.process) {
          console.log('Force killing backend process')
          this.process.kill('SIGKILL')
        }
      }, 10000)
    })
  }

  /**
   * Restart the backend
   */
  async restart(): Promise<void> {
    await this.stop()
    await new Promise(resolve => setTimeout(resolve, 2000)) // Wait before restart
    await this.start()
  }

  /**
   * Check if should attempt restart
   */
  private shouldAttemptRestart(): boolean {
    return this.restartAttempts < (this.config.maxRestartAttempts || this.MAX_RESTART_ATTEMPTS)
  }

  /**
   * Schedule a restart with backoff
   */
  private scheduleRestart(): void {
    this.restartAttempts++
    const delay = Math.min(1000 * Math.pow(2, this.restartAttempts), 30000) // Exponential backoff, max 30s

    this.updateStatus({
      state: 'error',
      message: `Restarting in ${delay / 1000}s (attempt ${this.restartAttempts}/${this.config.maxRestartAttempts})...`
    })

    this.restartTimeout = setTimeout(async () => {
      console.log(`Auto-restart attempt ${this.restartAttempts}`)
      try {
        await this.start()
      } catch (error) {
        console.error('Auto-restart failed:', error)
      }
    }, delay)
  }

  /**
   * Get backend logs (if available)
   */
  getLogs(): string[] {
    // TODO: Implement log collection
    return []
  }

  /**
   * Cleanup
   */
  async cleanup(): Promise<void> {
    await this.stop()
  }
}
