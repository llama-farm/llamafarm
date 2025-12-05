/**
 * Logger utility for the Electron app
 * Writes logs to ~/.llamafarm/logs/app.log
 */

import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

class Logger {
  private logFile: string
  private writeStream: fs.WriteStream | null = null
  private initialized = false
  private originalConsole: {
    log: typeof console.log
    warn: typeof console.warn
    error: typeof console.error
    debug: typeof console.debug
  }

  constructor() {
    // Store original console methods
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      debug: console.debug.bind(console)
    }

    // Set default log file path
    const homeDir = os.homedir()
    this.logFile = path.join(homeDir, '.llamafarm', 'logs', 'app.log')
  }

  /**
   * Initialize the logger and redirect console output
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return
    }

    try {
      // Ensure log directory exists
      const logDir = path.dirname(this.logFile)
      await fs.promises.mkdir(logDir, { recursive: true })

      // Rotate log if it's too large (> 10MB)
      try {
        const stats = await fs.promises.stat(this.logFile)
        if (stats.size > 10 * 1024 * 1024) {
          const rotatedPath = this.logFile.replace('.log', `.${Date.now()}.log`)
          await fs.promises.rename(this.logFile, rotatedPath)
        }
      } catch {
        // File doesn't exist yet, that's fine
      }

      // Create write stream with append mode
      this.writeStream = fs.createWriteStream(this.logFile, { flags: 'a' })

      // Write startup marker
      const startupMessage = `\n${'='.repeat(60)}\n[${this.timestamp()}] LlamaFarm Desktop starting...\n${'='.repeat(60)}\n`
      this.writeStream.write(startupMessage)

      // Redirect console methods
      this.redirectConsole()

      this.initialized = true
      this.info('Logger initialized', { logFile: this.logFile })
    } catch (error) {
      // If we can't initialize, just log to console
      this.originalConsole.error('Failed to initialize file logger:', error)
    }
  }

  /**
   * Redirect console.log/warn/error to also write to file
   */
  private redirectConsole(): void {
    console.log = (...args: unknown[]) => {
      this.log('info', ...args)
    }

    console.warn = (...args: unknown[]) => {
      this.log('warn', ...args)
    }

    console.error = (...args: unknown[]) => {
      this.log('error', ...args)
    }

    console.debug = (...args: unknown[]) => {
      this.log('debug', ...args)
    }
  }

  /**
   * Format a timestamp for log entries
   */
  private timestamp(): string {
    return new Date().toISOString()
  }

  /**
   * Format arguments for logging
   */
  private formatArgs(args: unknown[]): string {
    return args
      .map((arg) => {
        if (arg instanceof Error) {
          return `${arg.message}\n${arg.stack}`
        }
        if (typeof arg === 'object') {
          try {
            return JSON.stringify(arg, null, 2)
          } catch {
            return String(arg)
          }
        }
        return String(arg)
      })
      .join(' ')
  }

  /**
   * Write a log entry
   */
  private log(level: LogLevel, ...args: unknown[]): void {
    const message = this.formatArgs(args)
    const logLine = `[${this.timestamp()}] [${level.toUpperCase()}] ${message}\n`

    // Write to file
    if (this.writeStream) {
      this.writeStream.write(logLine)
    }

    // Also write to original console
    switch (level) {
      case 'error':
        this.originalConsole.error(...args)
        break
      case 'warn':
        this.originalConsole.warn(...args)
        break
      case 'debug':
        this.originalConsole.debug(...args)
        break
      default:
        this.originalConsole.log(...args)
    }
  }

  /**
   * Log an info message
   */
  info(message: string, data?: Record<string, unknown>): void {
    if (data) {
      this.log('info', message, data)
    } else {
      this.log('info', message)
    }
  }

  /**
   * Log a warning message
   */
  warn(message: string, data?: Record<string, unknown>): void {
    if (data) {
      this.log('warn', message, data)
    } else {
      this.log('warn', message)
    }
  }

  /**
   * Log an error message
   */
  error(message: string, error?: Error | unknown): void {
    if (error) {
      this.log('error', message, error)
    } else {
      this.log('error', message)
    }
  }

  /**
   * Log a debug message
   */
  debug(message: string, data?: Record<string, unknown>): void {
    if (data) {
      this.log('debug', message, data)
    } else {
      this.log('debug', message)
    }
  }

  /**
   * Close the log file
   */
  async close(): Promise<void> {
    if (this.writeStream) {
      this.info('Logger shutting down')
      await new Promise<void>((resolve) => {
        this.writeStream!.end(() => resolve())
      })
      this.writeStream = null
    }
  }

  /**
   * Get the path to the log file
   */
  getLogPath(): string {
    return this.logFile
  }
}

// Export singleton instance
export const logger = new Logger()
