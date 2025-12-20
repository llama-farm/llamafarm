/**
 * CLI Installer - Handles automatic installation of the LlamaFarm CLI
 * Based on the official install.sh script logic
 */

import { app } from 'electron'
import { exec, spawn } from 'child_process'
import { promisify } from 'util'
import * as fs from 'fs'
import * as path from 'path'
import * as https from 'https'
import { createWriteStream, promises as fsPromises } from 'fs'

const execAsync = promisify(exec)

export interface InstallProgress {
  step: 'checking' | 'downloading' | 'installing' | 'verifying' | 'complete' | 'error'
  message: string
  progress?: number
}

export class CLIInstaller {
  private readonly REPO = 'llama-farm/llamafarm'
  private readonly BINARY_NAME = 'lf'
  private cliPath: string

  constructor() {
    // Store CLI in app's userData directory for portable installation
    const userDataPath = app.getPath('userData')
    // On Windows, executables need .exe extension
    const binaryName = process.platform === 'win32' ? `${this.BINARY_NAME}.exe` : this.BINARY_NAME
    this.cliPath = path.join(userDataPath, 'bin', binaryName)
  }

  /**
   * Check if CLI is installed and accessible
   */
  async isInstalled(): Promise<boolean> {
    try {
      // First check our local installation
      if (fs.existsSync(this.cliPath)) {
        await fsPromises.access(this.cliPath, fs.constants.X_OK)
        return true
      }

      // Fallback: check if it's in system PATH
      const { stdout } = await execAsync(process.platform === 'win32' ? 'where lf' : 'which lf')
      if (stdout.trim()) {
        this.cliPath = stdout.trim()
        return true
      }

      return false
    } catch {
      return false
    }
  }

  /**
   * Get installed CLI version
   */
  async getInstalledVersion(): Promise<string | null> {
    try {
      const { stdout } = await execAsync(`"${this.cliPath}" version`)
      // Parse version from output like "lf version 0.0.14"
      const match = stdout.match(/version\s+v?(\d+\.\d+\.\d+)/i)
      if (match && match[1]) {
        return match[1]
      } else {
        console.warn(
          `Could not parse CLI version from output: "${stdout.trim()}". Output format may have changed.`
        )
        return null
      }
    } catch (err) {
      console.error("Error getting installed CLI version:", err)
      return null
    }
  }

  /**
   * Check if CLI needs upgrade
   */
  async needsUpgrade(): Promise<boolean> {
    try {
      const installed = await this.getInstalledVersion()
      if (!installed) return false

      const latest = await this.getLatestVersion()
      if (!latest) return false

      // Remove 'v' prefix if present
      const latestClean = latest.replace(/^v/, '')

      // Simple version comparison
      return latestClean !== installed
    } catch {
      return false
    }
  }

  /**
   * Upgrade CLI using the built-in upgrade command
   */
  async upgrade(onProgress?: (progress: InstallProgress) => void): Promise<void> {
    try {
      onProgress?.({ step: 'checking', message: 'Upgrading LlamaFarm CLI...' })

      console.log('Running lf version upgrade...')
      const { stdout, stderr } = await execAsync(`"${this.cliPath}" version upgrade`, {
        timeout: 120000 // 2 minutes timeout
      })

      console.log('Upgrade output:', stdout)
      if (stderr) console.error('Upgrade stderr:', stderr)

      onProgress?.({ step: 'complete', message: 'CLI upgraded successfully!' })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      onProgress?.({ step: 'error', message: `Upgrade failed: ${errorMessage}` })
      throw error
    }
  }

  /**
   * Get the CLI executable path
   */
  getCLIPath(): string {
    return this.cliPath
  }

  /**
   * Detect platform and architecture
   */
  private detectPlatform(): string {
    const platform = process.platform
    const arch = process.arch

    let os: string
    let architecture: string

    // Map platform
    switch (platform) {
      case 'darwin':
        os = 'darwin'
        break
      case 'linux':
        os = 'linux'
        break
      case 'win32':
        os = 'windows'
        break
      default:
        throw new Error(`Unsupported platform: ${platform}`)
    }

    // Map architecture
    switch (arch) {
      case 'x64':
        architecture = 'amd64'
        break
      case 'arm64':
        architecture = 'arm64'
        break
      case 'arm':
        architecture = 'arm'
        break
      case 'ia32':
        architecture = '386'
        break
      default:
        throw new Error(`Unsupported architecture: ${arch}`)
    }

    return `${os}-${architecture}`
  }

  /**
   * Get latest release version from GitHub
   */
  private async getLatestVersion(): Promise<string> {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.github.com',
        path: `/repos/${this.REPO}/releases/latest`,
        headers: {
          'User-Agent': 'LlamaFarm-Desktop'
        }
      }

      https.get(options, (res) => {
        let data = ''

        res.on('data', (chunk) => {
          data += chunk
        })

        res.on('end', () => {
          try {
            const release = JSON.parse(data)
            resolve(release.tag_name)
          } catch (err) {
            reject(new Error('Failed to parse release info'))
          }
        })
      }).on('error', (err) => {
        reject(err)
      })
    })
  }

  /**
   * Download file from URL
   */
  private async downloadFile(url: string, dest: string, onProgress?: (progress: number) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      const file = createWriteStream(dest)

      https.get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          // Handle redirect
          const redirectUrl = response.headers.location
          if (redirectUrl) {
            this.downloadFile(redirectUrl, dest, onProgress).then(resolve).catch(reject)
            return
          }
        }

        const totalSize = parseInt(response.headers['content-length'] || '0', 10)
        let downloadedSize = 0

        response.on('data', (chunk) => {
          downloadedSize += chunk.length
          if (onProgress && totalSize > 0) {
            onProgress(Math.round((downloadedSize / totalSize) * 100))
          }
        })

        response.pipe(file)

        file.on('finish', () => {
          file.close()
          resolve()
        })
      }).on('error', (err) => {
        fs.unlink(dest, () => { }) // Clean up
        reject(err)
      })
    })
  }

  /**
   * Install the CLI
   */
  async install(onProgress?: (progress: InstallProgress) => void): Promise<void> {
    try {
      onProgress?.({ step: 'checking', message: 'Detecting platform...' })

      const platform = this.detectPlatform()
      console.log('Detected platform:', platform)

      onProgress?.({ step: 'checking', message: 'Getting latest version...' })
      const version = await this.getLatestVersion()
      console.log('Latest version:', version)

      // Construct download URL
      let filename = `${this.BINARY_NAME}-${platform}`
      if (platform.startsWith('windows')) {
        filename += '.exe'
      }
      const downloadUrl = `https://github.com/${this.REPO}/releases/download/${version}/${filename}`

      onProgress?.({ step: 'downloading', message: `Downloading ${filename}...`, progress: 0 })

      // Ensure bin directory exists
      const binDir = path.dirname(this.cliPath)
      await fsPromises.mkdir(binDir, { recursive: true })

      // Download to temporary location first
      const tempPath = path.join(binDir, `${filename}.tmp`)

      await this.downloadFile(downloadUrl, tempPath, (progress) => {
        onProgress?.({ step: 'downloading', message: `Downloading ${filename}...`, progress })
      })

      onProgress?.({ step: 'installing', message: 'Installing CLI...' })

      // Move to final location
      await fsPromises.rename(tempPath, this.cliPath)

      // Make executable (Unix-like systems)
      if (process.platform !== 'win32') {
        await fsPromises.chmod(this.cliPath, 0o755)
      }

      // Wait for file system to finish (prevents "Text file busy" on Linux)
      await new Promise(resolve => setTimeout(resolve, 500))

      onProgress?.({ step: 'verifying', message: 'Verifying installation...' })

      // Verify installation with retry for "Text file busy" errors
      let verifyAttempts = 0
      const maxVerifyAttempts = 3
      while (verifyAttempts < maxVerifyAttempts) {
        try {
          const isInstalled = await this.isInstalled()
          if (!isInstalled) {
            throw new Error('Installation verification failed')
          }

          // Get version to confirm it works
          const { stdout } = await execAsync(`"${this.cliPath}" version`)
          console.log('CLI version:', stdout.trim())
          break // Success
        } catch (err) {
          verifyAttempts++
          const errorMsg = err instanceof Error ? err.message : String(err)
          if (errorMsg.includes('Text file busy') && verifyAttempts < maxVerifyAttempts) {
            console.log(`File busy, waiting before retry (${verifyAttempts}/${maxVerifyAttempts})...`)
            await new Promise(resolve => setTimeout(resolve, 1000))
          } else {
            throw err
          }
        }
      }

      onProgress?.({ step: 'complete', message: 'CLI installed successfully!' })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      onProgress?.({ step: 'error', message: `Installation failed: ${errorMessage}` })
      throw error
    }
  }

  /**
   * Uninstall the CLI (cleanup)
   */
  async uninstall(): Promise<void> {
    if (fs.existsSync(this.cliPath)) {
      await fsPromises.unlink(this.cliPath)
    }
  }
}
