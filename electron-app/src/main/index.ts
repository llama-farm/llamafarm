/**
 * Main Process Entry Point
 * Handles app lifecycle, backend orchestration, and window management
 */

import { app, BrowserWindow, ipcMain, dialog, nativeImage, nativeTheme } from 'electron'
import { autoUpdater } from 'electron-updater'
import { CLIInstaller, InstallProgress } from './backend/cli-installer'
import { ModelDownloader, ModelDownloadProgress } from './backend/model-downloader'
import { WindowManager } from './window-manager'
import { MenuManager } from './menu-manager'
import * as path from 'path'
import * as fs from 'fs'
import { promisify } from 'util'
import { exec } from 'child_process'
import axios from 'axios'

const execAsync = promisify(exec)

// Configure auto-updater
autoUpdater.autoDownload = true
autoUpdater.autoInstallOnAppQuit = true

class LlamaFarmApp {
  private cliInstaller: CLIInstaller
  private modelDownloader: ModelDownloader
  private windowManager: WindowManager
  private menuManager: MenuManager
  private isQuitting = false

  constructor() {
    // Set app name early
    app.setName('LlamaFarm')

    // Force dark mode for title bar to match Designer UI
    nativeTheme.themeSource = 'dark'

    // Set dock icon on macOS (in development)
    if (process.platform === 'darwin' && !app.isPackaged) {
      try {
        const iconPath = path.join(__dirname, '../../../designer/public/llama-farm-favicon.svg')
        if (fs.existsSync(iconPath)) {
          const icon = nativeImage.createFromPath(iconPath)
          app.dock?.setIcon(icon)
        }
      } catch (error) {
        console.log('Could not set dock icon:', error)
      }
    }

    this.cliInstaller = new CLIInstaller()
    this.modelDownloader = new ModelDownloader()
    this.windowManager = new WindowManager()
    this.menuManager = new MenuManager()

    this.setupEventHandlers()
    this.setupIPCHandlers()
    this.setupAutoUpdater()
  }

  /**
   * Setup auto-updater
   */
  private setupAutoUpdater(): void {
    // Only check for updates in production
    if (!app.isPackaged) {
      console.log('Skipping auto-update check in development mode')
      return
    }

    autoUpdater.on('checking-for-update', () => {
      console.log('Checking for updates...')
    })

    autoUpdater.on('update-available', (info) => {
      console.log('Update available:', info.version)
    })

    autoUpdater.on('update-not-available', () => {
      console.log('No updates available')
    })

    autoUpdater.on('error', (err) => {
      console.error('Auto-updater error:', err)
    })

    autoUpdater.on('download-progress', (progressObj) => {
      console.log(`Download progress: ${progressObj.percent.toFixed(1)}%`)
    })

    autoUpdater.on('update-downloaded', (info) => {
      console.log('Update downloaded:', info.version)

      // Show dialog to user
      const dialogOpts = {
        type: 'info' as const,
        buttons: ['Restart Now', 'Later'],
        title: 'Update Available',
        message: `LlamaFarm Designer ${info.version}`,
        detail: 'A new version has been downloaded. Restart the application to apply the updates.'
      }

      dialog.showMessageBox(dialogOpts).then((returnValue) => {
        if (returnValue.response === 0) {
          // Restart now
          autoUpdater.quitAndInstall()
        }
      })
    })
  }

  /**
   * Setup application event handlers
   */
  private setupEventHandlers(): void {
    app.on('ready', () => this.onReady())
    app.on('window-all-closed', () => this.onWindowsClosed())
    app.on('activate', () => this.onActivate())
    app.on('before-quit', () => this.onBeforeQuit())
    app.on('will-quit', (event) => this.onWillQuit(event))
  }

  /**
   * Setup IPC handlers for renderer communication
   */
  private setupIPCHandlers(): void {
    // Get CLI info
    ipcMain.handle('cli:info', async () => {
      const isInstalled = await this.cliInstaller.isInstalled()
      return {
        isInstalled,
        path: isInstalled ? this.cliInstaller.getCLIPath() : null
      }
    })

    // Services are managed automatically - no manual control needed
  }

  /**
   * App ready handler - main initialization
   */
  private async onReady(): Promise<void> {
    console.log('LlamaFarm starting...')

    // Create application menu
    this.menuManager.createMenu()

    // Create splash screen
    const splash = this.windowManager.createSplashWindow()

    try {
      // Step 1: Ensure CLI is installed and upgraded
      await this.ensureCLI()

      // Step 2: Services are already started by ensureCLI()
      // No need to run lf start - services start already started server + RAG
      // The server serves all projects from ~/.llamafarm/projects/

      // Step 3: Wait for server to be ready
      await this.waitForServer()

      // Step 4: Check and download required models
      await this.ensureModels()

      // Step 5: Create main window with Designer UI
      this.windowManager.updateSplash({
        message: 'Opening Designer...',
        progress: 98
      })

      // Give a moment for the message to show
      await new Promise(resolve => setTimeout(resolve, 500))

      this.windowManager.createMainWindow()

      // Step 6: Check for app updates (in background)
      if (app.isPackaged) {
        setTimeout(() => {
          autoUpdater.checkForUpdatesAndNotify().catch(err => {
            console.log('Failed to check for updates:', err)
          })
        }, 5000) // Wait 5 seconds after app starts
      }
    } catch (error) {
      console.error('Startup failed:', error)
      this.handleStartupError(error)
    }
  }

  /**
   * Ensure CLI is installed and upgraded
   */
  private async ensureCLI(): Promise<void> {
    this.windowManager.updateSplash({
      message: 'Checking for LlamaFarm CLI...',
      progress: 10
    })

    const isInstalled = await this.cliInstaller.isInstalled()

    if (!isInstalled) {
      console.log('CLI not found, installing...')

      await this.cliInstaller.install((progress: InstallProgress) => {
        console.log('Install progress:', progress.step, progress.message)

        const progressMap = {
          checking: 10,
          downloading: 30,
          installing: 60,
          verifying: 80,
          complete: 90
        }

        this.windowManager.updateSplash({
          message: progress.message,
          progress: progress.progress || progressMap[progress.step]
        })
      })
    } else {
      console.log('CLI found at:', this.cliInstaller.getCLIPath())
    }

    // Always run upgrade to ensure latest version
    this.windowManager.updateSplash({
      message: 'Checking for CLI updates...',
      progress: 50
    })

    try {
      await this.cliInstaller.upgrade((progress: InstallProgress) => {
        this.windowManager.updateSplash({
          message: progress.message,
          progress: 50 + (progress.progress || 0) * 0.2
        })
      })
    } catch (error) {
      // Upgrade failure is not critical, continue anyway
      console.warn('CLI upgrade failed (continuing anyway):', error)
    }

    // Start services to ensure RAG server and dependencies are downloaded
    this.windowManager.updateSplash({
      message: 'Preparing services...',
      progress: 70
    })

    try {
      await this.startServices()
    } catch (error) {
      // Service start failure is not critical, continue anyway
      console.warn('Services start failed (continuing anyway):', error)
    }

    this.windowManager.updateSplash({
      message: 'LlamaFarm CLI ready',
      progress: 90
    })
  }

  /**
   * Check and start services if needed
   */
  private async startServices(): Promise<void> {
    try {
      // First check if services are already running
      console.log('Checking services status...')

      const { stdout: statusOutput } = await execAsync(
        `"${this.cliInstaller.getCLIPath()}" services status`,
        { timeout: 30000 }
      )

      console.log('Services status:', statusOutput)

      // Check if server and RAG are running
      const serverRunning = statusOutput.includes('Service: server') &&
                           statusOutput.includes('State: ✓ running')
      const ragRunning = statusOutput.includes('Service: rag') &&
                        statusOutput.includes('State: ✓ running')

      if (serverRunning && ragRunning) {
        console.log('Services already running, skipping start')
        return
      }

      // Services not running, start them
      console.log('Starting services...')
      this.windowManager.updateSplash({
        message: 'Starting LlamaFarm services...',
        progress: 60
      })

      const { stdout, stderr } = await execAsync(
        `"${this.cliInstaller.getCLIPath()}" services start`,
        { timeout: 180000 } // 3 minutes timeout for downloads
      )

      console.log('Services output:', stdout)
      if (stderr) console.error('Services stderr:', stderr)
    } catch (error) {
      console.warn('Services start had issues:', error)
      // Continue anyway - not critical
    }
  }

  /**
   * Wait for server to be ready
   */
  private async waitForServer(): Promise<void> {
    this.windowManager.updateSplash({
      message: 'Waiting for server...',
      progress: 80
    })

    const maxAttempts = 30 // 30 attempts * 1 second = 30 seconds
    let attempts = 0

    while (attempts < maxAttempts) {
      try {
        // Check if server is responding - use 127.0.0.1 instead of localhost to avoid IPv6
        const response = await axios.get('http://127.0.0.1:8000/health', {
          timeout: 3000
        })

        console.log(`Health check response: ${response.status}`)
        if (response.status === 200) {
          console.log('Server is ready!')
          this.windowManager.updateSplash({
            message: 'Server ready!',
            progress: 90
          })
          return
        }
      } catch (error) {
        // Server not ready yet, continue waiting
        const errorMsg = error instanceof Error ? error.message : String(error)
        console.log(`Server check attempt ${attempts + 1}/${maxAttempts} - Error: ${errorMsg}`)
      }

      await new Promise(resolve => setTimeout(resolve, 1000))
      attempts++
    }

    throw new Error('Server failed to start - timeout waiting for http://127.0.0.1:8000/health')
  }

  /**
   * Ensure required models are downloaded
   */
  private async ensureModels(): Promise<void> {
    console.log('Checking required models...')

    this.windowManager.updateSplash({
      message: 'Checking required models...',
      progress: 85
    })

    try {
      const result = await this.modelDownloader.ensureModels((progress) => {
        // Map model statuses to splash format
        const models = progress.models.map(m => ({
          id: m.id,
          display_name: m.display_name,
          status: m.status,
          progress: m.progress
        }))

        this.windowManager.updateSplash({
          message: progress.message,
          progress: 85 + (progress.overall_progress * 0.1), // 85-95% range
          models
        })
      })

      if (result.success) {
        console.log('All required models are ready')
        this.windowManager.updateSplash({
          message: 'All models ready!',
          progress: 95,
          models: result.models.map(m => ({
            id: m.id,
            display_name: m.display_name,
            status: m.status,
            progress: m.progress
          }))
        })
      } else {
        // Some models failed but continue anyway
        const failedModels = result.models.filter(m => m.status === 'error')
        console.warn('Some models failed to download:', failedModels.map(m => m.id))
        this.windowManager.updateSplash({
          message: 'Some models unavailable (continuing...)',
          progress: 95,
          models: result.models.map(m => ({
            id: m.id,
            display_name: m.display_name,
            status: m.status,
            progress: m.progress
          }))
        })
      }

      // Small delay to show the model status
      await new Promise(resolve => setTimeout(resolve, 1000))
    } catch (error) {
      console.warn('Model check failed (continuing anyway):', error)
      // Model download failure is not critical - continue anyway
    }
  }

  /**
   * Handle startup errors
   */
  private handleStartupError(error: unknown): void {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    console.error('Startup error:', errorMessage)

    this.windowManager.showSplashError(
      'Failed to start LlamaFarm',
      errorMessage
    )

    // Show error dialog
    setTimeout(() => {
      dialog.showErrorBox(
        'LlamaFarm Startup Failed',
        `Failed to start LlamaFarm:\n\n${errorMessage}\n\nPlease check the logs and try again.`
      )
      app.quit()
    }, 3000)
  }

  /**
   * Window all closed handler
   */
  private onWindowsClosed(): void {
    // On macOS, keep app running when windows are closed
    if (process.platform !== 'darwin') {
      app.quit()
    }
  }

  /**
   * Activate handler (macOS)
   */
  private onActivate(): void {
    // On macOS, recreate window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      this.windowManager.createMainWindow()
    }
  }

  /**
   * Before quit handler
   */
  private onBeforeQuit(): void {
    this.isQuitting = true
  }

  /**
   * Will quit handler - cleanup
   */
  private async onWillQuit(event: Electron.Event): Promise<void> {
    if (!this.isQuitting) {
      return
    }

    event.preventDefault()

    console.log('Shutting down LlamaFarm...')

    try {
      // Stop services
      console.log('Stopping services...')
      try {
        await execAsync(
          `"${this.cliInstaller.getCLIPath()}" services stop`,
          { timeout: 30000 }
        )
      } catch (error) {
        console.warn('Services stop had issues:', error)
        // Continue anyway
      }

      // Cleanup windows
      this.windowManager.cleanup()

      console.log('Shutdown complete')
      app.exit(0)
    } catch (error) {
      console.error('Shutdown error:', error)
      app.exit(1)
    }
  }
}

// Create and start the app
new LlamaFarmApp()
