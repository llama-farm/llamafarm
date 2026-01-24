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
import { logger } from './logger'
import * as path from 'path'
import * as fs from 'fs'
import { promises as fsPromises } from 'fs'
import { promisify } from 'util'
import { exec } from 'child_process'
import axios from 'axios'
import * as os from 'os'

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
  private startupErrorPending = false

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

    // Splash minimize
    ipcMain.on('splash-minimize', () => {
      const splash = this.windowManager.getSplashWindow()
      if (splash && !splash.isDestroyed()) {
        splash.minimize()
      }
    })

    // Services are managed automatically - no manual control needed
  }

  /**
   * App ready handler - main initialization
   */
  private async onReady(): Promise<void> {
    // Initialize logger first - this also creates the logs directory
    await logger.initialize()

    console.log('LlamaFarm starting...')

    // Create application menu
    this.menuManager.createMenu()

    // Create splash screen
    const splash = this.windowManager.createSplashWindow()

    try {
      // Step 0: Ensure .llamafarm directory exists for logging
      await this.ensureLlamaFarmDirectory()

      // Step 2: Ensure CLI is installed and upgraded
      await this.ensureCLI()

      // Step 3: Services are already started by ensureCLI()
      // No need to run lf start - services start already started server + RAG
      // The server serves all projects from ~/.llamafarm/projects/

      // Step 4: Wait for server to be ready
      await this.waitForServer()

      // Step 5: Check and download required models
      await this.ensureModels()

      // Step 6: Create main window with Designer UI
      this.windowManager.updateSplash({
        message: 'Opening Designer...',
        progress: 98
      })

      // Give a moment for the message to show
      await new Promise(resolve => setTimeout(resolve, 500))

      this.windowManager.createMainWindow()

      // Step 7: Check for app updates (in background)
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
   * Ensure .llamafarm directory exists for logging and data storage
   */
  private async ensureLlamaFarmDirectory(): Promise<void> {
    try {
      const homeDir = os.homedir()
      const llamafarmDir = path.join(homeDir, '.llamafarm')
      const logsDir = path.join(llamafarmDir, 'logs')
      const projectsDir = path.join(llamafarmDir, 'projects')

      // Create directories if they don't exist
      await fsPromises.mkdir(logsDir, { recursive: true })
      await fsPromises.mkdir(projectsDir, { recursive: true })

      console.log('Ensured .llamafarm directory structure exists')
    } catch (error) {
      console.warn('Failed to create .llamafarm directory:', error)
      // Continue anyway - services might create it
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
    const startupMessage = 'Preparing services...'
    this.windowManager.updateSplash({
      message: startupMessage,
      progress: 70
    })

    // Don't catch errors here - let them propagate so we can show proper error messages
    await this.startServices()

    this.windowManager.updateSplash({
      message: '✓ Services started',
      progress: 75
    })
  }

  /**
   * Check and start services if needed.
   * The status check triggers environment setup (source download, dependency sync).
   * We use generous timeouts since first-time setup can take 10+ minutes.
   */
  private async startServices(): Promise<void> {
    // Check services status - this also triggers environment setup if needed
    // (the CLI's ServiceManager calls EnsureNativeEnvironment on init)
    console.log('Checking services status (this triggers environment setup if needed)...')
    this.windowManager.updateSplash({
      message: 'Preparing environment...',
      progress: 55
    })

    let statusOutput = ''
    let servicesRunning = false

    try {
      // Use generous timeout - first-time setup downloads source and syncs dependencies
      const result = await execAsync(
        `"${this.cliInstaller.getCLIPath()}" services status`,
        { timeout: 600000 } // 10 minutes - first-time setup can take a while
      )
      statusOutput = result.stdout
      console.log('Services status:', statusOutput)

      // Check if server and RAG are already running
      const serverRunning = statusOutput.includes('Service: server') &&
        statusOutput.includes('State: ✓ running')
      const ragRunning = statusOutput.includes('Service: rag') &&
        statusOutput.includes('State: ✓ running')

      servicesRunning = serverRunning && ragRunning
    } catch (error) {
      // Status check may fail if environment setup fails - we'll try to start anyway
      console.warn('Services status check failed:', error)
    }

    if (servicesRunning) {
      console.log('Services already running, skipping start')
      return
    }

    // Services not running, start them with retry logic
    console.log('Starting services...')
    this.windowManager.updateSplash({
      message: 'Starting LlamaFarm services...',
      progress: 60
    })

    const timeout = 600000 // 10 minutes - generous timeout for first-time setup
    const maxRetries = 2

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 1) {
          console.log(`Retry attempt ${attempt}/${maxRetries}...`)
          this.windowManager.updateSplash({
            message: `Starting services (attempt ${attempt}/${maxRetries})...`,
            progress: 60
          })
          await new Promise(resolve => setTimeout(resolve, 3000))
        }

        const { stdout, stderr } = await execAsync(
          `"${this.cliInstaller.getCLIPath()}" services start`,
          { timeout }
        )

        console.log('Services output:', stdout)
        if (stderr) {
          console.log('Services stderr:', stderr)
        }

        // Verify services started
        await new Promise(resolve => setTimeout(resolve, 2000))

        const verifyResult = await execAsync(
          `"${this.cliInstaller.getCLIPath()}" services status`,
          { timeout: 60000 }
        )

        const serverStarted = verifyResult.stdout.includes('Service: server') &&
          verifyResult.stdout.includes('State: ✓ running')
        const ragStarted = verifyResult.stdout.includes('Service: rag') &&
          verifyResult.stdout.includes('State: ✓ running')

        if (!serverStarted || !ragStarted) {
          console.warn('Services may not have started properly. Status:', verifyResult.stdout)
        }

        // Success - exit retry loop
        return
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error)
        console.error(`Service start attempt ${attempt} failed:`, errorMsg)

        if (attempt >= maxRetries) {
          if (errorMsg.includes('timeout') || errorMsg.includes('TIMEOUT')) {
            throw new Error(
              `Services failed to start within ${timeout / 1000} seconds. ` +
              `Please check logs in ${path.join(os.homedir(), '.llamafarm', 'logs')} and try again.`
            )
          }
          throw new Error(`Failed to start services: ${errorMsg}`)
        }
      }
    }
  }

  /**
   * Wait for server to be ready with exponential backoff
   */
  private async waitForServer(): Promise<void> {
    this.windowManager.updateSplash({
      message: 'Waiting for server...',
      progress: 80
    })

    const maxAttempts = 120 // Up to ~2 minutes with backoff
    let attempts = 0
    let delay = 500 // Start with 500ms
    const maxDelay = 2000 // Cap at 2 seconds

    while (attempts < maxAttempts) {
      try {
        // Check if server is responding - use 127.0.0.1 instead of localhost to avoid IPv6
        const response = await axios.get('http://127.0.0.1:8000/health', {
          timeout: 3000
        })

        if (response.status === 200) {
          console.log('Server is ready!')
          this.windowManager.updateSplash({
            message: '✓ Server ready',
            progress: 85
          })
          return
        }
      } catch (error) {
        // Server not ready yet, continue waiting
        const errorMsg = error instanceof Error ? error.message : String(error)
        if (attempts % 10 === 0 || attempts < 5) {
          console.log(`Server check attempt ${attempts + 1}/${maxAttempts} - ${errorMsg}`)
        }
      }

      await new Promise(resolve => setTimeout(resolve, delay))
      delay = Math.min(delay * 1.1, maxDelay)
      attempts++
    }

    throw new Error(
      `Server failed to respond after ${maxAttempts} attempts. ` +
      `Please check logs in ${path.join(os.homedir(), '.llamafarm', 'logs', 'server.log')} and run "lf services status" for details.`
    )
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
        // Only show simple status messages - no model list unless there's an error
        this.windowManager.updateSplash({
          message: progress.message,
          progress: 88 + (progress.overall_progress * 0.07) // 88-95% range
        })
      })

      if (result.success) {
        console.log('All required models are ready')
        this.windowManager.updateSplash({
          message: '✓ Models ready',
          progress: 95
        })
      } else {
        // Some models failed - show them in the error display
        const failedModels = result.models.filter(m => m.status === 'error')
        console.warn('Some models failed to download:', failedModels.map(m => m.id))
        this.windowManager.updateSplash({
          message: 'Some models unavailable',
          progress: 95,
          models: result.models.map(m => ({
            id: m.id,
            display_name: m.display_name,
            status: m.status,
            progress: m.progress
          }))
        })
      }

      // Brief pause before opening Designer
      await new Promise(resolve => setTimeout(resolve, 500))
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

    this.startupErrorPending = true
    this.windowManager.showSplashError('Failed to start LlamaFarm', errorMessage)

    // Remove always-on-top before showing dialog to ensure visibility
    setTimeout(() => {
      const splash = this.windowManager.getSplashWindow()
      if (splash && !splash.isDestroyed()) {
        splash.setAlwaysOnTop(false)
      }

      // Use modal dialog with better UX (keep splash open until after dialog)
      dialog.showMessageBoxSync({
        type: 'error',
        title: 'LlamaFarm Startup Failed',
        message: 'Failed to start LlamaFarm',
        detail: `${errorMessage}\n\nPlease check the logs and try again.`,
        buttons: ['OK']
      })

      this.windowManager.closeSplash()
      this.isQuitting = true
      app.quit()
    }, 3000)
  }

  /**
   * Window all closed handler
   */
  private onWindowsClosed(): void {
    // On macOS, keep app running when windows are closed
    if (this.startupErrorPending) {
      return
    }

    if (process.platform !== 'darwin') {
      app.quit()
    }
  }

  /**
   * Activate handler (macOS dock click, Windows/Linux taskbar click)
   */
  private onActivate(): void {
    if (this.windowManager.wasMinimizedOnComplete) {
      // Loading completed while minimized - show main window
      this.windowManager.showMainWindowFromActivation()
    } else if (BrowserWindow.getAllWindows().length === 0) {
      // On macOS, recreate window when dock icon is clicked
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

      // Close logger
      await logger.close()

      app.exit(0)
    } catch (error) {
      console.error('Shutdown error:', error)
      app.exit(1)
    }
  }
}

// Create and start the app
new LlamaFarmApp()
