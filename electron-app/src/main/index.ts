/**
 * Main Process Entry Point
 * Handles app lifecycle, backend orchestration, and window management
 */

import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { CLIInstaller, InstallProgress } from './backend/cli-installer'
import { BackendManager, BackendStatus } from './backend/backend-manager'
import { WindowManager } from './window-manager'

class LlamaFarmApp {
  private cliInstaller: CLIInstaller
  private backendManager: BackendManager | null = null
  private windowManager: WindowManager
  private isQuitting = false

  constructor() {
    this.cliInstaller = new CLIInstaller()
    this.windowManager = new WindowManager()

    this.setupEventHandlers()
    this.setupIPCHandlers()
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
    // Get backend status
    ipcMain.handle('backend:status', () => {
      return this.backendManager?.getStatus() || {
        state: 'stopped',
        message: 'Backend not initialized'
      }
    })

    // Restart backend
    ipcMain.handle('backend:restart', async () => {
      if (this.backendManager) {
        await this.backendManager.restart()
        return { success: true }
      }
      return { success: false, error: 'Backend not initialized' }
    })

    // Stop backend
    ipcMain.handle('backend:stop', async () => {
      if (this.backendManager) {
        await this.backendManager.stop()
        return { success: true }
      }
      return { success: false, error: 'Backend not initialized' }
    })

    // Get CLI info
    ipcMain.handle('cli:info', async () => {
      const isInstalled = await this.cliInstaller.isInstalled()
      return {
        isInstalled,
        path: isInstalled ? this.cliInstaller.getCLIPath() : null
      }
    })
  }

  /**
   * App ready handler - main initialization
   */
  private async onReady(): Promise<void> {
    console.log('LlamaFarm Desktop starting...')

    // Create splash screen
    const splash = this.windowManager.createSplashWindow()

    try {
      // Step 1: Ensure CLI is installed and upgraded
      await this.ensureCLI()

      // Step 2: Open Designer (no backend startup)
      // Users will choose/create projects in the Designer
      this.windowManager.updateSplash({
        message: 'Opening Designer...',
        progress: 95
      })

      // Give a moment for the message to show
      await new Promise(resolve => setTimeout(resolve, 500))

      // Step 3: Create main window
      this.windowManager.createMainWindow()
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
   * Start services (ensures RAG server and dependencies are downloaded)
   */
  private async startServices(): Promise<void> {
    try {
      console.log('Running lf services start...')
      const { execAsync } = require('util').promisify(require('child_process').exec)

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

  // Backend startup is now handled by the Designer when user selects a project
  // This method is no longer called during app initialization

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
      // Stop backend gracefully
      if (this.backendManager) {
        console.log('Stopping backend...')
        await this.backendManager.cleanup()
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
