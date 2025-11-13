/**
 * Window Manager - Manages Electron windows (splash, main, etc.)
 */

import { BrowserWindow, app, screen } from 'electron'
import * as path from 'path'

export class WindowManager {
  private splashWindow: BrowserWindow | null = null
  private mainWindow: BrowserWindow | null = null

  /**
   * Create splash screen
   */
  createSplashWindow(): BrowserWindow {
    this.splashWindow = new BrowserWindow({
      width: 500,
      height: 400,
      frame: false,
      transparent: true,
      resizable: false,
      webPreferences: {
        preload: path.join(__dirname, '../preload/index.js'),
        nodeIntegration: false,
        contextIsolation: true
      }
    })

    // Load splash screen
    if (app.isPackaged) {
      this.splashWindow.loadFile(path.join(__dirname, '../renderer/splash.html'))
    } else {
      // In development, use a simple splash or the dev server
      this.splashWindow.loadURL('http://localhost:5173/splash.html')
    }

    this.splashWindow.center()

    return this.splashWindow
  }

  /**
   * Update splash screen with status
   */
  updateSplash(status: { message: string; progress?: number; error?: string }): void {
    if (this.splashWindow && !this.splashWindow.isDestroyed()) {
      this.splashWindow.webContents.send('splash-status', status)
    }
  }

  /**
   * Create main application window
   */
  createMainWindow(): BrowserWindow {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize

    this.mainWindow = new BrowserWindow({
      width: Math.min(1400, width),
      height: Math.min(900, height),
      show: false, // Don't show until ready
      webPreferences: {
        preload: path.join(__dirname, '../preload/index.js'),
        nodeIntegration: false,
        contextIsolation: true,
        webSecurity: true
      },
      titleBarStyle: 'default',
      title: 'LlamaFarm'
    })

    // Load the designer UI
    if (app.isPackaged) {
      // In production, serve the bundled designer
      const designerPath = path.join(process.resourcesPath, 'designer', 'index.html')
      this.mainWindow.loadFile(designerPath)
    } else {
      // In development, load the renderer HTML (which has the designer iframe)
      // The vite dev server is already running at localhost:5173
      this.mainWindow.loadURL('http://localhost:5173')
    }

    // Open DevTools in development
    if (!app.isPackaged) {
      this.mainWindow.webContents.openDevTools()
    }

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow?.show()
      this.closeSplash()
    })

    // Handle window close
    this.mainWindow.on('closed', () => {
      this.mainWindow = null
    })

    return this.mainWindow
  }

  /**
   * Close splash screen
   */
  closeSplash(): void {
    if (this.splashWindow && !this.splashWindow.isDestroyed()) {
      this.splashWindow.close()
      this.splashWindow = null
    }
  }

  /**
   * Show error dialog on splash
   */
  showSplashError(message: string, details?: string): void {
    this.updateSplash({
      message,
      error: details
    })
  }

  /**
   * Get main window
   */
  getMainWindow(): BrowserWindow | null {
    return this.mainWindow
  }

  /**
   * Get splash window
   */
  getSplashWindow(): BrowserWindow | null {
    return this.splashWindow
  }

  /**
   * Cleanup all windows
   */
  cleanup(): void {
    if (this.splashWindow && !this.splashWindow.isDestroyed()) {
      this.splashWindow.close()
    }
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.close()
    }
  }
}
