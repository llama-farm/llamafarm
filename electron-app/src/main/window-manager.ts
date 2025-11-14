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
      alwaysOnTop: true,
      webPreferences: {
        preload: path.join(__dirname, '../preload/index.js'),
        nodeIntegration: false,
        contextIsolation: true
      }
    })

    // Load embedded splash HTML
    const splashHTML = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="UTF-8">
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              background: hsl(222.2, 47.4%, 11.2%);
              color: hsl(210, 40%, 98%);
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              height: 100vh;
              overflow: hidden;
            }
            .logo {
              font-size: 64px;
              margin-bottom: 24px;
              filter: drop-shadow(0 0 20px rgba(34, 211, 238, 0.3));
            }
            h1 {
              font-size: 36px;
              font-weight: 600;
              margin-bottom: 48px;
              background: linear-gradient(135deg, rgba(34, 211, 238, 1) 0%, rgba(59, 130, 246, 1) 100%);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              background-clip: text;
            }
            .status {
              font-size: 15px;
              opacity: 0.8;
              margin-bottom: 20px;
              color: hsl(215, 20%, 65%);
            }
            .progress-container {
              width: 320px;
              height: 6px;
              background: hsl(215, 28%, 17%);
              border-radius: 3px;
              overflow: hidden;
              box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            .progress-bar {
              height: 100%;
              background: linear-gradient(90deg, rgba(34, 211, 238, 0.9) 0%, rgba(59, 130, 246, 0.9) 100%);
              border-radius: 3px;
              transition: width 0.3s ease;
              width: 0%;
              box-shadow: 0 0 10px rgba(34, 211, 238, 0.5);
            }
            .spinner {
              width: 40px;
              height: 40px;
              border: 3px solid hsl(215, 28%, 17%);
              border-top-color: rgba(34, 211, 238, 0.9);
              border-radius: 50%;
              animation: spin 1s linear infinite;
              margin-top: 24px;
            }
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
            .error {
              color: #ff6b6b;
              background: hsl(215, 28%, 17%);
              padding: 16px;
              border-radius: 8px;
              margin-top: 24px;
              max-width: 400px;
              font-size: 13px;
              border: 1px solid rgba(255, 107, 107, 0.2);
            }
          </style>
        </head>
        <body>
          <div class="logo">🦙</div>
          <h1>LlamaFarm</h1>
          <div class="status" id="status">Starting...</div>
          <div class="progress-container">
            <div class="progress-bar" id="progress"></div>
          </div>
          <div class="spinner"></div>
          <div class="error" id="error" style="display: none;"></div>

          <script>
            window.llamafarm.splash.onStatus((status) => {
              document.getElementById('status').textContent = status.message;
              if (status.progress !== undefined) {
                document.getElementById('progress').style.width = status.progress + '%';
              }
              if (status.error) {
                document.getElementById('error').textContent = status.error;
                document.getElementById('error').style.display = 'block';
              }
            });
          </script>
        </body>
      </html>
    `

    this.splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(splashHTML)}`)
    this.splashWindow.center()
    this.splashWindow.show() // Show immediately!

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
        webSecurity: false // Disable for localhost CORS
      },
      titleBarStyle: 'hidden',
      trafficLightPosition: { x: 20, y: 18 },
      backgroundColor: '#1a1f2e',
      title: 'LlamaFarm'
    })

    // Load the Designer UI
    // The backend serves the Designer at localhost:8000 (via lf start/launch designer)
    // MUST use localhost (not 127.0.0.1) as the Designer's API config depends on it
    const designerURL = 'http://localhost:8000'
    console.log('Loading Designer from:', designerURL)
    this.mainWindow.loadURL(designerURL)

    // Inject CSS to make header draggable once page loads
    this.mainWindow.webContents.on('did-finish-load', () => {
      this.mainWindow?.webContents.insertCSS(`
        /* Make the Designer header draggable */
        header, [class*="header"], [class*="Header"] {
          -webkit-app-region: drag;
          padding-left: 80px !important; /* Make room for traffic lights */
        }
        /* But keep buttons/links clickable */
        header button, header a, header input, header select,
        [class*="header"] button, [class*="header"] a, [class*="header"] input, [class*="header"] select {
          -webkit-app-region: no-drag;
        }
      `)
    })

    // Open DevTools in development
    if (!app.isPackaged) {
      this.mainWindow.webContents.openDevTools()
    }

    // Show window when ready, or after timeout
    let shown = false
    const showWindow = () => {
      if (!shown && this.mainWindow && !this.mainWindow.isDestroyed()) {
        shown = true
        this.mainWindow.show()
        this.closeSplash()
      }
    }

    this.mainWindow.once('ready-to-show', showWindow)

    // Fallback: show window after 5 seconds even if not fully loaded
    setTimeout(() => {
      if (!shown) {
        console.log('Main window timeout - showing anyway')
        showWindow()
      }
    }, 5000)

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
