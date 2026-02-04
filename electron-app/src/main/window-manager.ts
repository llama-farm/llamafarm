/**
 * Window Manager - Manages Electron windows (splash, main, etc.)
 */

import { BrowserWindow, app, screen } from 'electron'
import * as path from 'path'
import * as fs from 'fs'

export class WindowManager {
  private splashWindow: BrowserWindow | null = null
  private mainWindow: BrowserWindow | null = null
  private _loadingComplete = false
  private _wasMinimizedOnComplete = false

  /**
   * Create splash screen
   */
  createSplashWindow(): BrowserWindow {
    this.splashWindow = new BrowserWindow({
      width: 500,
      height: 400,
      frame: false,
      transparent: false,
      resizable: false,
      alwaysOnTop: false,
      movable: true,
      minimizable: true,
      backgroundColor: '#1a1f2e',
      webPreferences: {
        preload: path.join(__dirname, '../preload/index.js'),
        nodeIntegration: false,
        contextIsolation: true
      }
    })

    // Load logo as base64
    let logoBase64 = ''
    try {
      // In development: logo is in build folder
      // In production: logo should be in extraResources
      const possiblePaths = [
        path.join(__dirname, '../../assets/images/splash-logo.png'),   // Development
        path.join(__dirname, '../../../assets/images/splash-logo.png'), // Packaged (app.asar)
        path.join(process.resourcesPath || '', 'splash-logo.png'),     // extraResources
        path.join(app.getAppPath(), 'assets/images/splash-logo.png')   // App path fallback
      ]

      for (const logoPath of possiblePaths) {
        if (fs.existsSync(logoPath)) {
          const logoData = fs.readFileSync(logoPath)
          logoBase64 = logoData.toString('base64')
          console.log('Loaded splash logo from:', logoPath)
          break
        }
      }

      if (!logoBase64) {
        console.warn('Splash logo not found in any expected location')
      }
    } catch (error) {
      console.warn('Failed to load splash logo:', error)
    }

    // Load embedded splash HTML with model status support
    const splashHTML = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="UTF-8">
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              background: linear-gradient(165deg, hsl(222, 47%, 14%) 0%, hsl(222, 47%, 11%) 50%, hsl(225, 50%, 9%) 100%);
              color: hsl(210, 40%, 98%);
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              height: 100vh;
              overflow: hidden;
            }
            .drag-bar {
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              height: 32px;
              -webkit-app-region: drag;
              display: flex;
              align-items: center;
              justify-content: flex-end;
              padding: 0 8px;
            }
            .window-controls {
              -webkit-app-region: no-drag;
            }
            .minimize-btn {
              width: 24px;
              height: 24px;
              border: none;
              border-radius: 4px;
              background: transparent;
              color: hsl(215, 20%, 65%);
              font-size: 18px;
              cursor: pointer;
              display: flex;
              align-items: center;
              justify-content: center;
              line-height: 1;
            }
            .minimize-btn:hover {
              background: hsl(215, 28%, 25%);
            }
            .logo {
              width: 180px;
              height: auto;
              margin-bottom: 36px;
              filter: drop-shadow(0 0 24px rgba(34, 211, 238, 0.25));
            }
            .status {
              font-size: 15px;
              opacity: 0.8;
              margin-bottom: 20px;
              color: hsl(215, 20%, 65%);
            }
            /* Error display - positioned below progress bar */
            .error-container {
              position: absolute;
              bottom: 40px;
              left: 50%;
              transform: translateX(-50%);
              width: 380px;
              display: none;
            }
            .error-container.has-errors {
              display: block;
            }
            .error-summary {
              display: flex;
              align-items: center;
              gap: 8px;
              padding: 12px 16px;
              background: hsla(0, 40%, 15%, 0.6);
              border: 1px solid hsla(0, 50%, 35%, 0.4);
              border-radius: 8px;
              font-size: 13px;
              color: hsl(0, 70%, 70%);
            }
            .error-icon {
              font-size: 14px;
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
          <div class="drag-bar">
            <div class="window-controls">
              <button class="minimize-btn" onclick="window.llamafarm.splash.minimize()" title="Minimize">−</button>
            </div>
          </div>
          ${logoBase64 ? `<img class="logo" src="data:image/png;base64,${logoBase64}" alt="LlamaFarm" />` : '<div class="logo" style="font-size: 64px;">🦙</div>'}
          <div class="status" id="status">Starting...</div>
          <div class="progress-container">
            <div class="progress-bar" id="progress"></div>
          </div>
          <div class="spinner" id="spinner"></div>
          <div class="error" id="error" style="display: none;"></div>
          <div class="error-container" id="error-container"></div>

          <script>
            // Show error summary with retry button
            function renderModels(models) {
              const container = document.getElementById('error-container');
              if (!models || models.length === 0) {
                container.className = 'error-container';
                container.innerHTML = '';
                return;
              }

              // Filter to only show models with errors
              const errorModels = models.filter(m => m.status === 'error');

              if (errorModels.length === 0) {
                container.className = 'error-container';
                container.innerHTML = '';
                return;
              }

              // Show compact error summary with retry button
              container.className = 'error-container has-errors';
              const count = errorModels.length;
              const modelText = count === 1 ? 'model' : 'models';

              container.innerHTML = '<div class="error-summary"><span class="error-icon">⚠</span> ' + count + ' ' + modelText + ' failed to load</div>';
            }

            window.llamafarm.splash.onStatus((status) => {
              document.getElementById('status').textContent = status.message;
              if (status.progress !== undefined) {
                document.getElementById('progress').style.width = status.progress + '%';
              }
              if (status.error) {
                document.getElementById('error').textContent = status.error;
                document.getElementById('error').style.display = 'block';
              }
              if (status.models) {
                renderModels(status.models);
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
  updateSplash(status: {
    message: string
    progress?: number
    error?: string
    models?: Array<{
      id: string
      display_name: string
      status: 'checking' | 'present' | 'downloading' | 'error'
      progress?: number
    }>
  }): void {
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
    // The backend serves the Designer at localhost:14345 (via lf start/launch designer)
    // MUST use localhost (not 127.0.0.1) as the Designer's API config depends on it
    const designerURL = 'http://localhost:14345'
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

        // Check if splash was minimized when loading completed
        const splashMinimized = this.splashWindow?.isMinimized() ?? false

        if (splashMinimized) {
          // Don't force window - user will activate via dock/taskbar
          this._loadingComplete = true
          this._wasMinimizedOnComplete = true
          this.notifyLoadingComplete()
          // Keep splash minimized, don't show main window yet
        } else {
          // Normal flow: show main window, close splash
          this.mainWindow.show()
          this.closeSplash()
        }
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

  /**
   * Check if loading completed while window was minimized
   */
  get wasMinimizedOnComplete(): boolean {
    return this._wasMinimizedOnComplete
  }

  /**
   * Notify user that loading is complete (platform-specific)
   */
  private notifyLoadingComplete(): void {
    switch (process.platform) {
      case 'darwin':
        // macOS: Bounce dock icon (critical = bounce until user responds)
        app.dock?.bounce('critical')
        break

      case 'win32':
        // Windows: Flash taskbar
        if (this.splashWindow && !this.splashWindow.isDestroyed()) {
          this.splashWindow.flashFrame(true)
        }
        break

      case 'linux':
        // Linux: Set urgency hint (flashes taskbar where supported)
        if (this.splashWindow && !this.splashWindow.isDestroyed()) {
          this.splashWindow.setProgressBar(1) // Show complete
          this.splashWindow.flashFrame(true) // Flash if supported
        }
        break
    }
  }

  /**
   * Handle app activation (dock click, taskbar click)
   * Called from index.ts onActivate when loading completed while minimized
   */
  showMainWindowFromActivation(): void {
    if (this._wasMinimizedOnComplete && this.mainWindow) {
      // Loading completed while minimized, now user activated
      this.closeSplash()
      this.mainWindow.show()
      this.mainWindow.focus()
      this._wasMinimizedOnComplete = false
      this._loadingComplete = false
    }
  }
}
