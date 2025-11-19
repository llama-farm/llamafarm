/**
 * Menu Manager - Creates application menu
 */

import { Menu, shell, app, BrowserWindow } from 'electron'

export class MenuManager {
  /**
   * Create and set application menu
   */
  createMenu(): void {
    const template: Electron.MenuItemConstructorOptions[] = [
      // App menu (macOS only)
      ...(process.platform === 'darwin'
        ? [
            {
              label: 'LlamaFarm',
              submenu: [
                { role: 'about' as const },
                { type: 'separator' as const },
                { role: 'hide' as const },
                { role: 'hideOthers' as const },
                { role: 'unhide' as const },
                { type: 'separator' as const },
                { role: 'quit' as const }
              ]
            }
          ]
        : []),

      // File menu
      {
        label: 'File',
        submenu: [
          process.platform === 'darwin'
            ? { role: 'close' as const }
            : { role: 'quit' as const }
        ]
      },

      // Edit menu
      {
        label: 'Edit',
        submenu: [
          { role: 'undo' as const },
          { role: 'redo' as const },
          { type: 'separator' as const },
          { role: 'cut' as const },
          { role: 'copy' as const },
          { role: 'paste' as const },
          { role: 'delete' as const },
          { type: 'separator' as const },
          { role: 'selectAll' as const }
        ]
      },

      // View menu
      {
        label: 'View',
        submenu: [
          { role: 'reload' as const },
          { role: 'forceReload' as const },
          { role: 'toggleDevTools' as const },
          { type: 'separator' as const },
          { role: 'resetZoom' as const },
          { role: 'zoomIn' as const },
          { role: 'zoomOut' as const },
          { type: 'separator' as const },
          { role: 'togglefullscreen' as const }
        ]
      },

      // Window menu
      {
        label: 'Window',
        submenu: [
          { role: 'minimize' as const },
          { role: 'zoom' as const },
          ...(process.platform === 'darwin'
            ? [
                { type: 'separator' as const },
                { role: 'front' as const },
                { type: 'separator' as const },
                { role: 'window' as const }
              ]
            : [{ role: 'close' as const }])
        ]
      },

      // Help menu
      {
        role: 'help' as const,
        submenu: [
          {
            label: 'LlamaFarm Website',
            click: async () => {
              await shell.openExternal('https://llamafarm.dev')
            }
          },
          {
            label: 'Documentation',
            click: async () => {
              await shell.openExternal('https://docs.llamafarm.dev')
            }
          },
          { type: 'separator' as const },
          {
            label: 'Open Source Project',
            click: async () => {
              await shell.openExternal('https://github.com/llama-farm/llamafarm')
            }
          },
          {
            label: 'Report Issue',
            click: async () => {
              await shell.openExternal('https://github.com/llama-farm/llamafarm/issues')
            }
          }
        ]
      }
    ]

    const menu = Menu.buildFromTemplate(template)
    Menu.setApplicationMenu(menu)
  }
}
