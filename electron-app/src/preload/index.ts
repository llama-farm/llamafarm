/**
 * Preload Script - Secure IPC bridge between main and renderer processes
 * This script runs in a privileged context and exposes safe APIs to the renderer
 */

import { contextBridge, ipcRenderer } from 'electron'

// Define the API that will be exposed to the renderer
const api = {
  // Backend operations
  backend: {
    getStatus: () => ipcRenderer.invoke('backend:status'),
    restart: () => ipcRenderer.invoke('backend:restart'),
    stop: () => ipcRenderer.invoke('backend:stop'),
    onStatusChange: (callback: (status: any) => void) => {
      const listener = (_event: any, status: any) => callback(status)
      ipcRenderer.on('backend-status', listener)
      // Return cleanup function to remove listener
      return () => ipcRenderer.removeListener('backend-status', listener)
    }
  },

  // CLI operations
  cli: {
    getInfo: () => ipcRenderer.invoke('cli:info')
  },

  // Splash screen operations
  splash: {
    onStatus: (callback: (status: any) => void) => {
      const listener = (_event: any, status: any) => callback(status)
      ipcRenderer.on('splash-status', listener)
      // Return cleanup function to remove listener
      return () => ipcRenderer.removeListener('splash-status', listener)
    },
    minimize: () => ipcRenderer.send('splash-minimize')
  },

  // System info
  platform: process.platform,
  version: process.versions.electron
}

// Expose the API to the renderer process
contextBridge.exposeInMainWorld('llamafarm', api)

// TypeScript declaration for the exposed API
export type LlamaFarmAPI = typeof api
