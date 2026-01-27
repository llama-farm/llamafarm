/**
 * Global event emitter for DevTools capture.
 *
 * This allows axios interceptors (which run outside React context) to emit
 * request/response events that the DevToolsContext can subscribe to.
 *
 * Architecture:
 *   Axios Interceptor ──emit──> DevToolsEmitter ──subscribe──> DevToolsContext
 *   WebSocket wrapper ──emit──> DevToolsEmitter ──subscribe──> DevToolsContext
 */

import type { CapturedRequest, WebSocketDirection } from '../contexts/DevToolsContext'

// Response data structure for the emitter
export interface DevToolsResponseData {
  status: number
  statusText: string
  headers: Record<string, string>
  body: unknown
  requestId?: string | null
}

// Event types that can be emitted
export type DevToolsEvent =
  // HTTP events
  | { type: 'request'; request: Omit<CapturedRequest, 'streamChunks' | 'streamComplete'> }
  | { type: 'response'; id: string; response: DevToolsResponseData }
  | { type: 'error'; id: string; error: string }
  // WebSocket events
  | { type: 'ws_open'; id: string; url: string }
  | { type: 'ws_message'; connectionId: string; direction: WebSocketDirection; data: any; isBinary: boolean; size: number }
  | { type: 'ws_close'; id: string; error?: string }

type DevToolsEventListener = (event: DevToolsEvent) => void

/**
 * Simple pub/sub event emitter for DevTools capture events.
 * Singleton instance is exported for use by axios interceptors and DevToolsContext.
 */
class DevToolsEmitter {
  private listeners: Set<DevToolsEventListener> = new Set()

  /**
   * Emit an event to all subscribed listeners
   */
  emit(event: DevToolsEvent): void {
    this.listeners.forEach(listener => {
      try {
        listener(event)
      } catch (err) {
        // Don't let listener errors break the emitter
        console.error('[DevToolsEmitter] Listener error:', err)
      }
    })
  }

  /**
   * Subscribe to DevTools events
   * @returns Unsubscribe function
   */
  subscribe(listener: DevToolsEventListener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /**
   * Check if there are any active subscribers
   * (useful for conditionally skipping capture when DevTools isn't mounted)
   */
  hasSubscribers(): boolean {
    return this.listeners.size > 0
  }
}

// Export singleton instance
export const devToolsEmitter = new DevToolsEmitter()
