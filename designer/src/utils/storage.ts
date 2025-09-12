/**
 * Centralized localStorage utilities for chat session management
 * Provides consistent key generation and storage operations across the app
 */

export interface SessionStorageKeys {
  currentSession: string
  sessionMessages: (sessionId: string) => string
  sessionList: string
}

/**
 * Generate project-specific localStorage keys
 */
export function getSessionStorageKeys(namespace?: string, project?: string): SessionStorageKeys {
  const hasProject = namespace && project
  
  return {
    currentSession: hasProject 
      ? `session_${namespace}_${project}` 
      : 'chatbox_current_session',
    sessionMessages: (sessionId: string) => `chatbox_messages_${sessionId}`,
    sessionList: hasProject 
      ? `chatbox_sessions_${namespace}_${project}` 
      : 'chatbox_sessions',
  }
}

/**
 * Session storage utilities with consistent error handling
 */
export class SessionStorage {
  /**
   * Get current session ID for a project
   */
  static getCurrentSessionId(namespace?: string, project?: string): string | null {
    if (typeof window === 'undefined') return null
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      return localStorage.getItem(keys.currentSession)
    } catch (error) {
      console.warn('Failed to get current session ID:', error)
      return null
    }
  }

  /**
   * Set current session ID for a project
   */
  static setCurrentSessionId(sessionId: string, namespace?: string, project?: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      localStorage.setItem(keys.currentSession, sessionId)
    } catch (error) {
      console.warn('Failed to set current session ID:', error)
    }
  }

  /**
   * Remove current session ID for a project
   */
  static removeCurrentSessionId(namespace?: string, project?: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      localStorage.removeItem(keys.currentSession)
    } catch (error) {
      console.warn('Failed to remove current session ID:', error)
    }
  }

  /**
   * Get messages for a session
   */
  static getSessionMessages(sessionId: string): any[] {
    if (typeof window === 'undefined') return []
    
    try {
      const keys = getSessionStorageKeys()
      const stored = localStorage.getItem(keys.sessionMessages(sessionId))
      if (!stored) return []
      
      const parsed = JSON.parse(stored)
      // Convert timestamp strings back to Date objects
      return parsed.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }))
    } catch (error) {
      console.warn(`Failed to get session messages for ${sessionId}:`, error)
      return []
    }
  }

  /**
   * Set messages for a session
   */
  static setSessionMessages(sessionId: string, messages: any[]): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys()
      localStorage.setItem(keys.sessionMessages(sessionId), JSON.stringify(messages))
    } catch (error) {
      console.warn(`Failed to set session messages for ${sessionId}:`, error)
    }
  }

  /**
   * Remove messages for a session
   */
  static removeSessionMessages(sessionId: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys()
      localStorage.removeItem(keys.sessionMessages(sessionId))
    } catch (error) {
      console.warn(`Failed to remove session messages for ${sessionId}:`, error)
    }
  }

  /**
   * Get all sessions for a project
   */
  static getSessionList(namespace?: string, project?: string): any[] {
    if (typeof window === 'undefined') return []
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      const stored = localStorage.getItem(keys.sessionList)
      if (!stored) return []
      
      const parsed = JSON.parse(stored)
      return parsed.map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastActivity: new Date(session.lastActivity)
      }))
    } catch (error) {
      console.warn('Failed to get session list:', error)
      return []
    }
  }

  /**
   * Set session list for a project
   */
  static setSessionList(sessions: any[], namespace?: string, project?: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      localStorage.setItem(keys.sessionList, JSON.stringify(sessions))
    } catch (error) {
      console.warn('Failed to set session list:', error)
    }
  }

  /**
   * Remove session list for a project
   */
  static removeSessionList(namespace?: string, project?: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const keys = getSessionStorageKeys(namespace, project)
      localStorage.removeItem(keys.sessionList)
    } catch (error) {
      console.warn('Failed to remove session list:', error)
    }
  }

  /**
   * Clear all session data for a project
   */
  static clearProjectSessions(namespace?: string, project?: string): void {
    if (typeof window === 'undefined') return
    
    try {
      const sessions = this.getSessionList(namespace, project)
      
      // Remove all session messages
      sessions.forEach(session => {
        this.removeSessionMessages(session.id)
      })
      
      // Remove session list and current session
      this.removeSessionList(namespace, project)
      this.removeCurrentSessionId(namespace, project)
    } catch (error) {
      console.warn('Failed to clear project sessions:', error)
    }
  }

  /**
   * Debug utility to inspect all session-related localStorage
   */
  static debugSessionStorage(): Record<string, string | null> {
    if (typeof window === 'undefined') return {}
    
    const result: Record<string, string | null> = {}
    
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i)
        if (key && (key.startsWith('session_') || key.startsWith('chatbox_'))) {
          result[key] = localStorage.getItem(key)
        }
      }
    } catch (error) {
      console.warn('Failed to debug session storage:', error)
    }
    
    return result
  }
}

export default SessionStorage
