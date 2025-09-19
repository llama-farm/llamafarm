/**
 * Project Session Manager
 * 
 * Manages localStorage structure for Designer Chat (chatService) and Project Chat (projectChatService)
 * with separate conversation streams per project context.
 * 
 * Storage Structure:
 * - "llamafarm_project_sessions": Session metadata by session ID
 * - "llamafarm_project_chat_history": Chat message history by session ID
 */

export interface SessionMetadata {
  namespace: string
  project: string
  chatService: 'designer' | 'project'
  createdAt: string
  lastUsed: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

export interface SessionsStorage {
  [sessionId: string]: SessionMetadata
}

export interface ChatHistoryStorage {
  [sessionId: string]: ChatMessage[]
}

// localStorage keys
const STORAGE_KEYS = {
  SESSIONS: 'llamafarm_project_sessions',
  CHAT_HISTORY: 'llamafarm_project_chat_history',
} as const

/**
 * Generate unique message ID with format: msg_{timestamp}_{randomId}
 */
export function generateMessageId(): string {
  const timestamp = Date.now()
  const random = Math.random().toString(36).substring(2, 9)
  return `msg_${timestamp}_${random}`
}

/**
 * Get current timestamp in ISO format
 */
function getCurrentTimestamp(): string {
  return new Date().toISOString()
}

/**
 * Safely get data from localStorage with error handling
 */
function getFromStorage<T>(key: string, defaultValue: T): T {
  try {
    const data = localStorage.getItem(key)
    return data ? JSON.parse(data) : defaultValue
  } catch (error) {
    console.warn(`Failed to parse ${key} from localStorage:`, error)
    return defaultValue
  }
}

/**
 * Safely set data to localStorage with error handling
 */
function setToStorage<T>(key: string, data: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(data))
  } catch (error) {
    console.error(`Failed to save ${key} to localStorage:`, error)
    // Handle quota exceeded or other storage errors
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      cleanupOldSessions()
      // Try again after cleanup
      try {
        localStorage.setItem(key, JSON.stringify(data))
      } catch (retryError) {
        console.error(`Failed to save ${key} even after cleanup:`, retryError)
      }
    }
  }
}

/**
 * Cleanup old sessions when storage quota is exceeded
 */
function cleanupOldSessions(): void {
  try {
    const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
    const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
    
    // Get sessions sorted by lastUsed (oldest first)
    const sessionEntries = Object.entries(sessions).sort(
      ([, a], [, b]) => new Date(a.lastUsed).getTime() - new Date(b.lastUsed).getTime()
    )
    
    // Remove oldest 50% of sessions
    const toRemove = Math.ceil(sessionEntries.length * 0.5)
    const sessionsToRemove = sessionEntries.slice(0, toRemove)
    
    sessionsToRemove.forEach(([sessionId]) => {
      delete sessions[sessionId]
      delete chatHistory[sessionId]
    })
    
    setToStorage(STORAGE_KEYS.SESSIONS, sessions)
    setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
    
    console.log(`Cleaned up ${toRemove} old sessions due to storage quota`)
  } catch (error) {
    console.error('Failed to cleanup old sessions:', error)
  }
}

/**
 * Cleanup orphaned temporary sessions (older than 1 hour)
 */
export function cleanupOrphanedTempSessions(): void {
  try {
    const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
    const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
    const now = Date.now()
    const oneHourAgo = now - (60 * 60 * 1000)
    
    let cleanedCount = 0
    
    Object.entries(sessions).forEach(([sessionId, session]) => {
      if (sessionId.startsWith('temp_')) {
        const createdAt = new Date(session.createdAt).getTime()
        if (createdAt < oneHourAgo) {
          console.log('Cleaning up orphaned temp session:', sessionId)
          delete sessions[sessionId]
          delete chatHistory[sessionId]
          cleanedCount++
        }
      }
    })
    
    if (cleanedCount > 0) {
      setToStorage(STORAGE_KEYS.SESSIONS, sessions)
      setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
      console.log(`Cleaned up ${cleanedCount} orphaned temporary sessions`)
    }
  } catch (error) {
    console.error('Failed to cleanup orphaned temp sessions:', error)
  }
}

/**
 * Find existing session for context (namespace + project + chatService)
 */
export function findSessionForContext(
  namespace: string,
  project: string,
  chatService: 'designer' | 'project'
): string | null {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  
  for (const [sessionId, metadata] of Object.entries(sessions)) {
    if (
      metadata.namespace === namespace &&
      metadata.project === project &&
      metadata.chatService === chatService
    ) {
      return sessionId
    }
  }
  
  return null
}

/**
 * Initialize empty chat history for session
 */
export function initializeChatHistory(sessionId: string): void {
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  chatHistory[sessionId] = []
  setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
}

/**
 * Get existing session for current context or return null
 * Sessions are created on first message send, not proactively
 */
export function getExistingSession(
  namespace: string,
  project: string,
  chatService: 'designer' | 'project'
): string | null {
  const existingSessionId = findSessionForContext(namespace, project, chatService)
  if (existingSessionId) {
    updateSessionLastUsed(existingSessionId)
    return existingSessionId
  }
  return null
}

/**
 * Create and store a new session with server-provided session ID
 * This is called after receiving a session ID from the server
 */
export function createSessionFromServer(
  sessionId: string,
  namespace: string,
  project: string,
  chatService: 'designer' | 'project'
): void {
  const sessionData: SessionMetadata = {
    namespace,
    project,
    chatService,
    createdAt: getCurrentTimestamp(),
    lastUsed: getCurrentTimestamp(),
  }
  
  saveSession(sessionId, sessionData)
  initializeChatHistory(sessionId)
}

/**
 * Save session metadata to localStorage
 */
export function saveSession(sessionId: string, sessionData: SessionMetadata): void {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  sessions[sessionId] = sessionData
  setToStorage(STORAGE_KEYS.SESSIONS, sessions)
}

/**
 * Load session by ID
 */
export function loadSession(sessionId: string): SessionMetadata | null {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  return sessions[sessionId] || null
}

/**
 * Update session lastUsed timestamp
 */
export function updateSessionLastUsed(sessionId: string): void {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  if (sessions[sessionId]) {
    sessions[sessionId].lastUsed = getCurrentTimestamp()
    setToStorage(STORAGE_KEYS.SESSIONS, sessions)
  }
}

/**
 * Add message to chat history
 */
export function addMessageToHistory(sessionId: string, message: Omit<ChatMessage, 'id'>): ChatMessage {
  const messageWithId: ChatMessage = {
    ...message,
    id: generateMessageId(),
  }
  
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  
  if (!chatHistory[sessionId]) {
    chatHistory[sessionId] = []
  }
  
  chatHistory[sessionId].push(messageWithId)
  setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
  
  // Update session lastUsed
  updateSessionLastUsed(sessionId)
  
  return messageWithId
}

/**
 * Load chat history for session
 */
export function loadChatHistory(sessionId: string): ChatMessage[] {
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  return chatHistory[sessionId] || []
}

/**
 * Clear chat history for session
 */
export function clearChatHistory(sessionId: string): void {
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  chatHistory[sessionId] = []
  setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
  updateSessionLastUsed(sessionId)
}

/**
 * Delete session and its chat history
 */
export function deleteSession(sessionId: string): void {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  
  delete sessions[sessionId]
  delete chatHistory[sessionId]
  
  setToStorage(STORAGE_KEYS.SESSIONS, sessions)
  setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
}

/**
 * Get all sessions for a specific context
 */
export function getSessionsForContext(
  namespace: string,
  project: string,
  chatService?: 'designer' | 'project'
): Array<{sessionId: string, metadata: SessionMetadata}> {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  
  return Object.entries(sessions)
    .filter(([, metadata]) => {
      const matchesNamespace = metadata.namespace === namespace
      const matchesProject = metadata.project === project
      const matchesService = !chatService || metadata.chatService === chatService
      
      return matchesNamespace && matchesProject && matchesService
    })
    .map(([sessionId, metadata]) => ({ sessionId, metadata }))
    .sort((a, b) => 
      new Date(b.metadata.lastUsed).getTime() - new Date(a.metadata.lastUsed).getTime()
    )
}

/**
 * Get all sessions
 */
export function getAllSessions(): Array<{sessionId: string, metadata: SessionMetadata}> {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  
  return Object.entries(sessions)
    .map(([sessionId, metadata]) => ({ sessionId, metadata }))
    .sort((a, b) => 
      new Date(b.metadata.lastUsed).getTime() - new Date(a.metadata.lastUsed).getTime()
    )
}

/**
 * Check if localStorage is available
 */
export function isStorageAvailable(): boolean {
  try {
    const testKey = '__storage_test__'
    localStorage.setItem(testKey, 'test')
    localStorage.removeItem(testKey)
    return true
  } catch (error) {
    return false
  }
}

/**
 * Export all session management functions
 */
export default {
  generateMessageId,
  findSessionForContext,
  getExistingSession,
  createSessionFromServer,
  saveSession,
  loadSession,
  updateSessionLastUsed,
  addMessageToHistory,
  loadChatHistory,
  clearChatHistory,
  deleteSession,
  getSessionsForContext,
  getAllSessions,
  isStorageAvailable,
  cleanupOrphanedTempSessions,
}