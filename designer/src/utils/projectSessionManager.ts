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
  serverId?: string    // Server-provided ID when available
  isPending?: boolean  // Waiting for server confirmation
  clientId: string     // Always track original client ID
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
 * Generate client session ID - clean, short identifier
 * Context information belongs in SessionMetadata, not the ID
 */
export function generateClientSessionId(): string {
  const timestamp = Date.now()
  const random = Math.random().toString(36).substring(2, 9)
  return `client_${timestamp}_${random}`
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
 * Cleanup pending sessions older than specified time
 * Conservative approach: only removes empty pending sessions
 */
export function cleanupPendingSessions(olderThanMinutes: number = 120): void { // Increased from 60 to 120
  try {
    const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
    const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
    const cutoff = Date.now() - (olderThanMinutes * 60 * 1000)
    
    let cleanedCount = 0
    let preservedCount = 0
    let skipCount = 0
    
    Object.entries(sessions).forEach(([sessionId, session]) => {
      const sessionAge = Date.now() - new Date(session.createdAt).getTime();
      const sessionAgeMinutes = sessionAge / (60 * 1000);
      
      // Only cleanup pending sessions that are old AND have no messages AND are really old (> cutoff)
      if (session.isPending && new Date(session.createdAt).getTime() < cutoff) {
        const messages = chatHistory[sessionId] || []
        const sessionAgeHours = sessionAgeMinutes / 60;
        
        // Additional safety: Don't cleanup sessions that are less than 1 hour old, regardless of other conditions
        // This prevents accidental cleanup of sessions that were just created
        if (sessionAgeHours < 1) {
          skipCount++
        } else if (messages.length === 0) {
          delete sessions[sessionId]
          delete chatHistory[sessionId]
          cleanedCount++
        } else {
          preservedCount++
        }
      } else {
        skipCount++
      }
    })
    
    if (cleanedCount > 0) {
      setToStorage(STORAGE_KEYS.SESSIONS, sessions)
      setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
    }
  } catch (error) {
    console.error('Failed to cleanup pending sessions:', error)
  }
}

/**
 * @deprecated Use cleanupPendingSessions instead
 * Cleanup orphaned temporary sessions (older than 1 hour)
 */
export function cleanupOrphanedTempSessions(): void {
  cleanupPendingSessions(60)
}

/**
 * Find existing session for context (namespace + project + chatService)
 * Prioritizes confirmed sessions over pending ones with enhanced debugging
 */
export function findSessionForContext(
  namespace: string,
  project: string,
  chatService: 'designer' | 'project'
): string | null {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  
  // Find all matching sessions
  const matches = Object.entries(sessions).filter(([, session]) => {
    return session.namespace === namespace &&
           session.project === project &&
           session.chatService === chatService;
  });
  
  if (matches.length === 0) {
    return null;
  }
  
  // Sort: confirmed sessions first, then by lastUsed (most recent first)
  matches.sort(([, a], [, b]) => {
    // Prioritize confirmed sessions (isPending: false)
    if (a.isPending !== b.isPending) {
      return a.isPending ? 1 : -1;
    }
    // Then sort by most recent
    return new Date(b.lastUsed).getTime() - new Date(a.lastUsed).getTime();
  });
  
  const [sessionId, sessionData] = matches[0];
  
  // Update lastUsed timestamp
  updateSessionLastUsed(sessionId);
  
  // Return the appropriate ID (serverId if available and different, otherwise sessionId)
  const finalId = sessionData.serverId && sessionData.serverId !== sessionId 
    ? sessionData.serverId 
    : sessionId;
    
  return finalId;
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
 * Create optimistic session with client-generated ID
 */
export function createOptimisticSession(
  namespace: string, 
  project: string, 
  chatService: 'designer' | 'project'
): string {
  const sessionId = generateClientSessionId()
  
  const sessionData: SessionMetadata = {
    namespace,
    project,
    chatService,
    createdAt: getCurrentTimestamp(),
    lastUsed: getCurrentTimestamp(),
    clientId: sessionId,
    isPending: true
  }
  
  saveSession(sessionId, sessionData)
  initializeChatHistory(sessionId)
  
  return sessionId
}

/**
 * Reconcile client session with server-provided session ID
 */
export function reconcileSessionWithServer(
  clientSessionId: string, 
  serverSessionId: string
): string {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  
  const sessionData = sessions[clientSessionId]
  if (!sessionData) {
    console.warn('Client session not found for reconciliation:', clientSessionId)
    return serverSessionId
  }

  // Get existing chat history from client session
  const existingMessages = chatHistory[clientSessionId] || []

  // Update session metadata
  const updatedSession: SessionMetadata = {
    ...sessionData,
    serverId: serverSessionId,
    isPending: false,
    lastUsed: getCurrentTimestamp()
  }

  if (clientSessionId === serverSessionId) {
    // Same ID, just update metadata
    sessions[clientSessionId] = updatedSession
  } else {
    // Different IDs, migrate data
    
    // Create new session with server ID
    sessions[serverSessionId] = updatedSession
    
    // Migrate chat history to server session ID
    chatHistory[serverSessionId] = existingMessages
    
    // Clean up old client session
    delete sessions[clientSessionId]
    delete chatHistory[clientSessionId]
  }

  // Save updated data
  setToStorage(STORAGE_KEYS.SESSIONS, sessions)
  setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
  
  return serverSessionId
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
    serverId: sessionId,
    clientId: sessionId,
    isPending: false
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
 * Load chat history for session with enhanced lookup and recovery
 * Checks both client and server session IDs to handle reconciliation cases
 */
export function loadChatHistory(sessionId: string): ChatMessage[] {
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  
  // First try the direct session ID
  if (chatHistory[sessionId]) {
    return chatHistory[sessionId]
  }
  
  // Get the session metadata for this session ID
  const session = sessions[sessionId]
  
  if (session) {
    // If this session has a client ID, try that
    if (session.clientId && session.clientId !== sessionId && chatHistory[session.clientId]) {
      // For safety, only return messages without migrating storage automatically
      // Let the reconciliation process handle this more carefully
      return chatHistory[session.clientId]
    }
    
    // If this session has a server ID, try that
    if (session.serverId && session.serverId !== sessionId && chatHistory[session.serverId]) {
      return chatHistory[session.serverId]
    }
  }
  
  // If this might be a server ID, find the session that has this as serverId
  const sessionWithThisServerId = Object.entries(sessions).find(
    ([, s]) => s.serverId === sessionId
  )
  
  if (sessionWithThisServerId) {
    const [clientId, sessionData] = sessionWithThisServerId
    
    // Try messages under the client ID
    if (chatHistory[clientId]) {
      // Migrate to server ID if different
      if (clientId !== sessionId) {
        const messages = chatHistory[clientId]
        chatHistory[sessionId] = messages
        delete chatHistory[clientId]
        setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
        return messages
      }
      return chatHistory[clientId]
    }
    
    // Try messages under the clientId of the session
    if (sessionData.clientId && chatHistory[sessionData.clientId]) {
      const messages = chatHistory[sessionData.clientId]
      chatHistory[sessionId] = messages
      delete chatHistory[sessionData.clientId]
      setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
      return messages
    }
  }
  
  return []
}

/**
 * Save chat history for session
 */
export function saveChatHistory(sessionId: string, messages: ChatMessage[]): void {
  try {
    const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
    chatHistory[sessionId] = messages
    setToStorage(STORAGE_KEYS.CHAT_HISTORY, chatHistory)
    updateSessionLastUsed(sessionId)
  } catch (error) {
    console.error('Failed to save chat history:', error);
  }
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
 * Debug utility to inspect session storage state
 * Call this from browser console: window.debugSessions()
 */
export function debugSessionStorage(): void {
  console.log('=== SESSION STORAGE DEBUG ===');
  
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  
  console.log('Sessions:', sessions);
  console.log('Chat History Keys:', Object.keys(chatHistory));
  
  Object.entries(sessions).forEach(([sessionId, session]) => {
    const messageCount = chatHistory[sessionId]?.length || 0
    const serverMessageCount = session.serverId && session.serverId !== sessionId ? 
      (chatHistory[session.serverId]?.length || 0) : 0
      
    console.log(`
📝 Session: ${sessionId}`);
    console.log(`  • Namespace: ${session.namespace}`);
    console.log(`  • Project: ${session.project}`);
    console.log(`  • Service: ${session.chatService}`);
    console.log(`  • Pending: ${session.isPending}`);
    console.log(`  • Client ID: ${session.clientId}`);
    console.log(`  • Server ID: ${session.serverId}`);
    console.log(`  • Messages (client): ${messageCount}`);
    if (serverMessageCount > 0) {
      console.log(`  • Messages (server): ${serverMessageCount}`);
    }
    
    if (messageCount > 0) {
      console.log(`  • Sample messages:`, chatHistory[sessionId]?.slice(0, 2));
    }
  });
  
  // Check for orphaned chat history
  const sessionIds = new Set(Object.keys(sessions))
  const serverIds = new Set(Object.values(sessions).map(s => s.serverId).filter(Boolean))
  const allValidIds = new Set([...sessionIds, ...serverIds])
  
  const orphanedHistory = Object.keys(chatHistory).filter(id => !allValidIds.has(id))
  if (orphanedHistory.length > 0) {
    console.log(`
⚠️ Orphaned chat history found:`, orphanedHistory);
    orphanedHistory.forEach(id => {
      console.log(`  • ${id}: ${chatHistory[id]?.length || 0} messages`);
    });
  }
  
  console.log('=== END DEBUG ===');
}

/**
 * Debug session state transitions
 * Call from browser console: window.debugSessionState()
 */
export function debugSessionState(): { sessions: SessionsStorage; chatHistory: ChatHistoryStorage } {
  const sessions = getFromStorage<SessionsStorage>(STORAGE_KEYS.SESSIONS, {})
  const chatHistory = getFromStorage<ChatHistoryStorage>(STORAGE_KEYS.CHAT_HISTORY, {})
  
  console.log('=== SESSION DEBUG ===');
  console.log('Total sessions:', Object.keys(sessions).length);
  console.log('Sessions with messages:', Object.keys(chatHistory).filter(id => (chatHistory[id] || []).length > 0).length);
  
  // Group sessions by project context
  const projectGroups: Record<string, Array<{ id: string; session: SessionMetadata; messageCount: number }>> = {}
  
  Object.entries(sessions).forEach(([id, session]) => {
    const messageCount = (chatHistory[id] || []).length
    const key = `${session.namespace}/${session.project}/${session.chatService}`
    
    if (!projectGroups[key]) projectGroups[key] = []
    projectGroups[key].push({ id, session, messageCount })
    
    console.log(`Session ${id}:`, {
      ...session,
      messageCount,
      hasMessages: messageCount > 0
    })
  })
  
  // Show duplicate sessions
  console.log('\n=== PROJECT GROUPS ===');
  Object.entries(projectGroups).forEach(([project, sessions]) => {
    console.log(`Project ${project}:`, sessions.length, 'sessions');
    if (sessions.length > 1) {
      console.log('🔴 MULTIPLE SESSIONS FOR SAME PROJECT:', sessions);
    }
    sessions.forEach(({ id, session, messageCount }) => {
      console.log(`  • ${id}: ${session.isPending ? 'PENDING' : 'CONFIRMED'}, ${messageCount} messages`);
    })
  })
  
  return { sessions, chatHistory }
}

// Make debug functions available globally for browser console
if (typeof window !== 'undefined') {
  (window as any).debugSessions = debugSessionStorage;
  (window as any).debugSessionState = debugSessionState;
}

/**
 * Export all session management functions
 */
export default {
  generateMessageId,
  generateClientSessionId,
  createOptimisticSession,
  reconcileSessionWithServer,
  findSessionForContext,
  getExistingSession,
  createSessionFromServer,
  saveSession,
  loadSession,
  updateSessionLastUsed,
  addMessageToHistory,
  loadChatHistory,
  saveChatHistory,
  clearChatHistory,
  deleteSession,
  getSessionsForContext,
  getAllSessions,
  isStorageAvailable,
  cleanupPendingSessions,
  cleanupOrphanedTempSessions,
  debugSessionStorage,
  debugSessionState,
}