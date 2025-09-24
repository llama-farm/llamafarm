/**
 * Project Session Manager - Phase 1: Simplified Storage
 * 
 * Simple single-bucket storage for project sessions with messages included.
 * Only server-provided session IDs are stored - no client session generation.
 */

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

interface SessionData {
  namespace: string
  project: string
  chatService: 'designer' | 'project'
  createdAt: string
  lastUsed: string
  messages: ChatMessage[]
}

type SessionStorage = Record<string, SessionData>

// Single localStorage key
const STORAGE_KEY = 'lf_sessions'

/**
 * Generate unique message ID with format: msg_{timestamp}_{randomId}
 */
export function generateMessageId(): string {
  const timestamp = Date.now()
  const random = Math.random().toString(36).substring(2, 9)
  return `msg_${timestamp}_${random}`
}

/**
 * Get stored sessions from localStorage
 */
function getStoredSessions(): SessionStorage {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : {}
  } catch (error) {
    console.warn('Failed to parse sessions from localStorage:', error)
    return {}
  }
}

/**
 * Save sessions to localStorage
 */
function saveStoredSessions(sessions: SessionStorage): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
  } catch (error) {
    console.error('Failed to save sessions to localStorage:', error)
  }
}

/**
 * Find existing session for context
 */
function findExistingSession(
  namespace: string,
  project: string,
  chatService: 'designer' | 'project'
): string | null {
  const sessions = getStoredSessions()
  
  for (const [sessionId, session] of Object.entries(sessions)) {
    if (session.namespace === namespace &&
        session.project === project &&
        session.chatService === chatService) {
      return sessionId
    }
  }
  
  return null
}

/**
 * Create a message object with generated ID and timestamp
 */
function createMessage(role: 'user' | 'assistant', content: string): ChatMessage {
  if (!content || content.trim() === '') {
    throw new Error('Cannot create message with empty content');
  }
  
  return {
    id: generateMessageId(),
    role,
    content: content.trim(), // Ensure we trim whitespace
    timestamp: new Date().toISOString()
  };
}

/**
 * Create and save a persistent session to localStorage
 */
function createPersistentSession(
  sessionId: string,
  namespace: string,
  project: string,
  chatService: 'designer' | 'project',
  initialMessages: ChatMessage[] = []
): void {
  // SAFEGUARD: Check if session already exists with messages
  const existingSessions = getStoredSessions();
  if (existingSessions[sessionId] && existingSessions[sessionId].messages.length > 0) {
    if (initialMessages.length === 0) {
      return; // Don't overwrite existing session with empty array
    }
  }
  
  const sessions = getStoredSessions()
  sessions[sessionId] = {
    namespace,
    project,
    chatService,
    createdAt: existingSessions[sessionId]?.createdAt || new Date().toISOString(),
    lastUsed: new Date().toISOString(),
    messages: initialMessages
  }
  saveStoredSessions(sessions)
}

/**
 * Add a message to an existing persistent session
 */
function addMessageToPersistentSession(sessionId: string, message: ChatMessage): void {
  const sessions = getStoredSessions();
  if (sessions[sessionId]) {
    sessions[sessionId].messages.push(message);
    sessions[sessionId].lastUsed = new Date().toISOString();
    saveStoredSessions(sessions);
  } else {
    console.error('Session not found for message addition:', sessionId);
  }
}


// Make functions available globally for testing in browser console
if (typeof window !== 'undefined') {
  // Individual functions
  (window as any).getStoredSessions = getStoredSessions;
  (window as any).saveStoredSessions = saveStoredSessions;
  (window as any).findExistingSession = findExistingSession;
  (window as any).generateMessageId = generateMessageId;
  (window as any).createMessage = createMessage;
  (window as any).createPersistentSession = createPersistentSession;
  (window as any).addMessageToPersistentSession = addMessageToPersistentSession;
  
  // Debug helper for session restoration issues
  (window as any).debugSessionRestore = () => {
    console.log('=== SESSION RESTORE DEBUG ===');
    const sessions = getStoredSessions();
    console.log('All stored sessions:', sessions);
    console.log('Session count:', Object.keys(sessions).length);
    
    Object.entries(sessions).forEach(([sessionId, session]) => {
      console.log(`Session ${sessionId}:`, {
        namespace: session.namespace,
        project: session.project,
        chatService: session.chatService,
        messageCount: session.messages?.length || 0,
        lastUsed: session.lastUsed
      });
    });
    
    console.log('=== END DEBUG ===');
  };
  
  // Phase 2: Add test helper for creating sessions with messages
  (window as any).createTestSession = (sessionId: string, namespace: string = 'default', project: string = 'testproject') => {
    const testMessages = [
      createMessage('user', 'Hello, this is a test message'),
      createMessage('assistant', 'Hello! This is a test response from the assistant.')
    ]
    createPersistentSession(sessionId, namespace, project, 'designer', testMessages)
    return { sessionId, messages: testMessages }
  }
  
  // Test empty message validation
  (window as any).testEmptyMessage = () => {
    try {
      createMessage('user', '');
    } catch (error) {
      // Empty message correctly rejected
    }
    
    try {
      createMessage('assistant', '   ');
    } catch (error) {
      // Whitespace-only message correctly rejected
    }
    
    try {
      createMessage('user', 'This is valid');
      // Valid message created
    } catch (error) {
      // Valid message incorrectly rejected
    }
  }
}

/**
 * Phase 3: Export message helper functions including persistence
 */
export {
  getStoredSessions,
  saveStoredSessions,
  findExistingSession,
  createMessage,
  createPersistentSession,
  addMessageToPersistentSession
}