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

  
  console.log('🔍 findExistingSession called with:', { namespace, project, chatService });
  console.log('🔍 Available sessions:', sessions);
  
  for (const [sessionId, session] of Object.entries(sessions)) {
    console.log('🔍 Checking session:', sessionId, {
      storedNamespace: session.namespace,
      storedProject: session.project,
      storedChatService: session.chatService,
      matches: {
        namespace: session.namespace === namespace,
        project: session.project === project,
        chatService: session.chatService === chatService
      }
    });
    
    if (session.namespace === namespace &&
        session.project === project &&
        session.chatService === chatService) {
      console.log('✅ Found matching session:', sessionId);
      return sessionId
    }
  }
  
  console.log('❌ No matching session found');
  return null
}

/**
 * Create a message object with generated ID and timestamp
 */
function createMessage(role: 'user' | 'assistant', content: string): ChatMessage {
  // Add validation and logging
  console.log('🔍 Creating message:', { role, content: `"${content}"`, contentLength: content.length });
  
  if (!content || content.trim() === '') {
    console.warn('⚠️ Attempting to create message with empty content!');
    console.trace('Empty message creation stack trace');
    throw new Error('Cannot create message with empty content');
  }
  
  const message = {
    id: generateMessageId(),
    role,
    content: content.trim(), // Ensure we trim whitespace
    timestamp: new Date().toISOString()
  };
  
  console.log('✅ Created message:', message);
  return message;
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
  console.log('💾 Creating persistent session:', sessionId, 'with', initialMessages.length, 'messages');
  
  // SAFEGUARD: Check if session already exists with messages
  const existingSessions = getStoredSessions();
  if (existingSessions[sessionId] && existingSessions[sessionId].messages.length > 0) {
    console.warn('⚠️ Session already exists with', existingSessions[sessionId].messages.length, 'messages');
    console.log('⚠️ New session would have', initialMessages.length, 'messages');
    
    if (initialMessages.length === 0) {
      console.error('❌ PREVENTED: Overwrite of session with empty messages array!');
      console.log('❌ Keeping existing session with', existingSessions[sessionId].messages.length, 'messages');
      return; // Don't overwrite existing session with empty array
    }
    
    if (initialMessages.length < existingSessions[sessionId].messages.length) {
      console.warn('⚠️ WARNING: New session has fewer messages than existing. Proceeding but this may be unintended.');
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
  
  // Verify save worked
  const verification = getStoredSessions();
  if (verification[sessionId]) {
    console.log('✅ Session saved successfully:', sessionId, 'with', verification[sessionId].messages.length, 'messages');
  } else {
    console.error('❌ Session save failed:', sessionId);
  }
}

/**
 * Add a message to an existing persistent session
 */
function addMessageToPersistentSession(sessionId: string, message: ChatMessage): void {
  console.log('💾 Adding message to persistent session:', sessionId);
  
  const sessions = getStoredSessions();
  if (sessions[sessionId]) {
    sessions[sessionId].messages.push(message);
    sessions[sessionId].lastUsed = new Date().toISOString();
    saveStoredSessions(sessions);
    console.log('✅ Message added to persistent session:', sessionId);
  } else {
    console.error('❌ Session not found for message addition:', sessionId);
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
    console.log('✅ Created test session:', sessionId, 'with', testMessages.length, 'messages')
    return { sessionId, messages: testMessages }
  }
  
  // Test empty message validation
  (window as any).testEmptyMessage = () => {
    console.log('🧪 Testing empty message validation...');
    try {
      createMessage('user', '');
    } catch (error) {
      console.log('✅ Empty message correctly rejected:', error instanceof Error ? error.message : String(error));
    }
    
    try {
      createMessage('assistant', '   ');
    } catch (error) {
      console.log('✅ Whitespace-only message correctly rejected:', error instanceof Error ? error.message : String(error));
    }
    
    try {
      const validMessage = createMessage('user', 'This is valid');
      console.log('✅ Valid message created:', validMessage);
    } catch (error) {
      console.log('❌ Valid message incorrectly rejected:', error instanceof Error ? error.message : String(error));
    }
    
    console.log('🧪 Empty message validation test complete');
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