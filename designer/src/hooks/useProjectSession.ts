/**
 * React hook for project session management
 * 
 * Manages session state for both Designer Chat and Project Chat services
 * with localStorage persistence and project context switching
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useActiveProject } from './useActiveProject'
import {
  findExistingSession,
  getStoredSessions,
  saveStoredSessions,
  createMessage,
  createPersistentSession,
  addMessageToPersistentSession,
  type ChatMessage,
} from '../utils/projectSessionManager'

export interface ProjectSessionOptions {
  chatService: 'designer' | 'project'
  autoCreate?: boolean
}

export interface ProjectSessionState {
  sessionId: string | null
  messages: ChatMessage[]
  error: string | null
  isTemporaryMode: boolean
  tempMessages: ChatMessage[]
}

export interface ProjectSessionActions {
  addMessage: (content: string, role: 'user' | 'assistant') => ChatMessage
  addTempMessage: (message: ChatMessage) => void
  addPersistentMessage: (message: ChatMessage) => void
  clearHistory: () => void
  deleteCurrentSession: () => void
  refreshSession: () => void
  createSessionFromServer: (serverSessionId: string) => void
  reconcileWithServer: (clientSessionId: string, serverSessionId: string) => void
  debugState: () => void
}

/**
 * Hook for managing project sessions with project context integration
 * Phase 2: Added temporary message state for optimistic updates
 */
export function useProjectSession(
  options: ProjectSessionOptions
): ProjectSessionState & ProjectSessionActions {
  const { chatService } = options
  const activeProject = useActiveProject()
  
  // Existing state
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [error, setError] = useState<string | null>(null)
  
  // NEW: Add temporary message state for pre-session messages
  const [tempMessages, setTempMessages] = useState<ChatMessage[]>([])
  
  // FIX: Add ref to track current temp messages to avoid stale closure
  const tempMessagesRef = useRef<ChatMessage[]>([])
  
  // FIX: Add transfer lock to prevent session resets during/after transfer
  const transferLockRef = useRef(false)
  
  // FIX: Helper function to check if session reset should be prevented
  const shouldPreventReset = useCallback((context: string) => {
    if (transferLockRef.current) {
      console.log(`🚨 PREVENTED SESSION RESET during transfer - context: ${context}`);
      console.trace('Stack trace for prevented reset');
      return true;
    }
    return false;
  }, [])
  
  // NEW: Determine if we're in temporary or persistent mode
  const isTemporaryMode = !sessionId
  const displayMessages = isTemporaryMode ? tempMessages : messages
  
  // DEBUG: Log display message calculation
  useEffect(() => {
    console.log('📺 Display messages calculation:', {
      sessionId,
      isTemporaryMode,
      tempMessagesLength: tempMessages.length,
      messagesLength: messages.length,
      displayMessagesLength: displayMessages.length,
      showingTemp: isTemporaryMode,
      showingPersistent: !isTemporaryMode,
      source: isTemporaryMode ? 'tempMessages (temp mode)' : 'messages (persistent mode)'
    });
    
    if (!isTemporaryMode && messages.length > 0 && displayMessages.length === 0) {
      console.error('🚨 DISPLAY BUG: In persistent mode with messages but displayMessages is empty!');
    }
    
    if (isTemporaryMode && tempMessages.length === 0 && messages.length > 0) {
      console.warn('⚠️ POTENTIAL ISSUE: In temp mode but have persistent messages');
    }
  }, [sessionId, isTemporaryMode, tempMessages.length, messages.length, displayMessages.length, messages, tempMessages])
  
  // FIX: Update ref whenever tempMessages changes to avoid stale closure
  useEffect(() => {
    tempMessagesRef.current = tempMessages;
    console.log('📝 Updated tempMessagesRef.current with', tempMessages.length, 'messages');
  }, [tempMessages]);

  // DEBUG: State inspection function
  const debugState = useCallback(() => {
    console.log('🔍 === PROJECT SESSION STATE DEBUG ===');
    console.log('🔍 sessionId:', sessionId);
    console.log('🔍 sessionId type:', typeof sessionId);
    console.log('🔍 sessionId truthy:', !!sessionId);
    console.log('🔍 isTemporaryMode:', isTemporaryMode);
    console.log('🔍 isTemporaryMode calculation (!sessionId):', !sessionId);
    console.log('🔍 tempMessages array:', tempMessages);
    console.log('🔍 tempMessages length:', tempMessages.length);
    console.log('🔍 tempMessagesRef.current:', tempMessagesRef.current);
    console.log('🔍 tempMessagesRef.current length:', tempMessagesRef.current.length);
    console.log('🔍 messages array:', messages);
    console.log('🔍 messages length:', messages.length);
    console.log('🔍 displayMessages array:', displayMessages);
    console.log('🔍 displayMessages length:', displayMessages.length);
    console.log('🔍 displayMessages source:', isTemporaryMode ? 'tempMessages' : 'messages');
    console.log('🔍 UI should show:', displayMessages.length, 'messages');
    console.log('🔍 activeProject:', activeProject);
    console.log('🔍 === END DEBUG ===');
  }, [sessionId, isTemporaryMode, tempMessages, messages, displayMessages, activeProject])
  
  // Project change effect - update to clear temp messages too
  useEffect(() => {
    // Don't reset session if we're in the middle of a transfer
    if (shouldPreventReset('project change effect')) {
      return;
    }
    
    console.log('🔍 Session lookup triggered by project/service change');
    
    if (activeProject) {
      console.log('🔍 Active project details:', {
        namespace: activeProject.namespace,
        project: activeProject.project,
        chatService: chatService
      });
      
      // Debug storage contents
      const allSessions = getStoredSessions();
      console.log('🔍 All stored sessions:', allSessions);
      console.log('🔍 Session count:', Object.keys(allSessions).length);
      
      const existingSessionId = findExistingSession(
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      
      console.log('🔍 findExistingSession result:', existingSessionId);
      
      if (existingSessionId) {
        // Load existing session
        const sessions = getStoredSessions()
        const sessionData = sessions[existingSessionId]
        setSessionId(existingSessionId)
        setMessages(sessionData ? sessionData.messages : [])
        setTempMessages([]) // Clear temp messages
        console.log('📋 Loaded existing session:', existingSessionId, 'with', sessionData?.messages?.length || 0, 'messages')
        console.log('📋 Session data details:', sessionData);
      } else {
        // No existing session - start fresh
        console.log('🔍 No existing session found - starting in temporary mode')
        console.log('🔍 Searched for session with:', {
          namespace: activeProject.namespace,
          project: activeProject.project,
          chatService: chatService
        });
        setSessionId(null)
        setMessages([])
        setTempMessages([])
      }
    } else {
      // No active project
      console.log('🔍 No active project - clearing session state')
      setSessionId(null)
      setMessages([])
      setTempMessages([])
    }
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  // NEW: Function to add message to temporary state
  const addTempMessage = useCallback((message: ChatMessage) => {
    console.log('🔍 addTempMessage called with:', message);
    
    // Validate message before adding
    if (!message || !message.content || message.content.trim() === '') {
      console.warn('⚠️ Rejecting empty message in addTempMessage:', message);
      return;
    }
    
    console.log('✅ Adding valid temp message:', message.content.substring(0, 50) + (message.content.length > 50 ? '...' : ''));
    setTempMessages(prev => {
      const newMessages = [...prev, message];
      console.log('📝 Temp messages state after add:', newMessages);
      return newMessages;
    })
  }, [])
  
  // NEW: Function to add message to persistent state
  const addPersistentMessage = useCallback((message: ChatMessage) => {
    console.log('💾 addPersistentMessage called for session:', sessionId);
    
    // Validate message before adding
    if (!message || !message.content || message.content.trim() === '') {
      console.warn('⚠️ Rejecting empty message in addPersistentMessage:', message);
      return;
    }
    
    if (!sessionId) {
      console.error('❌ Cannot add persistent message - no session ID');
      return;
    }
    
    console.log('✅ Adding persistent message:', message.content.substring(0, 50) + (message.content.length > 50 ? '...' : ''));
    
    // Update local state
    setMessages(prev => [...prev, message])
    
    // Update localStorage using new helper function
    addMessageToPersistentSession(sessionId, message);
  }, [sessionId])
  
  // NEW: Transfer function to move temp messages to server session
  const transferToServerSession = useCallback((serverSessionId: string, tempMessages: ChatMessage[]) => {
    console.log('🔄 Transferring to server session:', serverSessionId, 'with', tempMessages.length, 'messages');
    console.log('🔒 Setting transfer lock to prevent session resets');
    transferLockRef.current = true;
    
    if (!activeProject) {
      console.error('❌ Cannot transfer - no active project');
      transferLockRef.current = false;
      return;
    }
    
    // Filter out placeholder "Thinking..." messages before transfer
    const messagesToTransfer = tempMessages.filter(msg => {
      const isThinkingPlaceholder = msg.role === 'assistant' && msg.content === 'Thinking...';
      if (isThinkingPlaceholder) {
        console.log('🚫 Filtering out placeholder message:', msg.content);
        return false;
      }
      return true;
    });
    
    console.log('📝 Filtered messages for transfer:', messagesToTransfer.length, 'out of', tempMessages.length);
    
    // Create persistent session with filtered messages
    createPersistentSession(
      serverSessionId,
      activeProject.namespace,
      activeProject.project,
      chatService,
      messagesToTransfer
    );
    
    // Update local state with filtered messages
    setSessionId(serverSessionId);
    setMessages(messagesToTransfer);
    setTempMessages([]); // Clear temp messages
    
    console.log('✅ Transfer complete - now in persistent mode');
    console.log('✅ State after transfer:', {
      newSessionId: serverSessionId,
      transferredMessagesCount: messagesToTransfer.length,
      filteredOut: tempMessages.length - messagesToTransfer.length,
      expectedPersistentMode: true
    });
    
    // Release lock after a brief delay to prevent immediate resets
    setTimeout(() => {
      transferLockRef.current = false;
      console.log('🔓 Released transfer lock - session lookup can resume');
    }, 100); // Reduced delay to fix timing issues
    
    // DEBUG: Check state after next render
    setTimeout(() => {
      console.log('📺 State check after transfer (next tick):', {
        sessionId: serverSessionId,
        isTemporaryMode: !serverSessionId,
        shouldShowPersistent: !!serverSessionId
      });
    }, 0);
  }, [activeProject, chatService])
  
  // Updated: Add message function that chooses temp vs persistent
  const addMessage = useCallback((content: string, role: 'user' | 'assistant'): ChatMessage => {
    console.log('🔍 addMessage called:', { content: `"${content}"`, role, contentLength: content.length, isTemporaryMode });
    
    // Validate content before creating message
    if (!content || content.trim() === '') {
      console.warn('⚠️ Attempted to add empty message, aborting');
      throw new Error('Cannot add message with empty content');
    }
    
    try {
      const message = createMessage(role, content)
      
      if (isTemporaryMode) {
        console.log('📝 Adding to temporary state:', content.substring(0, 50) + (content.length > 50 ? '...' : ''))
        addTempMessage(message)
      } else {
        console.log('💾 Adding to persistent state:', content.substring(0, 50) + (content.length > 50 ? '...' : ''))
        addPersistentMessage(message)
      }
      
      return message
    } catch (error) {
      console.error('❌ Error creating/adding message:', error);
      throw error;
    }
  }, [isTemporaryMode, addTempMessage, addPersistentMessage])
  
  // Clear history - clears both temp and persistent
  const clearHistory = useCallback(() => {
    if (shouldPreventReset('clearHistory')) {
      return;
    }
    
    console.log('Clearing history - temp mode:', isTemporaryMode)
    if (isTemporaryMode) {
      setTempMessages([])
    } else {
      setMessages([])
      // Also clear from localStorage
      if (sessionId) {
        const sessions = getStoredSessions()
        if (sessions[sessionId]) {
          sessions[sessionId].messages = []
          sessions[sessionId].lastUsed = new Date().toISOString()
          saveStoredSessions(sessions)
        }
      }
    }
  }, [isTemporaryMode, sessionId, shouldPreventReset])
  
  // Delete current session
  const deleteCurrentSession = useCallback(() => {
    if (shouldPreventReset('deleteCurrentSession')) {
      return;
    }
    
    console.log('Deleting current session')
    if (sessionId) {
      // Remove from localStorage
      const sessions = getStoredSessions()
      delete sessions[sessionId]
      saveStoredSessions(sessions)
    }
    
    setSessionId(null)
    setMessages([])
    setTempMessages([])
  }, [sessionId, shouldPreventReset])
  
  // Refresh session (reload from storage)
  const refreshSession = useCallback(() => {
    console.log('Refreshing session')
    if (sessionId) {
      const sessions = getStoredSessions()
      const sessionData = sessions[sessionId]
      if (sessionData) {
        setMessages(sessionData.messages)
        setError(null)
      }
    }
  }, [sessionId])
  
  // Reconcile with server - transition from temp to persistent
  const reconcileWithServer = useCallback((clientSessionId: string, serverSessionId: string) => {
    console.log('🔄 Reconcile check:', { 
      clientSessionId, 
      serverSessionId, 
      areEqual: clientSessionId === serverSessionId,
      clientType: typeof clientSessionId,
      serverType: typeof serverSessionId
    });
    
    // CRITICAL FIX: If session IDs are the same, no reconciliation needed
    if (clientSessionId === serverSessionId && 
        typeof clientSessionId === 'string' && 
        typeof serverSessionId === 'string' &&
        clientSessionId.length > 0) {
      console.log('✅ Session IDs are identical, no reconciliation needed - skipping');
      return; // Exit early - don't touch existing session!
    }
    
    console.log('🔄 Session IDs differ, proceeding with reconciliation');
    console.log('🔍 Reconcile temp messages array:', tempMessages);  // Debug the actual array
    console.log('🔍 Reconcile persistent messages array:', messages);  // Debug the actual array
    
    // Create persistent session with temp messages
    if (activeProject && tempMessages.length > 0) {
      console.log('🔄 Reconciling with temp messages to transfer');
      createPersistentSession(
        serverSessionId,
        activeProject.namespace,
        activeProject.project,
        chatService,
        tempMessages
      )
    } else if (activeProject) {
      console.log('🔄 Reconciling with empty session');
      createPersistentSession(
        serverSessionId,
        activeProject.namespace,
        activeProject.project,
        chatService,
        []
      )
    }
    
    // Transition to persistent mode
    setSessionId(serverSessionId)
    setMessages(tempMessages)
    setTempMessages([])
    setError(null)
  }, [tempMessages, messages, activeProject, chatService])
  
  // Create session from server
  const createSessionFromServerCallback = useCallback((serverSessionId: string) => {
    console.log('🆕 Creating session from server:', serverSessionId);
    console.log('🆕 Active project:', activeProject);
    
    // FIX: Use ref.current to get latest state instead of stale closure
    const currentTempMessages = tempMessagesRef.current;
    const currentMessages = messages;
    const currentIsTemporaryMode = !sessionId; // Calculate current mode
    
    console.log('🔍 Temp messages array (from ref):', currentTempMessages);
    console.log('🔍 Current temp messages (from ref):', currentTempMessages.length);
    console.log('🔍 Persistent messages array:', currentMessages);
    console.log('🔍 Current persistent messages:', currentMessages.length);
    console.log('🔍 Current temporary mode:', currentIsTemporaryMode);
    
    if (!activeProject) {
      console.error('❌ Cannot create session - no active project');
      return;
    }
    
    if (currentIsTemporaryMode && currentTempMessages.length > 0) {
      // We have temp messages to transfer
      console.log('🔄 Transferring temp messages to server session');
      transferToServerSession(serverSessionId, currentTempMessages);
    } else if (!currentIsTemporaryMode && currentMessages.length > 0) {
      // We already have persistent messages, just update the session ID and save
      console.log('🔄 Updating existing persistent session');
      createPersistentSession(
        serverSessionId,
        activeProject.namespace,
        activeProject.project,
        chatService,
        currentMessages
      );
      setSessionId(serverSessionId);
    } else {
      // No messages to transfer, just create empty session
      console.log('🆕 Creating empty server session');
      createPersistentSession(
        serverSessionId,
        activeProject.namespace,
        activeProject.project,
        chatService,
        []
      );
      setSessionId(serverSessionId);
      setTempMessages([]);
    }
    
    setError(null);
  }, [activeProject, chatService, sessionId, messages, transferToServerSession]) // Removed tempMessages and isTemporaryMode from dependencies
  
  // Expose debug function globally for testing
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).debugProjectSession = debugState;
      (window as any).debugCurrentProject = () => {
        console.log('=== CURRENT PROJECT DEBUG ===');
        console.log('Active project:', activeProject);
        console.log('Chat service:', chatService);
        console.log('Current session ID:', sessionId);
        console.log('Is temporary mode:', isTemporaryMode);
        console.log('Messages count:', messages.length);
        console.log('Temp messages count:', tempMessages.length);
        console.log('Display messages count:', displayMessages.length);
        console.log('=== END DEBUG ===');
      };
    }
  }, [debugState, activeProject, chatService, sessionId, isTemporaryMode, messages.length, tempMessages.length, displayMessages.length]);

  // DEBUG: Log when messages are returned to consuming components
  useEffect(() => {
    console.log('📤 ProjectSession: Returning messages to components');
    console.log('📤 ProjectSession: Returning', displayMessages.length, 'messages');
    console.log('📤 ProjectSession: Messages array:', displayMessages);
  }, [displayMessages])

  return {
    // State
    sessionId,
    messages: displayMessages,
    error,
    isTemporaryMode,
    tempMessages,
    
    // Actions
    addMessage,
    addTempMessage,
    addPersistentMessage,
    clearHistory,
    deleteCurrentSession,
    refreshSession,
    createSessionFromServer: createSessionFromServerCallback,
    reconcileWithServer,
    
    // Debug
    debugState
  }
}

/**
 * Hook for getting sessions for the current project context
 * Phase 1: Simplified stub version - functionality temporarily disabled
 */
export function useProjectSessions(chatService?: 'designer' | 'project') {
  const [sessions, setSessions] = useState<Array<{sessionId: string, metadata: any}>>([])
  const [isLoading, setIsLoading] = useState(false)
  
  const refreshSessions = useCallback(() => {
    console.log('Phase 1 stub: refreshSessions called', { chatService })
    setSessions([])
    setIsLoading(false)
  }, [chatService])
  
  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])
  
  return {
    sessions,
    isLoading,
    refreshSessions,
  }
}

export default useProjectSession