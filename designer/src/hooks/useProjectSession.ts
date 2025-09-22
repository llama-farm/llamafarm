/**
 * React hook for project session management
 * 
 * Manages session state for both Designer Chat and Project Chat services
 * with localStorage persistence and project context switching
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useActiveProject } from './useActiveProject'
import {
  getExistingSession,
  findSessionForContext,
  createSessionFromServer,
  createOptimisticSession,
  reconcileSessionWithServer,
  loadChatHistory,
  saveChatHistory,
  addMessageToHistory,
  clearChatHistory,
  deleteSession,
  getSessionsForContext,
  cleanupPendingSessions,
  loadSession,
  type ChatMessage,
  type SessionMetadata,
} from '../utils/projectSessionManager'

export interface ProjectSessionOptions {
  chatService: 'designer' | 'project'
  autoCreate?: boolean
}

export interface ProjectSessionState {
  sessionId: string | null
  messages: ChatMessage[]
  error: string | null
}

export interface ProjectSessionActions {
  addMessage: (content: string, role: 'user' | 'assistant') => ChatMessage
  clearHistory: () => void
  deleteCurrentSession: () => void
  refreshSession: () => void
  createSessionFromServer: (serverSessionId: string) => void
  reconcileWithServer: (clientSessionId: string, serverSessionId: string) => void
}

/**
 * Hook for managing project sessions with project context integration
 */
export function useProjectSession(
  options: ProjectSessionOptions
): ProjectSessionState & ProjectSessionActions {
  const { chatService, autoCreate = true } = options
  const activeProject = useActiveProject()
  
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [error, setError] = useState<string | null>(null)
  
  // Memoize project key to avoid unnecessary effects
  const projectKey = useMemo(() => {
    if (!activeProject) return null
    return `${activeProject.namespace}/${activeProject.project}/${chatService}`
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  // Memoize current session ID with project context validation
  const currentSessionId = useMemo(() => {
    if (!activeProject) {
      return null;
    }

    // If we have a sessionId, validate it belongs to the current project context
    if (sessionId) {
      // Load the session metadata to check if it matches current project
      const sessionMetadata = loadSession(sessionId);
      if (sessionMetadata) {
        const isContextMatch = 
          sessionMetadata.namespace === activeProject.namespace &&
          sessionMetadata.project === activeProject.project &&
          sessionMetadata.chatService === chatService;
        
        if (isContextMatch) {
          return sessionId;
        }
        // Fall through to look for a proper session for this context
      }
      // Fall through to look for a proper session for this context
    }

    // Look for existing session for current project context
    const existingSessionId = findSessionForContext(
      activeProject.namespace,
      activeProject.project,
      chatService
    );

    return existingSessionId;
  }, [sessionId, activeProject, chatService]);
  
  // Cleanup pending sessions on mount
  useEffect(() => {
    // Only cleanup very old sessions (24 hours) and only run occasionally
    const lastCleanup = localStorage.getItem('lastSessionCleanup');
    const now = Date.now();
    const oneDayMs = 24 * 60 * 60 * 1000;
    
    if (!lastCleanup || (now - parseInt(lastCleanup)) > oneDayMs) {
      cleanupPendingSessions(24 * 60); // 24 hours instead of 2 hours
      localStorage.setItem('lastSessionCleanup', now.toString());
    }
  }, [])
  
  
  // Cross-tab session synchronization
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'llamafarm_project_sessions' || event.key === 'llamafarm_project_chat_history') {
        if (currentSessionId) {
          const updatedHistory = loadChatHistory(currentSessionId);
          setMessages(updatedHistory);
        }
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [currentSessionId])

  // Reset session when active project changes
  useEffect(() => {
    // Always reset session ID and messages when project changes
    // This ensures we don't carry over sessions from other projects
    setSessionId(null);
    setMessages([]);
  }, [activeProject?.namespace, activeProject?.project, chatService]);

  // Load existing session when session ID changes
  useEffect(() => {
    if (!currentSessionId) {
      if (!activeProject) {
        setSessionId(null);
        setMessages([]);
        return;
      }

      // Double-check that no session exists before concluding there's none
      const doubleCheckSessionId = findSessionForContext(
        activeProject.namespace,
        activeProject.project,
        chatService
      );
      
      if (doubleCheckSessionId) {
        setSessionId(doubleCheckSessionId);
        const history = loadChatHistory(doubleCheckSessionId);
        setMessages(history);
        return;
      }

      // Truly no existing session
      setSessionId(null);
      setMessages([]);
    } else {
      // Load existing session
      setSessionId(currentSessionId);
      
      const history = loadChatHistory(currentSessionId);
      setMessages(history);
    }
  }, [currentSessionId, activeProject, chatService])
  
  // Add message to history
  const addMessage = useCallback((content: string, role: 'user' | 'assistant'): ChatMessage => {
    let currentSessionId = sessionId
    
    if (!currentSessionId) {
      if (!activeProject) {
        throw new Error('No active project and no session')
      }
      
      // Look for existing session
      const existingSessionId = getExistingSession(
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      
      if (existingSessionId) {
        currentSessionId = existingSessionId
        setSessionId(currentSessionId)
        const history = loadChatHistory(currentSessionId)
        setMessages(history)
      } else {
        // Create new optimistic session
        currentSessionId = createOptimisticSession(
          activeProject.namespace,
          activeProject.project,
          chatService
        )
        
        setSessionId(currentSessionId)
        setMessages([])
      }
    }
    
    // Add message to session
    const message = addMessageToHistory(currentSessionId, {
      role,
      content,
      timestamp: new Date().toISOString(),
    })
    
    setMessages(prev => [...prev, message]);
    return message
  }, [sessionId, activeProject, chatService])
  
  // Clear chat history
  const clearHistory = useCallback(() => {
    if (!sessionId) return
    
    try {
      clearChatHistory(sessionId)
      setMessages([])
    } catch (err) {
      console.error('Failed to clear history:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to clear history'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId])
  
  // Delete current session
  const deleteCurrentSession = useCallback(() => {
    if (!sessionId) return
    
    try {
      deleteSession(sessionId)
      setSessionId(null)
      setMessages([])
      
      // Note: New sessions will be created on next message send
    } catch (err) {
      console.error('Failed to delete session:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete session'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId, autoCreate, projectKey])
  
  // Refresh session (reload from storage)
  const refreshSession = useCallback(() => {
    if (!sessionId) return
    
    try {
      const history = loadChatHistory(sessionId)
      setMessages(history)
      setError(null)
    } catch (err) {
      console.error('Failed to refresh session:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to refresh session'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId])
  
  // Reconcile client session with server session ID
  const reconcileWithServer = useCallback((clientSessionId: string, serverSessionId: string) => {
    if (!sessionId || sessionId !== clientSessionId) {
      return;
    }
    
    try {
      // Store current messages before reconciliation
      const currentMessages = messages;
      
      // Perform reconciliation
      const finalSessionId = reconcileSessionWithServer(clientSessionId, serverSessionId);
      
      // Update the session ID
      setSessionId(finalSessionId);
      
      // Reload messages from the reconciled session
      const reconciledMessages = loadChatHistory(finalSessionId);
      
      // If no messages were found after reconciliation, but we had messages before,
      // something went wrong during migration - restore the messages
      if (reconciledMessages.length === 0 && currentMessages.length > 0) {
        // Save current messages to the new session ID
        saveChatHistory(finalSessionId, currentMessages);
        setMessages(currentMessages);
      } else {
        setMessages(reconciledMessages);
      }
      
      setError(null);
      
    } catch (error) {
      console.error('Error during session reconciliation:', error);
      // Keep the current session and messages on error
      const errorMessage = error instanceof Error ? error.message : 'Failed to reconcile session';
      setError(errorMessage);
    }
  }, [sessionId, messages])
  
  // Create session with server-provided session ID
  const createSessionFromServerCallback = useCallback((serverSessionId: string) => {
    if (!activeProject) {
      throw new Error('No active project')
    }
    
    try {
      // Check if we have an existing session even if sessionId is null
      let existingSessionId = sessionId
      
      if (!existingSessionId) {
        // Look for existing session for this project
        existingSessionId = getExistingSession(
          activeProject.namespace,
          activeProject.project,
          chatService
        )
      }
      
      if (existingSessionId) {
        // Use the reconciliation function to handle existing session
        reconcileWithServer(existingSessionId, serverSessionId);
      } else {
        // Truly no existing session, create new one
        createSessionFromServer(
          serverSessionId,
          activeProject.namespace,
          activeProject.project,
          chatService
        )
        
        setSessionId(serverSessionId)
        setMessages([])
        setError(null)
      }
    } catch (err) {
      console.error('Failed to create session from server:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to create session from server'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [activeProject?.namespace, activeProject?.project, chatService, sessionId, reconcileWithServer])
  
  return {
    // State
    sessionId: currentSessionId,
    messages,
    error,
    
    // Actions
    addMessage,
    clearHistory,
    deleteCurrentSession,
    refreshSession,
    createSessionFromServer: createSessionFromServerCallback,
    reconcileWithServer,
  }
}

/**
 * Hook for getting sessions for the current project context
 */
export function useProjectSessions(chatService?: 'designer' | 'project') {
  const activeProject = useActiveProject()
  const [sessions, setSessions] = useState<Array<{sessionId: string, metadata: SessionMetadata}>>([])
  const [isLoading, setIsLoading] = useState(false)
  
  const refreshSessions = useCallback(() => {
    if (!activeProject) {
      setSessions([])
      return
    }
    
    setIsLoading(true)
    try {
      const projectSessions = getSessionsForContext(
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      setSessions(projectSessions)
    } catch (err) {
      console.error('Failed to load sessions:', err)
      throw new Error(err instanceof Error ? err.message : 'Failed to load sessions')
    } finally {
      setIsLoading(false)
    }
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  useEffect(() => {
    refreshSessions()
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  return {
    sessions,
    isLoading,
    refreshSessions,
  }
}

export default useProjectSession