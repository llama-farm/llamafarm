/**
 * Enhanced TestChat component with Project Session Management
 * 
 * Integrates with the project session manager for Project Chat service
 * while maintaining backward compatibility with existing functionality
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import FontIcon from '../../common/FontIcon'
import { ChatboxMessage } from '../../types/chatbox'
import { Badge } from '../ui/badge'
import { useActiveProject } from '../../hooks/useActiveProject'
import { 
  useProjectChatMessage,
  useProjectChatParams 
} from '../../hooks/useProjectChat'
import { useProjectChatSession } from '../../hooks/useProjectChatSession'
import { useProjectSession } from '../../hooks/useProjectSession'
import { addMessageToHistory } from '../../utils/projectSessionManager'
// Import TestChatMessage from TestChat since it's defined there
import { TestChatMessage } from './TestChat'

export interface TestChatWithProjectSessionProps {
  showReferences: boolean
  allowRanking: boolean
  useTestData?: boolean
  showPrompts?: boolean
  showThinking?: boolean
  showGenSettings?: boolean
}

const containerClasses =
  // Match page background with clear outlines
  'w-full h-full flex flex-col rounded-xl border border-border bg-background text-foreground'

const inputContainerClasses = 
  'flex flex-col gap-2 p-3 md:p-4 border-t bg-muted/10'

const textareaClasses = 
  'w-full resize-none bg-background border border-input rounded-md px-3 py-2 ' +
  'text-sm text-foreground placeholder:text-muted-foreground ' +
  'focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent ' +
  'disabled:cursor-not-allowed disabled:opacity-50'

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center space-y-4 text-muted-foreground">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
        <FontIcon type="test" className="w-6 h-6" />
      </div>
      <div>
        <h3 className="font-medium text-foreground mb-1">Start a conversation</h3>
        <p className="text-sm">
          Ask questions about your project data, configurations, or anything else.
        </p>
        <div className="text-xs mt-2 opacity-75">
          Tip: Press Enter to send
        </div>
      </div>
    </div>
  )
}

export default function TestChatWithProjectSession({
  showReferences,
  allowRanking,
  useTestData,
  showPrompts,
  showThinking,
  showGenSettings,
}: TestChatWithProjectSessionProps) {
  // Get active project for project chat API
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)
  
  // Project session management for Project Chat
  const projectSession = useProjectSession({
    chatService: 'project',
    autoCreate: false, // Sessions created on first message
  })
  
  // Legacy project chat session management (for existing API integration)
  const projectChatSession = useProjectChatSession(
    chatParams?.namespace,
    chatParams?.projectId
  )
  
  // Project chat message sending
  const projectChatMessage = useProjectChatMessage()


  // Mock mode controlled by parent
  const MOCK_MODE = Boolean(useTestData)
  
  // Use project chat if we have an active project and not in mock mode
  const USE_PROJECT_CHAT = !MOCK_MODE && !!chatParams
  const USE_PROJECT_SESSION = USE_PROJECT_CHAT && !!activeProject
  
  // Convert project session messages to chatbox message format
  const projectSessionMessages: ChatboxMessage[] = projectSession.messages.map(msg => ({
    id: msg.id,
    type: msg.role === 'user' ? 'user' : 'assistant',
    content: msg.content,
    timestamp: new Date(msg.timestamp),
  }))
  
  // Use project session messages
  const messages = USE_PROJECT_SESSION ? projectSessionMessages : []
  
  // UI state
  const [inputValue, setInputValue] = useState('')
  
  // Combined loading state
  const isProjectChatLoading = projectChatMessage.isPending
  const isProjectSessionLoading = projectSession.isLoading
  const combinedIsSending = isProjectChatLoading || isProjectSessionLoading
  
  // Combined error state
  const projectChatError = projectChatMessage.error || projectChatSession.error
  const projectSessionError = projectSession.error
  const combinedError = (projectChatError ? projectChatError.message : null) || projectSessionError
  
  // Combined canSend state
  const combinedCanSend = inputValue.trim().length > 0 && 
    !combinedIsSending && 
    (!USE_PROJECT_CHAT || !!chatParams)

  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const lastUserInputRef = useRef<string>('')

  // Auto-grow textarea up to a comfortable max height before scrolling
  const resizeTextarea = useCallback(() => {
    const el = inputRef.current
    if (!el) return
    const maxHeight = 220 // ~6 lines depending on line-height
    el.style.height = 'auto'
    const newHeight = Math.min(el.scrollHeight, maxHeight)
    el.style.height = `${newHeight}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [])

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    } else if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages])

  // Resize textarea on mount and input changes
  useEffect(() => {
    resizeTextarea()
  }, [inputValue, resizeTextarea])

  // Update input value
  const updateInput = useCallback((value: string) => {
    setInputValue(value)
  }, [])

  // Handle sending message
  const handleSend = useCallback(async () => {
    const messageContent = inputValue.trim()
    if (!combinedCanSend || !messageContent) return

    // Store user input for reference
    lastUserInputRef.current = messageContent

    if (USE_PROJECT_SESSION) {
      // Use project session system for Project Chat
      try {
        setInputValue('')

        // Send to project chat API
        if (chatParams) {
          const response = await projectChatMessage.mutateAsync({
            namespace: chatParams.namespace,
            projectId: chatParams.projectId,
            message: messageContent,
            sessionId: projectSession.sessionId || undefined,
          })

          // Handle session creation if we got a new session ID from server
          if (response.sessionId && !projectSession.sessionId) {
            try {
              projectSession.createSessionFromServer(response.sessionId)
            } catch (sessionError) {
              console.error('Failed to create session from server response:', sessionError)
              // Don't fail the whole request for session management errors
            }
          }

          // Add messages to project session (now that we have a session)
          const activeSessionId = response.sessionId || projectSession.sessionId
          if (activeSessionId) {
            // Add messages directly to storage and update hook state
            addMessageToHistory(activeSessionId, {
              role: 'user',
              content: messageContent,
              timestamp: new Date().toISOString(),
            })
            
            if (response.completion.choices && response.completion.choices.length > 0) {
              const assistantResponse = response.completion.choices[0].message.content
              addMessageToHistory(activeSessionId, {
                role: 'assistant',
                content: assistantResponse,
                timestamp: new Date().toISOString(),
              })
            }
            
            // Refresh the hook state to reflect the new messages
            projectSession.refreshSession()
          }
        }
      } catch (error) {
        console.error('Project chat error:', error)
        // Add error message to project session if we have one
        if (projectSession.sessionId) {
          addMessageToHistory(projectSession.sessionId, {
            role: 'assistant',
            content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
            timestamp: new Date().toISOString(),
          })
          projectSession.refreshSession()
        }
      }
    } else {
      // No project session - show error  
      console.error('TestChatWithProjectSession requires a project to be selected')
    }
  }, [
    inputValue,
    combinedCanSend,
    USE_PROJECT_SESSION,
    projectSession,
    chatParams,
    projectChatMessage,
  ])

  // Handle clear chat
  const handleClear = useCallback(async () => {
    if (USE_PROJECT_SESSION) {
      try {
        projectSession.clearHistory()
      } catch (error) {
        console.error('Failed to clear project session:', error)
      }
    }
    // No fallback needed - TestChatWithProjectSession only supports project sessions
  }, [USE_PROJECT_SESSION, projectSession])

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Show loading indicator if project session is loading
  if (USE_PROJECT_SESSION && isProjectSessionLoading) {
    return (
      <div className={containerClasses}>
        <div className="flex items-center justify-center h-full">
          <div className="text-muted-foreground">Loading session...</div>
        </div>
      </div>
    )
  }

  return (
    <div className={containerClasses}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 md:p-4 border-b bg-muted/5">
        <div className="flex items-center gap-2">
          <FontIcon type="test" className="w-5 h-5 text-primary" />
          <span className="font-medium">
            {USE_PROJECT_CHAT ? 'Project Chat' : 'Test Chat'}
          </span>
          {USE_PROJECT_SESSION && projectSession.sessionId && (
            <Badge variant="secondary" className="text-xs">
              Session Active
            </Badge>
          )}
        </div>
        
        {messages.length > 0 && (
          <FontIcon
            isButton
            type="trashcan"
            className="w-4 h-4 text-muted-foreground hover:text-destructive"
            handleOnClick={handleClear}
          />
        )}
      </div>

      {/* Error Display */}
      {combinedError && (
        <div className="mx-3 md:mx-4 mt-3 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <div className="flex items-center gap-2 text-destructive text-sm">
            <FontIcon type="close" className="w-4 h-4" />
            <span>{combinedError}</span>
          </div>
        </div>
      )}

      {/* Messages */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-3 md:p-4">
        <div className="flex flex-col gap-4 h-full">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            messages.map((m: ChatboxMessage) => (
              <TestChatMessage
                key={m.id}
                message={m}
                showReferences={showReferences}
                allowRanking={allowRanking}
                showPrompts={showPrompts}
                showThinking={showThinking}
                lastUserInput={lastUserInputRef.current}
                showGenSettings={showGenSettings}
              />
            ))
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className={inputContainerClasses}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={e => updateInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={combinedIsSending || (!MOCK_MODE && !chatParams)}
          placeholder={
            combinedIsSending
              ? 'Waiting for response…'
              : !MOCK_MODE && !chatParams
                ? 'Select a project to start chatting…'
                : 'Type a message and press Enter'
          }
          className={textareaClasses}
          aria-label="Message input"
        />
        <div className="flex items-center justify-between">
          {combinedIsSending && (
            <span className="text-xs text-muted-foreground">
              {USE_PROJECT_CHAT ? 'Sending to project…' : 'Sending…'}
            </span>
          )}
          <FontIcon
            isButton
            type="arrow-filled"
            className={`w-8 h-8 self-end ${!combinedCanSend || (!MOCK_MODE && !chatParams) ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
            handleOnClick={handleSend}
          />
        </div>
      </div>
    </div>
  )
}