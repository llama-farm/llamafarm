/**
 * Example usage of the Project Chat API system
 * 
 * This file demonstrates how to use the new project chat service and hooks
 * that are completely separate from the existing chat system.
 * 
 * DO NOT USE THIS IN PRODUCTION - This is just an example/reference
 */

import React, { useState, useCallback } from 'react'
import { useActiveProject } from '../hooks/useActiveProject'
import { 
  useProjectChatMessage,
  useProjectChatStreamingMessage,
  useProjectChatParams
} from '../hooks/useProjectChat'
import { 
  useProjectChatSession
} from '../hooks/useProjectChatSession'
import {
  ProjectChatStreamChunk,
  ProjectChatMessage
} from '../api/projectChatService'

/**
 * Example component showing basic project chat usage
 */
export const BasicProjectChatExample: React.FC = () => {
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')

  // Use the simple message hook
  const chatMessage = useProjectChatMessage()

  const handleSendMessage = useCallback(async () => {
    if (!chatParams || !message.trim()) return

    try {
      const result = await chatMessage.mutateAsync({
        namespace: chatParams.namespace,
        projectId: chatParams.projectId,
        message: message.trim(),
      })

      const assistantMessage = result.completion.choices[0]?.message?.content || 'No response'
      setResponse(assistantMessage)
      setMessage('')
    } catch (error) {
      console.error('Failed to send message:', error)
      setResponse('Error: Failed to send message')
    }
  }, [chatParams, message, chatMessage])

  if (!activeProject) {
    return <div>No active project selected</div>
  }

  return (
    <div className="p-4 border rounded">
      <h3 className="text-lg font-semibold mb-4">Basic Project Chat Example</h3>
      <p className="text-sm text-gray-600 mb-4">
        Project: {activeProject.namespace}/{activeProject.project}
      </p>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Message:</label>
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            className="w-full p-2 border rounded"
            placeholder="Type your message..."
            disabled={chatMessage.isPending}
          />
        </div>
        
        <button
          onClick={handleSendMessage}
          disabled={chatMessage.isPending || !message.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          {chatMessage.isPending ? 'Sending...' : 'Send Message'}
        </button>
        
        {response && (
          <div className="p-3 bg-gray-100 rounded">
            <strong>Response:</strong> {response}
          </div>
        )}
        
        {chatMessage.error && (
          <div className="p-3 bg-red-100 text-red-700 rounded">
            Error: {chatMessage.error.message}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Example component showing streaming project chat usage
 */
export const StreamingProjectChatExample: React.FC = () => {
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)
  const [message, setMessage] = useState('')
  const [streamingResponse, setStreamingResponse] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

  const streamingMessage = useProjectChatStreamingMessage()

  const handleStreamMessage = useCallback(async () => {
    if (!chatParams || !message.trim()) return

    setIsStreaming(true)
    setStreamingResponse('')

    try {
      await streamingMessage.mutateAsync({
        namespace: chatParams.namespace,
        projectId: chatParams.projectId,
        message: message.trim(),
        streamingOptions: {
          onChunk: (chunk: ProjectChatStreamChunk) => {
            const content = chunk.choices[0]?.delta?.content || ''
            if (content) {
              setStreamingResponse(prev => prev + content)
            }
          },
          onComplete: () => {
            setIsStreaming(false)
          },
          onError: (error) => {
            console.error('Streaming error:', error)
            setIsStreaming(false)
          }
        }
      })

      setMessage('')
    } catch (error) {
      console.error('Failed to stream message:', error)
      setIsStreaming(false)
    }
  }, [chatParams, message, streamingMessage])

  if (!activeProject) {
    return <div>No active project selected</div>
  }

  return (
    <div className="p-4 border rounded">
      <h3 className="text-lg font-semibold mb-4">Streaming Project Chat Example</h3>
      <p className="text-sm text-gray-600 mb-4">
        Project: {activeProject.namespace}/{activeProject.project}
      </p>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Message:</label>
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            className="w-full p-2 border rounded"
            placeholder="Type your message..."
            disabled={isStreaming}
          />
        </div>
        
        <button
          onClick={handleStreamMessage}
          disabled={isStreaming || !message.trim()}
          className="px-4 py-2 bg-green-500 text-white rounded disabled:opacity-50"
        >
          {isStreaming ? 'Streaming...' : 'Stream Message'}
        </button>
        
        {streamingResponse && (
          <div className="p-3 bg-gray-100 rounded">
            <strong>Streaming Response:</strong>
            <div className="mt-2 whitespace-pre-wrap">{streamingResponse}</div>
            {isStreaming && <span className="animate-pulse">▊</span>}
          </div>
        )}
        
        {streamingMessage.error && (
          <div className="p-3 bg-red-100 text-red-700 rounded">
            Error: {streamingMessage.error.message}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Example component showing session management with server-managed sessions
 */
export const ProjectChatSessionExample: React.FC = () => {
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)
  
  // Session management with server-provided session IDs
  const {
    sessionId,
    isSessionActive,
    sendMessage,
    clearSession,
    isLoading,
    sessionError
  } = useProjectChatSession(chatParams?.namespace, chatParams?.projectId)

  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState<ProjectChatMessage[]>([])

  const handleSendWithSession = useCallback(async () => {
    if (!chatParams || !message.trim()) return

    const userMessage: ProjectChatMessage = {
      role: 'user',
      content: message.trim()
    }

    const updatedConversation = [...conversation, userMessage]
    setConversation(updatedConversation)

    try {
      // sendMessage will start new session if none exists, or continue existing session
      const result = await sendMessage(message.trim())
      
      if (result) {
        const assistantMessage: ProjectChatMessage = {
          role: 'assistant',
          content: result.completion.choices[0]?.message?.content || 'No response'
        }

        setConversation(prev => [...prev, assistantMessage])
      }
      
      setMessage('')
    } catch (error) {
      console.error('Failed to send message with session:', error)
    }
  }, [chatParams, message, conversation, sendMessage])

  if (!activeProject) {
    return <div>No active project selected</div>
  }

  return (
    <div className="p-4 border rounded">
      <h3 className="text-lg font-semibold mb-4">Project Chat Session Example</h3>
      <p className="text-sm text-gray-600 mb-4">
        Project: {activeProject.namespace}/{activeProject.project}
      </p>
      
      <div className="space-y-4">
        {/* Session Management */}
        <div className="p-3 bg-blue-50 rounded">
          <h4 className="font-medium mb-2">Session Management (Server-Managed)</h4>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm">
              Status: {isSessionActive ? 'Active' : 'No Session'}
            </span>
            {sessionId && (
              <span className="text-xs text-gray-500">
                ID: {sessionId.slice(-8)}
              </span>
            )}
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={clearSession}
              disabled={!isSessionActive}
              className="px-3 py-1 bg-red-500 text-white rounded text-sm disabled:opacity-50"
            >
              Clear Session
            </button>
          </div>
          
          {sessionError && (
            <div className="mt-2 text-sm text-red-600">
              Error: {sessionError.message}
            </div>
          )}
          
          <div className="mt-2 text-sm text-gray-600">
            Note: Sessions are automatically created by the server when you send your first message.
          </div>
        </div>

        {/* Chat Interface - Always available, session created automatically */}
        {(
          <>
            <div className="max-h-60 overflow-y-auto border rounded p-3">
              <h4 className="font-medium mb-2">Conversation</h4>
              {conversation.length === 0 ? (
                <p className="text-gray-500 text-sm">No messages yet</p>
              ) : (
                <div className="space-y-2">
                  {conversation.map((msg, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded text-sm ${
                        msg.role === 'user'
                          ? 'bg-blue-100 ml-4'
                          : 'bg-gray-100 mr-4'
                      }`}
                    >
                      <strong>{msg.role}:</strong> {msg.content}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Message:</label>
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="w-full p-2 border rounded"
                placeholder="Type your message..."
                disabled={isLoading}
                onKeyPress={(e) => e.key === 'Enter' && handleSendWithSession()}
              />
            </div>
            
            <button
              onClick={handleSendWithSession}
              disabled={isLoading || !message.trim()}
              className="px-4 py-2 bg-purple-500 text-white rounded disabled:opacity-50"
            >
              {isLoading ? 'Sending...' : 'Send Message'}
            </button>
            
            {sessionError && (
              <div className="p-3 bg-red-100 text-red-700 rounded">
                Error: {sessionError.message}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/**
 * Combined example component showing all features
 */
export const ProjectChatExamples: React.FC = () => {
  return (
    <div className="space-y-6 p-6">
      <h2 className="text-2xl font-bold mb-4">Project Chat API Examples</h2>
      <p className="text-gray-600 mb-6">
        These examples demonstrate the new project chat system that is completely 
        separate from the existing chat functionality.
      </p>
      
      <BasicProjectChatExample />
      <StreamingProjectChatExample />
      <ProjectChatSessionExample />
      
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">
        <h4 className="font-medium text-yellow-800 mb-2">Important Notes:</h4>
        <ul className="text-sm text-yellow-700 space-y-1">
          <li>• This is a completely separate chat system from the existing chatbox</li>
          <li>• All functions are prefixed with "project" to avoid naming conflicts</li>
          <li>• Sessions are managed by the server, not client-generated UUIDs</li>
          <li>• Server creates session IDs and returns them in response headers</li>
          <li>• Uses the same HTTP client configuration but different endpoints</li>
          <li>• Supports both streaming and non-streaming chat completions</li>
          <li>• First message starts new session, subsequent messages continue it</li>
        </ul>
      </div>
    </div>
  )
}

export default ProjectChatExamples
