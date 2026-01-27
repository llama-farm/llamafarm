import { useState, useRef, useEffect } from 'react'
import { X, Trash2, ArrowUp, ArrowDown } from 'lucide-react'
import type { CapturedRequest, CapturedWebSocket, CapturedWebSocketMessage } from '../../contexts/DevToolsContext'
import DevToolsRequestTab from './DevToolsRequestTab'
import DevToolsResponseTab from './DevToolsResponseTab'
import DevToolsCodeSnippets from './DevToolsCodeSnippets'
import { StatusBadge, MethodBadge } from './DevToolsShared'
import { cn } from '@/lib/utils'

type ActiveTab = 'request' | 'response' | 'code'
type ListMode = 'http' | 'websocket'

interface DevToolsExpandedPanelProps {
  requests: CapturedRequest[]
  selectedRequest: CapturedRequest | null
  activeTab: ActiveTab
  onTabChange: (tab: ActiveTab) => void
  onSelectRequest: (request: CapturedRequest) => void
  onClearHistory: () => void
  onClose: () => void
  webSockets: CapturedWebSocket[]
  selectedWebSocket: CapturedWebSocket | null
  onSelectWebSocket: (ws: CapturedWebSocket | null) => void
}

const tabs: { id: ActiveTab; label: string }[] = [
  { id: 'request', label: 'Request' },
  { id: 'response', label: 'Response' },
  { id: 'code', label: 'Code' },
]

function EmptyState({ mode }: { mode: ListMode }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center flex-1">
      <div className="text-muted-foreground mb-2">
        <svg
          className="w-12 h-12 mx-auto mb-3 opacity-50"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      </div>
      <p className="text-sm text-muted-foreground">
        {mode === 'http' ? 'No API requests captured yet' : 'No WebSocket connections yet'}
      </p>
      <p className="text-xs text-muted-foreground/70 mt-1">
        {mode === 'http'
          ? 'Send a message to see request details here'
          : 'Start a voice chat to see WebSocket messages'}
      </p>
    </div>
  )
}

function RequestListItem({
  request,
  isSelected,
  onClick,
}: {
  request: CapturedRequest
  isSelected: boolean
  onClick: () => void
}) {
  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  // Extract just the path from the URL
  const urlPath = request.url.replace(/^https?:\/\/[^/]+/, '')

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2 border-b border-border hover:bg-accent/50 transition-colors',
        isSelected && 'bg-accent'
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <MethodBadge method={request.method} />
        <StatusBadge status={request.status} />
        {request.isStreaming && !request.streamComplete && (
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-teal-500 shrink-0" />
        )}
        {request.duration !== undefined && (
          <span className="text-[10px] text-muted-foreground ml-auto shrink-0">
            {request.duration}ms
          </span>
        )}
      </div>
      <div className="text-xs font-mono text-muted-foreground truncate">
        {urlPath}
      </div>
      <div className="text-[10px] text-muted-foreground/60 mt-0.5">
        {formatTime(request.timestamp)}
      </div>
    </button>
  )
}

function WebSocketListItem({
  ws,
  isSelected,
  onClick,
}: {
  ws: CapturedWebSocket
  isSelected: boolean
  onClick: () => void
}) {
  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  // Extract path from WebSocket URL
  const urlPath = ws.url.replace(/^wss?:\/\/[^/]+/, '').split('?')[0]

  const statusColor = ws.status === 'open'
    ? 'bg-green-500'
    : ws.status === 'error'
      ? 'bg-red-500'
      : 'bg-gray-400'

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2 border-b border-border hover:bg-accent/50 transition-colors',
        isSelected && 'bg-accent'
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-600 dark:text-purple-400">
          WS
        </span>
        <span className={cn('h-2 w-2 rounded-full', statusColor)} title={ws.status} />
        {ws.status === 'open' && (
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green-500 shrink-0" />
        )}
        <span className="text-[10px] text-muted-foreground ml-auto shrink-0">
          {ws.messageCount} msgs
        </span>
      </div>
      <div className="text-xs font-mono text-muted-foreground truncate">
        {urlPath}
      </div>
      <div className="text-[10px] text-muted-foreground/60 mt-0.5">
        {formatTime(ws.openedAt)}
      </div>
    </button>
  )
}

function WebSocketMessageItem({ message }: { message: CapturedWebSocketMessage }) {
  const formatTime = (ts: number) => {
    const date = new Date(ts)
    const base = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    const ms = String(date.getMilliseconds()).padStart(3, '0')
    return `${base}.${ms}`
  }

  const isJson = !message.isBinary && typeof message.data === 'object'
  const displayData = isJson ? JSON.stringify(message.data, null, 2) : String(message.data)
  const messageType = isJson && message.data?.type ? message.data.type : null

  return (
    <div className={cn(
      'px-3 py-2 border-b border-border/50 text-xs',
      message.direction === 'send' ? 'bg-blue-500/5' : 'bg-green-500/5'
    )}>
      <div className="flex items-center gap-2 mb-1">
        {message.direction === 'send' ? (
          <ArrowUp className="w-3 h-3 text-blue-600 dark:text-blue-300" />
        ) : (
          <ArrowDown className="w-3 h-3 text-green-600 dark:text-green-400" />
        )}
        <span className={cn(
          'text-[10px] font-medium',
          message.direction === 'send' ? 'text-blue-600 dark:text-blue-300' : 'text-green-600 dark:text-green-400'
        )}>
          {message.direction === 'send' ? 'SEND' : 'RECV'}
        </span>
        {messageType && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
            {messageType}
          </span>
        )}
        {message.isBinary && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400">
            binary
          </span>
        )}
        <span className="text-[10px] text-muted-foreground ml-auto">
          {message.size} bytes
        </span>
        <span className="text-[10px] text-muted-foreground/60">
          {formatTime(message.timestamp)}
        </span>
      </div>
      <pre className="font-mono text-[11px] text-foreground/80 whitespace-pre-wrap break-all max-h-32 overflow-y-auto">
        {displayData}
      </pre>
    </div>
  )
}

function WebSocketDetail({ ws }: { ws: CapturedWebSocket }) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [ws.messages.length])

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString()
  }

  return (
    <div className="flex flex-col h-full">
      {/* WebSocket Info Header */}
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-medium px-2 py-1 rounded bg-purple-500/20 text-purple-600 dark:text-purple-400">
            WebSocket
          </span>
          <span className={cn(
            'text-xs px-2 py-1 rounded',
            ws.status === 'open' ? 'bg-green-500/20 text-green-600 dark:text-green-400' :
            ws.status === 'error' ? 'bg-red-500/20 text-red-600 dark:text-red-400' :
            'bg-gray-500/20 text-gray-600 dark:text-gray-400'
          )}>
            {ws.status}
          </span>
        </div>
        <div className="text-xs font-mono text-muted-foreground break-all">
          {ws.url}
        </div>
        <div className="flex items-center gap-4 mt-2 text-[10px] text-muted-foreground">
          <span>Opened: {formatTime(ws.openedAt)}</span>
          {ws.closedAt && <span>Closed: {formatTime(ws.closedAt)}</span>}
          <span>{ws.messageCount} total messages</span>
        </div>
        {ws.error && (
          <div className="mt-2 text-xs text-red-400">
            Error: {ws.error}
          </div>
        )}
      </div>

      {/* Messages List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {ws.messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            No messages yet
          </div>
        ) : (
          <>
            {ws.messages.map((msg) => (
              <WebSocketMessageItem key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>
    </div>
  )
}

export default function DevToolsExpandedPanel({
  requests,
  selectedRequest,
  activeTab,
  onTabChange,
  onSelectRequest,
  onClearHistory,
  onClose,
  webSockets,
  selectedWebSocket,
  onSelectWebSocket,
}: DevToolsExpandedPanelProps) {
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [listMode, setListMode] = useState<ListMode>('http')
  const clearTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Cleanup timeout on unmount to prevent memory leak
  useEffect(() => {
    return () => {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }
    }
  }, [])

  const formatTimestamp = (ts: number) => {
    return new Date(ts).toLocaleTimeString()
  }

  const handleClearClick = () => {
    if (showClearConfirm) {
      onClearHistory()
      setShowClearConfirm(false)
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
        clearTimeoutRef.current = null
      }
    } else {
      setShowClearConfirm(true)
      // Auto-hide confirmation after 3 seconds
      clearTimeoutRef.current = setTimeout(() => setShowClearConfirm(false), 3000)
    }
  }

  // Default to the latest request (first in array) if none selected
  const displayedRequest = selectedRequest ?? requests[0] ?? null
  const displayedWebSocket = selectedWebSocket ?? webSockets[0] ?? null

  const totalCount = requests.length + webSockets.length
  const isEmpty = listMode === 'http' ? requests.length === 0 : webSockets.length === 0

  return (
    <div className="bg-card border-t border-border flex flex-col h-[60vh] rounded-t-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-foreground">Dev Tools</span>
          {/* Mode Toggle */}
          <div className="flex items-center gap-1 bg-muted/50 rounded p-0.5">
            <button
              onClick={() => setListMode('http')}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors',
                listMode === 'http'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              HTTP ({requests.length})
            </button>
            <button
              onClick={() => setListMode('websocket')}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors',
                listMode === 'websocket'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              WS ({webSockets.length})
            </button>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {totalCount > 0 && (
            <button
              onClick={handleClearClick}
              className={cn(
                'flex items-center gap-1 px-1.5 py-1 rounded transition-colors text-xs',
                showClearConfirm
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                  : 'hover:bg-muted text-muted-foreground'
              )}
              aria-label={showClearConfirm ? 'Confirm clear' : 'Clear history'}
              title={showClearConfirm ? 'Click again to confirm' : 'Clear history'}
            >
              <Trash2 className="w-4 h-4" />
              {showClearConfirm && <span>Clear?</span>}
            </button>
          )}
          <button
            onClick={onClose}
            className="p-1 hover:bg-muted rounded transition-colors"
            aria-label="Close Dev Tools"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      </div>

      {isEmpty ? (
        <EmptyState mode={listMode} />
      ) : (
        <div className="flex flex-1 min-h-0">
          {/* List - Left Panel */}
          <div className="w-72 shrink-0 border-r border-border overflow-y-auto scrollbar-thin">
            {listMode === 'http' ? (
              requests.map(req => (
                <RequestListItem
                  key={req.id}
                  request={req}
                  isSelected={displayedRequest?.id === req.id}
                  onClick={() => onSelectRequest(req)}
                />
              ))
            ) : (
              webSockets.map(ws => (
                <WebSocketListItem
                  key={ws.id}
                  ws={ws}
                  isSelected={displayedWebSocket?.id === ws.id}
                  onClick={() => onSelectWebSocket(ws)}
                />
              ))
            )}
          </div>

          {/* Detail Panel - Right Side */}
          <div className="flex-1 flex flex-col min-w-0">
            {listMode === 'http' && displayedRequest && (
              <>
                {/* Tabs */}
                <div className="flex gap-1 px-4 pt-2 border-b border-border shrink-0">
                  {tabs.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => onTabChange(tab.id)}
                      className={cn(
                        'px-3 py-1.5 text-xs font-medium rounded-t transition-colors',
                        activeTab === tab.id
                          ? 'bg-muted text-foreground'
                          : 'text-muted-foreground hover:text-foreground'
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-hidden">
                  {activeTab === 'request' && <DevToolsRequestTab request={displayedRequest} />}
                  {activeTab === 'response' && <DevToolsResponseTab request={displayedRequest} />}
                  {activeTab === 'code' && <DevToolsCodeSnippets request={displayedRequest} />}
                </div>

                {/* Footer */}
                <div className="px-4 py-2 border-t border-border flex items-center justify-between text-xs text-muted-foreground shrink-0">
                  <span>
                    Request ID:{' '}
                    <span className="font-mono text-foreground/70">
                      {displayedRequest.requestId || displayedRequest.id}
                    </span>
                  </span>
                  <span>{formatTimestamp(displayedRequest.timestamp)}</span>
                </div>
              </>
            )}

            {listMode === 'websocket' && displayedWebSocket && (
              <WebSocketDetail ws={displayedWebSocket} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
