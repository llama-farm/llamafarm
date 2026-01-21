import { useState, useRef, useEffect } from 'react'
import { X, Trash2 } from 'lucide-react'
import type { CapturedRequest } from '../../contexts/DevToolsContext'
import DevToolsRequestTab from './DevToolsRequestTab'
import DevToolsResponseTab from './DevToolsResponseTab'
import DevToolsCodeSnippets from './DevToolsCodeSnippets'
import { StatusBadge, MethodBadge } from './DevToolsShared'
import { cn } from '@/lib/utils'

type ActiveTab = 'request' | 'response' | 'code'

interface DevToolsExpandedPanelProps {
  requests: CapturedRequest[]
  selectedRequest: CapturedRequest | null
  activeTab: ActiveTab
  onTabChange: (tab: ActiveTab) => void
  onSelectRequest: (request: CapturedRequest) => void
  onClearHistory: () => void
  onClose: () => void
}

const tabs: { id: ActiveTab; label: string }[] = [
  { id: 'request', label: 'Request' },
  { id: 'response', label: 'Response' },
  { id: 'code', label: 'Code' },
]

function EmptyState() {
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
      <p className="text-sm text-muted-foreground">No API requests captured yet</p>
      <p className="text-xs text-muted-foreground/70 mt-1">
        Send a message to see request details here
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

export default function DevToolsExpandedPanel({
  requests,
  selectedRequest,
  activeTab,
  onTabChange,
  onSelectRequest,
  onClearHistory,
  onClose,
}: DevToolsExpandedPanelProps) {
  const [showClearConfirm, setShowClearConfirm] = useState(false)
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

  return (
    <div className="bg-card border-t border-border flex flex-col h-[60vh] rounded-t-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-foreground">Dev Tools</span>
          <span className="text-xs text-muted-foreground">
            {requests.length} request{requests.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {requests.length > 0 && (
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

      {requests.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="flex flex-1 min-h-0">
          {/* Request List - Left Panel */}
          <div className="w-72 shrink-0 border-r border-border overflow-y-auto scrollbar-thin">
            {requests.map(req => (
              <RequestListItem
                key={req.id}
                request={req}
                isSelected={displayedRequest?.id === req.id}
                onClick={() => onSelectRequest(req)}
              />
            ))}
          </div>

          {/* Detail Panel - Right Side */}
          <div className="flex-1 flex flex-col min-w-0">
            {displayedRequest && (
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
          </div>
        </div>
      )}
    </div>
  )
}
