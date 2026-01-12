import { X } from 'lucide-react'
import type { CapturedRequest } from '../../contexts/DevToolsContext'
import DevToolsRequestTab from './DevToolsRequestTab'
import DevToolsResponseTab from './DevToolsResponseTab'
import DevToolsCodeSnippets from './DevToolsCodeSnippets'
import { cn } from '@/lib/utils'

type ActiveTab = 'request' | 'response' | 'code'

interface DevToolsExpandedPanelProps {
  request: CapturedRequest | null
  activeTab: ActiveTab
  onTabChange: (tab: ActiveTab) => void
  onClose: () => void
}

const tabs: { id: ActiveTab; label: string }[] = [
  { id: 'request', label: 'Request' },
  { id: 'response', label: 'Response' },
  { id: 'code', label: 'Code' },
]

const methodColors: Record<string, string> = {
  GET: 'bg-blue-500/20 text-blue-400',
  POST: 'bg-green-500/20 text-green-400',
  PUT: 'bg-yellow-500/20 text-yellow-400',
  PATCH: 'bg-orange-500/20 text-orange-400',
  DELETE: 'bg-red-500/20 text-red-400',
}

function StatusBadge({ status }: { status?: number }) {
  if (!status) return null

  const isSuccess = status >= 200 && status < 300
  return (
    <span
      className={cn(
        'px-1.5 py-0.5 rounded text-xs font-medium',
        isSuccess
          ? 'bg-emerald-500/20 text-emerald-400'
          : 'bg-red-500/20 text-red-400'
      )}
    >
      {status}
    </span>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
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

export default function DevToolsExpandedPanel({
  request,
  activeTab,
  onTabChange,
  onClose,
}: DevToolsExpandedPanelProps) {
  const formatTimestamp = (ts: number) => {
    return new Date(ts).toLocaleTimeString()
  }

  return (
    <div className="bg-card border-t border-border flex flex-col max-h-[70vh] rounded-t-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-foreground">Dev Tools</span>
          {request && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span
                className={cn(
                  'px-1.5 py-0.5 rounded font-mono font-medium',
                  methodColors[request.method] || 'bg-muted'
                )}
              >
                {request.method}
              </span>
              <span className="font-mono truncate max-w-[300px]">{request.url}</span>
              <StatusBadge status={request.status} />
              {request.duration !== undefined && (
                <span>{request.duration}ms</span>
              )}
              {request.isStreaming && !request.streamComplete && (
                <span className="inline-flex items-center gap-1 text-teal-400">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-teal-500" />
                  streaming
                </span>
              )}
            </div>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-muted rounded transition-colors"
          aria-label="Close Dev Tools"
        >
          <X className="w-4 h-4 text-muted-foreground" />
        </button>
      </div>

      {request ? (
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
            {activeTab === 'request' && <DevToolsRequestTab request={request} />}
            {activeTab === 'response' && <DevToolsResponseTab request={request} />}
            {activeTab === 'code' && <DevToolsCodeSnippets request={request} />}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-border flex items-center justify-between text-xs text-muted-foreground shrink-0">
            <span>
              Request ID:{' '}
              <span className="font-mono text-foreground/70">
                {request.requestId || request.id}
              </span>
            </span>
            <span>{formatTimestamp(request.timestamp)}</span>
          </div>
        </>
      ) : (
        <EmptyState />
      )}
    </div>
  )
}
