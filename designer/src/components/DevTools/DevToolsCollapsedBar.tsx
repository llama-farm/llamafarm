import { ChevronUp } from 'lucide-react'
import type { CapturedRequest } from '../../contexts/DevToolsContext'
import { StatusBadge, MethodBadge } from './DevToolsShared'

interface DevToolsCollapsedBarProps {
  request: CapturedRequest | null
  onClick: () => void
}

export default function DevToolsCollapsedBar({
  request,
  onClick,
}: DevToolsCollapsedBarProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-2 bg-card border-t border-border hover:bg-accent/40 transition-colors cursor-pointer text-left rounded-t-lg"
    >
      {/* Dev Tools label with caret */}
      <div className="flex items-center gap-1.5 text-sm font-medium text-foreground">
        <span>Dev Tools</span>
        <ChevronUp className="w-4 h-4 text-muted-foreground" />
      </div>

      {/* Request info */}
      {request ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground flex-1 min-w-0">
          <MethodBadge method={request.method} />
          <span className="font-mono truncate">{request.url}</span>
          <StatusBadge status={request.status} />
          {request.duration !== undefined && (
            <span className="text-muted-foreground shrink-0">
              {request.duration}ms
            </span>
          )}
          {request.isStreaming && !request.streamComplete && (
            <span className="inline-flex items-center gap-1 text-teal-400 shrink-0">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-teal-500" />
              streaming
            </span>
          )}
        </div>
      ) : (
        <span className="text-xs text-muted-foreground">No requests yet</span>
      )}
    </button>
  )
}
