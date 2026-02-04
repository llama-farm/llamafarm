import { useState, useRef, useEffect } from 'react'
import {
  Activity,
  RefreshCw,
  X,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
} from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from './ui/dropdown-menu'
import { useServiceHealth, AggregateStatus, ServiceDisplay } from '../hooks/useServiceHealth'
import { cn } from '../lib/utils'

/**
 * Banner styles based on aggregate status
 */
const BANNER_STYLES: Record<NonNullable<AggregateStatus>, string> = {
  healthy: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  degraded: 'bg-amber-500/10 text-amber-600 dark:text-amber-400',
  unhealthy: 'bg-red-500/10 text-red-600 dark:text-red-400',
}

/**
 * Banner messages based on aggregate status
 */
const BANNER_MESSAGES: Record<NonNullable<AggregateStatus>, string> = {
  healthy: 'All services operational',
  degraded: 'Some services need attention',
  unhealthy: 'Service issues detected',
}

/**
 * Status icon component
 */
function StatusIcon({ status }: { status: ServiceDisplay['status'] }) {
  switch (status) {
    case 'healthy':
      return <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
    case 'degraded':
      return <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
    case 'unhealthy':
      return <XCircle className="w-4 h-4 text-red-500 shrink-0" />
  }
}

/**
 * Message length threshold for showing "show more" button
 */
const MESSAGE_TRUNCATE_LENGTH = 60

/**
 * Service row component
 */
function ServiceRow({ service }: { service: ServiceDisplay }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const shouldTruncate = service.message.length > MESSAGE_TRUNCATE_LENGTH

  return (
    <div className="flex items-start justify-between py-3 px-4">
      <div className="flex items-start gap-3 min-w-0 flex-1">
        <div className="mt-0.5">
          <StatusIcon status={service.status} />
        </div>
        <div className="flex flex-col min-w-0 flex-1">
          <span className="text-sm font-medium text-foreground">
            {service.displayName}
          </span>
          <span
            className={cn(
              'text-xs text-muted-foreground break-words',
              !isExpanded && shouldTruncate && 'line-clamp-2'
            )}
          >
            {service.message}
          </span>
          {shouldTruncate && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs text-primary hover:underline self-start mt-0.5"
            >
              {isExpanded ? 'Show less' : 'Show more'}
            </button>
          )}
          {service.host && (
            <span className="text-[10px] text-muted-foreground/60 font-mono break-all">
              {service.host}
            </span>
          )}
        </div>
      </div>
      {service.latencyMs !== undefined && (
        <span className="text-xs text-muted-foreground shrink-0 ml-2">
          {service.latencyMs}ms
        </span>
      )}
    </div>
  )
}

/**
 * Loading spinner component
 */
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-8">
      <div className="animate-spin rounded-full w-6 h-6 border-2 border-primary border-t-transparent" />
    </div>
  )
}

/**
 * Error state component
 */
function ErrorState({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="py-6 px-4 text-center">
      <XCircle className="w-8 h-8 text-destructive mx-auto mb-2" />
      <p className="text-sm text-muted-foreground mb-3">
        Unable to fetch service status
      </p>
      <button
        onClick={onRetry}
        className="text-sm text-primary hover:underline"
      >
        Retry
      </button>
    </div>
  )
}

/**
 * Service Status Panel component
 *
 * Displays health status of core LlamaFarm services in a dropdown panel.
 * Auto-refreshes every 30 seconds while open.
 */
// Minimum spin duration for visual feedback (Tailwind animate-spin is 1s)
const MIN_SPIN_DURATION_MS = 600

export function ServiceStatusPanel() {
  const [isOpen, setIsOpen] = useState(false)
  const [isSpinning, setIsSpinning] = useState(false)
  const spinTimeoutRef = useRef<number | null>(null)
  const { services, aggregateStatus, isLoading, error, refresh } =
    useServiceHealth(isOpen)

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (spinTimeoutRef.current) {
        clearTimeout(spinTimeoutRef.current)
      }
    }
  }, [])

  const handleRefresh = () => {
    setIsSpinning(true)
    refresh()
    // Clear any existing timeout
    if (spinTimeoutRef.current) {
      clearTimeout(spinTimeoutRef.current)
    }
    // Ensure at least one visible spin
    spinTimeoutRef.current = window.setTimeout(() => {
      setIsSpinning(false)
      spinTimeoutRef.current = null
    }, MIN_SPIN_DURATION_MS)
  }

  const showLoading = isLoading && services.length === 0
  const showError = error && services.length === 0

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <button
          className="w-8 h-7 flex items-center justify-center rounded-lg border border-border bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
          title="Service Status"
          aria-label="Service status"
        >
          <Activity className="w-4 h-4" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent
        align="end"
        className="w-80 p-0"
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            {aggregateStatus && (
              <AlertTriangle
                className={cn(
                  'w-4 h-4',
                  aggregateStatus === 'healthy'
                    ? 'text-emerald-500'
                    : aggregateStatus === 'degraded'
                      ? 'text-amber-500'
                      : 'text-red-500'
                )}
              />
            )}
            <span className="text-sm font-semibold text-foreground">
              Service Status
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={(e) => {
                e.preventDefault()
                handleRefresh()
              }}
              className="p-1.5 rounded hover:bg-accent transition-colors"
              disabled={isSpinning}
              title="Refresh"
              aria-label="Refresh service status"
            >
              <RefreshCw
                className={cn('w-4 h-4', isSpinning && 'animate-spin')}
              />
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1.5 rounded hover:bg-accent transition-colors"
              title="Close"
              aria-label="Close service status panel"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        {showLoading ? (
          <LoadingSpinner />
        ) : showError ? (
          <ErrorState onRetry={handleRefresh} />
        ) : (
          <>
            {/* Summary Banner */}
            {aggregateStatus && (
              <div
                className={cn(
                  'px-4 py-2 text-sm font-medium',
                  BANNER_STYLES[aggregateStatus]
                )}
              >
                {BANNER_MESSAGES[aggregateStatus]}
              </div>
            )}

            {/* Service List */}
            <div className="divide-y divide-border">
              {services.map((service) => (
                <ServiceRow key={service.id} service={service} />
              ))}
            </div>

            {/* Footer */}
            <div className="border-t border-border px-4 py-3">
              <a
                href="https://docs.llamafarm.dev/docs/troubleshooting"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                Troubleshooting docs
              </a>
            </div>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
