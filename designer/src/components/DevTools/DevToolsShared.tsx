import { cn } from '@/lib/utils'

/**
 * Shared components and constants for DevTools
 */

// HTTP method color mapping
export const methodColors: Record<string, string> = {
  GET: 'bg-sky-500/20 text-sky-300',
  POST: 'bg-green-500/20 text-green-400',
  PUT: 'bg-yellow-500/20 text-yellow-400',
  PATCH: 'bg-orange-500/20 text-orange-400',
  DELETE: 'bg-red-500/20 text-red-400',
}

// Status code badge component
export function StatusBadge({ status }: { status?: number }) {
  if (!status) return null

  const isSuccess = status >= 200 && status < 300
  return (
    <span
      className={cn(
        'px-1 py-0.5 rounded text-[10px] font-medium',
        isSuccess
          ? 'bg-emerald-500/20 text-emerald-400'
          : 'bg-red-500/20 text-red-400'
      )}
    >
      {status}
    </span>
  )
}

// HTTP method badge component
export function MethodBadge({ method }: { method: string }) {
  return (
    <span
      className={cn(
        'px-1 py-0.5 rounded text-[10px] font-mono font-medium shrink-0',
        methodColors[method] || 'bg-muted'
      )}
    >
      {method}
    </span>
  )
}

// Code block for displaying JSON/text content
export function CodeBlock({ content }: { content: string }) {
  return (
    <pre className="block p-3 rounded bg-muted text-foreground font-mono text-xs overflow-auto scrollbar-thin">
      {content}
    </pre>
  )
}

// Headers table component
export function HeadersTable({ headers }: { headers: Record<string, string> }) {
  const entries = Object.entries(headers)
  if (entries.length === 0) {
    return <span className="text-xs text-muted-foreground">No headers</span>
  }

  return (
    <div className="space-y-1">
      {entries.map(([key, value]) => (
        <div key={key} className="flex gap-2 text-xs">
          <span className="font-mono text-muted-foreground shrink-0">{key}:</span>
          <span className="font-mono text-foreground break-all">{value}</span>
        </div>
      ))}
    </div>
  )
}
