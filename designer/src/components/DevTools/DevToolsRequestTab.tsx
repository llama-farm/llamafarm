import type { CapturedRequest } from '../../contexts/DevToolsContext'

interface DevToolsRequestTabProps {
  request: CapturedRequest
}

function CodeBlock({ content }: { content: string }) {
  return (
    <pre className="block p-3 rounded bg-muted text-foreground font-mono text-xs overflow-x-auto max-h-48 scrollbar-thin">
      {content}
    </pre>
  )
}

function HeadersTable({ headers }: { headers: Record<string, string> }) {
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

export default function DevToolsRequestTab({ request }: DevToolsRequestTabProps) {
  const bodyContent = request.body
    ? typeof request.body === 'string'
      ? request.body
      : JSON.stringify(request.body, null, 2)
    : null

  return (
    <div className="space-y-4 p-4 overflow-y-auto max-h-64 scrollbar-thin">
      {/* URL */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-1">URL</h4>
        <code className="text-xs font-mono text-foreground break-all">
          {request.fullUrl}
        </code>
      </div>

      {/* Headers */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">Headers</h4>
        <HeadersTable headers={request.headers} />
      </div>

      {/* Body */}
      {bodyContent && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">Body</h4>
          <CodeBlock content={bodyContent} />
        </div>
      )}
    </div>
  )
}
