import type { CapturedRequest } from '../../contexts/DevToolsContext'
import { CodeBlock, HeadersTable } from './DevToolsShared'

interface DevToolsRequestTabProps {
  request: CapturedRequest
}

export default function DevToolsRequestTab({ request }: DevToolsRequestTabProps) {
  // Check if body is defined (not null/undefined), even if it's an empty string
  const hasBody = request.body !== null && request.body !== undefined
  const bodyContent = hasBody
    ? typeof request.body === 'string'
      ? request.body
      : JSON.stringify(request.body, null, 2)
    : null

  return (
    <div className="space-y-4 p-4 overflow-y-auto h-full scrollbar-thin">
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
      {hasBody && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">Body</h4>
          <CodeBlock content={bodyContent ?? ''} />
        </div>
      )}
    </div>
  )
}
