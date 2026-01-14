import type { CapturedRequest } from '../../contexts/DevToolsContext'
import { CodeBlock, HeadersTable } from './DevToolsShared'

interface DevToolsResponseTabProps {
  request: CapturedRequest
}

function StreamingIndicator({ chunkCount }: { chunkCount: number }) {
  return (
    <div className="flex items-center gap-2 text-xs text-teal-400 mb-2">
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-teal-500" />
      <span>Streaming response...</span>
      <span className="text-muted-foreground">({chunkCount} chunks)</span>
    </div>
  )
}

export default function DevToolsResponseTab({ request }: DevToolsResponseTabProps) {
  const isStreaming = request.isStreaming && !request.streamComplete
  const hasError = !!request.error
  // Check if response is complete (has status code)
  const hasResponse = request.status !== undefined

  // For streaming responses, show accumulated chunks
  let responseContent: string | null = null
  let hasResponseBody = false

  if (request.isStreaming && request.streamChunks.length > 0) {
    // Try to extract content from stream chunks
    const contents = request.streamChunks
      .map(chunk => {
        // Handle different chunk formats
        if (typeof chunk === 'string') return chunk
        if (chunk?.choices?.[0]?.delta?.content) return chunk.choices[0].delta.content
        if (chunk?.content) return chunk.content
        return ''
      })
      .filter(Boolean)

    if (contents.length > 0) {
      responseContent = contents.join('')
      hasResponseBody = true
    } else {
      // Fallback: show raw chunks
      responseContent = JSON.stringify(request.streamChunks, null, 2)
      hasResponseBody = true
    }
  } else if (request.responseBody !== null && request.responseBody !== undefined) {
    // Check explicitly for null/undefined to allow empty strings
    hasResponseBody = true
    responseContent =
      typeof request.responseBody === 'string'
        ? request.responseBody
        : JSON.stringify(request.responseBody, null, 2)
  }

  return (
    <div className="space-y-4 p-4 overflow-y-auto h-full scrollbar-thin">
      {/* Streaming indicator */}
      {isStreaming && <StreamingIndicator chunkCount={request.streamChunks.length} />}

      {/* Error display */}
      {hasError && (
        <div className="p-3 rounded bg-red-500/10 border border-red-500/20">
          <h4 className="text-xs font-medium text-red-400 mb-1">Error</h4>
          <p className="text-xs text-red-300">{request.error}</p>
        </div>
      )}

      {/* Status */}
      {request.status !== undefined && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1">Status</h4>
          <span className="text-xs font-mono text-foreground">
            {request.status} {request.statusText}
          </span>
        </div>
      )}

      {/* Response Headers */}
      {request.responseHeaders && Object.keys(request.responseHeaders).length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            Response Headers
          </h4>
          <HeadersTable headers={request.responseHeaders} />
        </div>
      )}

      {/* Response Body */}
      {hasResponseBody ? (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            {request.isStreaming ? 'Streamed Content' : 'Response Body'}
          </h4>
          <CodeBlock content={responseContent ?? ''} />
        </div>
      ) : !isStreaming && !hasError && !hasResponse ? (
        <div className="text-xs text-muted-foreground">Awaiting response...</div>
      ) : null}
    </div>
  )
}
