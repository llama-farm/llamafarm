/**
 * Utility functions for handling Server-Sent Events (SSE) parsing
 * Provides reusable logic for buffer management, line splitting, and event parsing
 */

/**
 * Configuration for SSE parsing
 */
export interface SSEParseOptions {
  /** Signal to abort the parsing operation */
  signal?: AbortSignal
  /** Callback called when the stream is complete */
  onComplete?: () => void
  /** Callback called when an error occurs */
  onError?: (error: Error) => void
}

/**
 * SSE event data parsed from a stream
 */
export interface SSEEvent {
  /** The event type (typically 'data') */
  type: string
  /** The parsed JSON data from the event */
  data: any
  /** The raw line content */
  rawLine: string
}

/**
 * Generic callback for handling parsed SSE events
 */
export type SSEEventHandler<T = any> = (event: SSEEvent & { data: T }) => void

/**
 * Parse Server-Sent Events from a ReadableStream
 * Handles buffer management, line splitting, DONE detection, and JSON parsing
 *
 * @param reader - The ReadableStreamDefaultReader to parse from
 * @param onEvent - Callback for handling parsed events
 * @param options - Additional parsing options
 * @returns Promise that resolves when parsing is complete
 */
export async function parseSSEStream<T = any>(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: SSEEventHandler<T>,
  options: SSEParseOptions = {}
): Promise<void> {
  const { signal, onComplete, onError } = options
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      // Check for abort signal
      if (signal?.aborted) {
        throw new Error('SSE parsing aborted')
      }

      const { done, value } = await reader.read()

      if (done) break

      const decodedChunk = decoder.decode(value, { stream: true })
      buffer += decodedChunk

      // Process complete lines
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmedLine = line.trim()

        if (trimmedLine === '') continue

        console.log('SSE line received:', trimmedLine.substring(0, 200))

        // Handle DONE signal
        if (trimmedLine === 'data: [DONE]') {
          console.log('SSE stream complete: [DONE] received')
          onComplete?.()
          return
        }

        // Parse SSE events
        if (trimmedLine.startsWith('data: ')) {
          try {
            const jsonData = trimmedLine.slice(6) // Remove 'data: ' prefix
            console.log('Parsing SSE data:', jsonData.substring(0, 200))
            const parsedData = JSON.parse(jsonData)

            console.log('SSE data parsed successfully, calling onEvent')

            const event: SSEEvent & { data: T } = {
              type: 'data',
              data: parsedData,
              rawLine: trimmedLine,
            }

            onEvent(event)
            console.log('onEvent called successfully')
          } catch (parseError) {
            console.error(
              'Failed to parse SSE chunk:',
              parseError,
              'Line:',
              trimmedLine.substring(0, 200)
            )
          }
        } else if (trimmedLine.startsWith('event: ')) {
          // Handle other event types if needed
          const eventType = trimmedLine.slice(7)
          console.log('SSE event type received:', eventType)
          const event: SSEEvent = {
            type: eventType,
            data: null,
            rawLine: trimmedLine,
          }
          onEvent(event as SSEEvent & { data: T })
        } else {
          console.log(
            'SSE line does not match expected format:',
            trimmedLine.substring(0, 200)
          )
        }
      }
    }
  } catch (error) {
    const parseError =
      error instanceof Error ? error : new Error('Unknown SSE parsing error')
    onError?.(parseError)
    throw parseError
  } finally {
    reader.releaseLock()
  }

  onComplete?.()
}

/**
 * Simplified SSE parser for chat streaming with chunk handling
 * Specifically designed for chat API responses
 *
 * @param reader - The ReadableStreamDefaultReader to parse from
 * @param onChunk - Callback for handling chat chunks
 * @param options - Additional parsing options
 * @returns Promise that resolves when parsing is complete
 */
export async function parseChatSSEStream<ChunkType = any>(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onChunk: (chunk: ChunkType) => void,
  options: SSEParseOptions = {}
): Promise<void> {
  console.log('parseChatSSEStream called, starting SSE parsing')
  return parseSSEStream<ChunkType>(
    reader,
    event => {
      console.log('SSE event received in parseChatSSEStream:', {
        type: event.type,
        hasData: !!event.data,
        dataKeys: event.data ? Object.keys(event.data) : [],
      })
      if (event.type === 'data' && event.data) {
        console.log('Calling onChunk with data:', event.data)
        onChunk(event.data)
        console.log('onChunk called successfully')
      } else {
        console.log(
          'Skipping event - type:',
          event.type,
          'hasData:',
          !!event.data
        )
      }
    },
    options
  )
}

/**
 * Utility to create a ReadableStreamDefaultReader from a Response body
 * Handles validation and error cases
 *
 * @param response - The fetch Response object
 * @returns ReadableStreamDefaultReader for the response body
 * @throws Error if response body is not available
 */
export function getSSEReader(
  response: Response
): ReadableStreamDefaultReader<Uint8Array> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('No response body available for SSE parsing')
  }
  return reader
}

/**
 * Helper to handle SSE response from fetch with automatic reader setup
 * Combines response validation, reader creation, and parsing
 *
 * @param response - The fetch Response object
 * @param onChunk - Callback for handling chat chunks
 * @param options - Additional parsing options
 * @returns Promise that resolves when parsing is complete
 */
export async function handleSSEResponse<ChunkType = any>(
  response: Response,
  onChunk: (chunk: ChunkType) => void,
  options: SSEParseOptions = {}
): Promise<void> {
  console.log('handleSSEResponse called:', {
    ok: response.ok,
    status: response.status,
    contentType: response.headers.get('content-type'),
    hasBody: !!response.body,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error')
    throw new Error(
      `HTTP ${response.status}: ${response.statusText} - ${errorText}`
    )
  }

  const reader = getSSEReader(response)
  console.log('SSE reader created, starting parseChatSSEStream')

  try {
    await parseChatSSEStream<ChunkType>(reader, onChunk, options)
    console.log('parseChatSSEStream completed successfully')
  } catch (error) {
    console.error('Error in parseChatSSEStream:', error)
    // Ensure reader is properly released on error
    try {
      reader.releaseLock()
    } catch (releaseError) {
      // Ignore release errors
    }
    throw error
  }
}

/**
 * Default export with all SSE utilities
 */
export default {
  parseSSEStream,
  parseChatSSEStream,
  getSSEReader,
  handleSSEResponse,
}
