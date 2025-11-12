/**
 * Message Content Parser Utility
 *
 * Extracts and processes special content sections from assistant messages:
 * - <think> tags (thinking/reasoning steps)
 * - <tool_call> tags (tool invocation blocks)
 */

/**
 * Parsed message content with separated thinking and display content
 */
export interface ParsedMessageContent {
  /** Extracted thinking/reasoning content from <think> tags */
  thinking: string | null
  /** Display content with thinking and tool calls removed */
  contentWithoutThinking: string
}

/**
 * Parse message content to extract thinking steps and remove tool calls
 *
 * This function is designed to be memoized or cached to avoid recreating
 * regex patterns on every render.
 *
 * @param content - Raw message content from assistant
 * @param messageType - Type of message (only processes 'assistant' messages)
 * @returns Parsed content with thinking separated
 */
export function parseMessageContent(
  content: string,
  messageType: string
): ParsedMessageContent {
  // Only parse assistant messages
  if (messageType !== 'assistant') {
    return { thinking: null, contentWithoutThinking: content }
  }

  let processedContent = content
  let thinking = ''

  // Extract all <think> sections (handles multiple blocks)
  const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/g
  processedContent = content.replace(thinkRegex, (_match, thinkContent) => {
    const trimmedContent = thinkContent.trim()
    if (trimmedContent) {
      thinking += (thinking ? '\n\n' : '') + trimmedContent
    }
    return '' // Remove the entire <think> block
  })

  // Remove any orphaned closing tags
  processedContent = processedContent.replace(/<\/think>/g, '')

  // Remove <tool_call> XML tags from display (handles both closed and unclosed during streaming)
  // First remove complete tool_call blocks
  processedContent = processedContent.replace(
    /<tool_call>[\s\S]*?<\/tool_call>/g,
    ''
  )
  // Then remove any unclosed tool_call tags (streaming in progress)
  processedContent = processedContent.replace(/<tool_call>[\s\S]*$/g, '')
  // Also remove orphaned closing tags
  processedContent = processedContent.replace(/<\/tool_call>/g, '')

  // Clean up extra whitespace
  processedContent = processedContent.trim()

  return {
    thinking: thinking || null,
    contentWithoutThinking: processedContent,
  }
}

/**
 * Memoization cache for parsed message content
 * Key: `${messageId}:${contentLength}` (simple hash to detect content changes)
 */
const parseCache = new Map<string, ParsedMessageContent>()

/**
 * Memoized version of parseMessageContent
 * Uses a simple cache to avoid re-parsing unchanged messages
 *
 * @param content - Raw message content
 * @param messageType - Type of message
 * @param messageId - Unique message identifier for cache key
 * @returns Parsed content (cached if available)
 */
export function parseMessageContentMemo(
  content: string,
  messageType: string,
  messageId: string
): ParsedMessageContent {
  // Create cache key based on message ID and content length
  // This is a simple but effective way to detect changes
  const cacheKey = `${messageId}:${content.length}`

  // Check cache first
  const cached = parseCache.get(cacheKey)
  if (cached) {
    return cached
  }

  // Parse and cache
  const parsed = parseMessageContent(content, messageType)
  parseCache.set(cacheKey, parsed)

  // Limit cache size to prevent memory leaks
  if (parseCache.size > 100) {
    // Remove oldest entry (first in map)
    const firstKey = parseCache.keys().next().value
    if (firstKey) {
      parseCache.delete(firstKey)
    }
  }

  return parsed
}

/**
 * Clear the parse cache (useful for testing or memory management)
 */
export function clearParseCache(): void {
  parseCache.clear()
}
