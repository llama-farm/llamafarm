/**
 * @deprecated This file is kept for backward compatibility only.
 * Please use the unified useChatbox hook with { useProjectSession: true } option instead.
 *
 * This file now simply re-exports the unified hook with appropriate defaults.
 */

import { useChatbox } from './useChatbox'

/**
 * @deprecated Use useChatbox({ useProjectSession: true }) instead
 */
export function useChatboxWithProjectSession(enableStreaming: boolean = true) {
  return useChatbox({
    useProjectSession: true,
    enableStreaming,
    chatService: 'designer',
  })
}

// Re-export for backward compatibility
export default useChatboxWithProjectSession

// Re-export constants that were in the original file
export const PROJECT_SEED_NAMESPACE = 'llamafarm'
export const PROJECT_SEED_PROJECT = 'project_seed'
