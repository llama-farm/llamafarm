/**
 * Legacy chat service - minimal functions kept for compatibility
 * All new code should use chatCompletionsService.ts instead
 */

import { apiClient } from './client'
import { DeleteSessionResponse } from '../types/chat'

/**
 * Delete a chat session
 * @param sessionId - The session ID to delete
 * @returns Promise<DeleteSessionResponse>
 * @deprecated Use project-scoped session deletion via unified chat completions interface
 */
export async function deleteChatSession(
  sessionId: string
): Promise<DeleteSessionResponse> {
  const response = await apiClient.delete<DeleteSessionResponse>(
    `/inference/chat/sessions/${encodeURIComponent(sessionId)}`
  )
  return response.data
}

export default {
  deleteChatSession,
}
