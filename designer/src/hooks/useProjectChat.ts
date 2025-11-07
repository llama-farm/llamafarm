/**
 * Project Chat Utilities
 *
 * Provides utility functions for project chat operations.
 * Chat completion functionality has been moved to useChatCompletions.ts
 */

/**
 * Utility hook to extract and validate chat parameters from active project
 * @param activeProject - The active project information
 * @returns Validated namespace and projectId, or null if invalid
 */
export const useProjectChatParams = (
  activeProject: { namespace: string; project: string } | null
) => {
  if (!activeProject?.namespace || !activeProject?.project) {
    return null
  }

  return {
    namespace: activeProject.namespace,
    projectId: activeProject.project,
  }
}

export default {
  useProjectChatParams,
}
