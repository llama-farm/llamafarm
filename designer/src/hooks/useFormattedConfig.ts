import { useMemo } from 'react'
import { useProject } from './useProjects'
import { useActiveProject } from './useActiveProject'
import type { UseFormattedConfigReturn } from '../types/config'

/**
 * Hook that handles all data fetching and formatting logic for project configuration
 * Extracted from ConfigEditor to improve separation of concerns
 */
export function useFormattedConfig(): UseFormattedConfigReturn {
  // Get current project info using reactive hook
  const activeProject = useActiveProject()
  
  // Fetch project data with improved loading state
  const { 
    data: projectResponse, 
    isLoading, 
    error,
    refetch
  } = useProject(
    activeProject?.namespace || '', 
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project // Only enable when we have both values
  )

  // Format the configuration for display
  const formattedConfig = useMemo(() => {
    if (error) {
      return `# Error loading project configuration
# Error: ${error instanceof Error ? error.message : 'Unknown error'}
# 
# Please check:
# - Project exists and is accessible
# - Network connection
# - Server status
#
# You can try refreshing to reload the configuration.

{
  "error": {
    "message": "${error instanceof Error ? error.message : 'Unknown error'}",
    "timestamp": "${new Date().toISOString()}"
  }
}`
    }

    if (!projectResponse || !activeProject) {
      return `# No project configuration available
#
# This could be because:
# - No project is currently selected
# - Project configuration is empty
# - Project is still loading
#
# Try selecting a project from the sidebar or creating a new one.

{
  "message": "No project configuration available",
  "activeProject": ${activeProject ? `"${activeProject.project}"` : 'null'}
}`
    }

    // Return formatted JSON without hardcoded lastUpdated
    // Note: Timestamps not included as they're not available from the API
    const configData = {
      project: activeProject.project,
      namespace: activeProject.namespace,
      config: projectResponse.project?.config || {},
      metadata: {
        source: "llamafarm-designer"
      }
    }

    return JSON.stringify(configData, null, 2)
  }, [error, projectResponse, activeProject])

  return {
    formattedConfig,
    isLoading,
    error,
    refetch,
    projectData: projectResponse
  }
}
