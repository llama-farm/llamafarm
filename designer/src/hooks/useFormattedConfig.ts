import { useMemo } from 'react'
import { useProject } from './useProjects'
import { useActiveProject } from './useActiveProject'
import yaml from 'yaml'
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
      // Escape error message for safe YAML embedding
      const escapeYamlString = (str: string) => {
        return str
          .replace(/\\/g, '\\\\')
          .replace(/"/g, '\\"')
          .replace(/\n/g, '\\n')
          .replace(/\r/g, '\\r')
          .replace(/\t/g, '\\t')
      }

      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      const escapedErrorMessage = escapeYamlString(errorMessage)

      return `# Error loading project configuration
# Error: ${escapedErrorMessage}
#
# Please check:
# - Project exists and is accessible
# - Network connection
# - Server status
#
# You can try refreshing to reload the configuration.

error:
  message: "${escapedErrorMessage}"
  timestamp: "${new Date().toISOString()}"
`
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

message: "No project configuration available"
activeProject: ${activeProject ? `"${activeProject.project}"` : 'null'}
`
    }

    // Return the config as YAML
    // This is the actual project configuration that can be edited
    const config = projectResponse.project?.config || {}

    try {
      return yaml.stringify(config, {
        indent: 2,
        lineWidth: 0, // Don't wrap lines
        minContentWidth: 0
      })
    } catch (yamlError) {
      // Handle circular references or non-serializable values
      return `# Error converting configuration to YAML
# Error: ${yamlError instanceof Error ? yamlError.message : 'Failed to serialize configuration'}
#
# This could be due to:
# - Circular references in the configuration
# - Non-serializable values
# - Invalid data structure
#
# Raw config (as JSON):

${JSON.stringify(config, null, 2)}
`
    }
  }, [error, projectResponse, activeProject])

  return {
    formattedConfig,
    isLoading,
    error,
    refetch,
    projectData: projectResponse
  }
}
