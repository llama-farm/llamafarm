import { lazy, Suspense } from 'react'
import { useProject } from '../hooks/useProjects'
import { useActiveProject } from '../hooks/useActiveProject'
import Loader from '../common/Loader'
import FontIcon from '../common/FontIcon'

// Lazy load the CodeMirror editor
const CodeMirrorEditor = lazy(() => import('./CodeMirrorEditor'))

interface ConfigEditorProps {
  className?: string
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ 
  className = '' 
}) => {
  // Get current project info using reactive hook
  const activeProject = useActiveProject()
  
  // Fetch project data with improved loading state
  const { 
    data: projectResponse, 
    isLoading, 
    error
  } = useProject(
    activeProject?.namespace || '', 
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project // Only enable when we have both values
  )

  // Format the configuration for display
  const formattedConfig = () => {
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

    // Return formatted JSON
    return JSON.stringify({
      project: activeProject.project,
      namespace: activeProject.namespace,
      config: projectResponse.project?.config || {},
      metadata: {
        lastUpdated: new Date().toISOString(),
        source: "llamafarm-designer"
      }
    }, null, 2)
  }

  // Only show loading on initial load, not for subsequent fetches
  const isActuallyLoading = isLoading && !projectResponse
  
  if (isActuallyLoading) {
    return (
      <div className={`config-editor w-full h-full max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}>
        {/* Header */}
        <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FontIcon type="code" className="w-4 h-4 text-foreground" />
              <h2 className="text-sm font-semibold text-foreground">
                Project Configuration
              </h2>
              <span className="text-xs text-muted-foreground">
                ({activeProject?.project || 'No project'})
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Loading...</span>
            </div>
          </div>
        </div>

        {/* Loading content */}
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader className="w-8 h-8" />
            <span className="text-sm text-muted-foreground">Loading project configuration...</span>
            {activeProject && (
              <span className="text-xs text-muted-foreground">
                Project: {activeProject.project}
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <Suspense 
      fallback={
        <div className={`config-editor w-full h-full max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}>
          {/* Header */}
          <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FontIcon type="code" className="w-4 h-4 text-foreground" />
                <h2 className="text-sm font-semibold text-foreground">
                  Project Configuration
                </h2>
                <span className="text-xs text-muted-foreground">
                  ({activeProject?.project || 'No project'})
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Loading editor...</span>
              </div>
            </div>
          </div>

          {/* Loading editor */}
          <div className="flex-1 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <Loader className="w-8 h-8" />
              <span className="text-sm text-muted-foreground">Loading code editor...</span>
            </div>
          </div>
        </div>
      }
    >
      <CodeMirrorEditor 
        content={formattedConfig()} 
        className={className}
      />
    </Suspense>
  )
}

export default ConfigEditor
