import { lazy, Suspense } from 'react'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useFormattedConfig } from '../../hooks/useFormattedConfig'
import { useTheme } from '../../contexts/ThemeContext'
import ConfigLoading from './ConfigLoading'
import ConfigError from './ConfigError'
import Loader from '../../common/Loader'
import FontIcon from '../../common/FontIcon'

// Lazy load the CodeMirror editor
const CodeMirrorEditor = lazy(() => import('../CodeMirrorEditor/CodeMirrorEditor'))

interface ConfigEditorProps {
  className?: string
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ 
  className = '' 
}) => {
  // Get current project info using reactive hook
  const activeProject = useActiveProject()
  const { theme } = useTheme()
  
  // Use the extracted hook for all data fetching and formatting
  const { 
    formattedConfig, 
    isLoading, 
    error, 
    refetch,
    projectData 
  } = useFormattedConfig()

  // Only show loading on initial load, not for subsequent fetches
  const isActuallyLoading = isLoading && !projectData
  
  if (isActuallyLoading) {
    return <ConfigLoading activeProject={activeProject} className={className} />
  }

  if (error) {
    return (
      <ConfigError 
        error={error} 
        activeProject={activeProject} 
        onRetry={refetch}
        className={className} 
      />
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
        content={formattedConfig} 
        className={className}
        theme={theme}
      />
    </Suspense>
  )
}

export default ConfigEditor
