import React from 'react'
import Loader from '../../common/Loader'
import FontIcon from '../../common/FontIcon'
import type { ConfigLoadingProps } from '../../types/config'

/**
 * Loading state component for ConfigEditor
 * Displays a consistent loading UI while project configuration is being fetched
 */
const ConfigLoading: React.FC<ConfigLoadingProps> = ({ 
  activeProject, 
  className = '' 
}) => {
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

export default ConfigLoading
