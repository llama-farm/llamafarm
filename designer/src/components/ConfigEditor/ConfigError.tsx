import React from 'react'
import FontIcon from '../../common/FontIcon'
import type { ConfigErrorProps } from '../../types/config'

/**
 * Error state component for ConfigEditor
 * Displays error information and provides retry functionality
 */
const ConfigError: React.FC<ConfigErrorProps> = ({ 
  error, 
  activeProject, 
  onRetry,
  className = '' 
}) => {
  const errorMessage = error instanceof Error ? error.message : 'Unknown error'

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
            <FontIcon type="info" className="w-4 h-4 text-destructive" />
            <span className="text-xs text-destructive">Error</span>
          </div>
        </div>
      </div>

      {/* Error content */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md text-center space-y-4">
          <div className="w-16 h-16 mx-auto rounded-full bg-destructive/10 flex items-center justify-center">
            <FontIcon type="info" className="w-8 h-8 text-destructive" />
          </div>
          
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">
              Failed to Load Configuration
            </h3>
            <p className="text-sm text-muted-foreground">
              There was an error loading the project configuration.
            </p>
          </div>

          <div className="p-3 bg-destructive/5 border border-destructive/20 rounded-md text-left">
            <p className="text-xs font-mono text-destructive break-words">
              {errorMessage}
            </p>
          </div>

          <div className="space-y-2 text-xs text-muted-foreground">
            <p>Please check:</p>
            <ul className="list-disc list-inside space-y-1 text-left">
              <li>Project exists and is accessible</li>
              <li>Network connection</li>
              <li>Server status</li>
            </ul>
          </div>

          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-foreground bg-primary rounded-md hover:bg-primary/90 transition-colors"
            >
              <FontIcon type="recently-viewed" className="w-3 h-3" />
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default ConfigError
