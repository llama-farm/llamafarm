import React from 'react'
import FontIcon from '../../common/FontIcon'
import type { EditorToolbarProps } from '../../types/config'

/**
 * Toolbar component for CodeMirror editor
 * Displays project information and editor controls
 */
const EditorToolbar: React.FC<EditorToolbarProps> = ({ 
  activeProject, 
  readOnly = true,
  onRefresh 
}) => {
  return (
    <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FontIcon type="code" className="w-4 h-4 text-foreground" />
          <h2 className="text-sm font-semibold text-foreground">
            Project Configuration
          </h2>
          {activeProject && (
            <span className="text-xs text-muted-foreground">
              ({activeProject.project})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors p-1 rounded"
              title="Refresh configuration"
            >
              <FontIcon type="recently-viewed" className="w-3 h-3" />
            </button>
          )}
          {readOnly && (
            <>
              <span className="text-xs text-muted-foreground">Read-only</span>
              <FontIcon type="eye-off" className="w-3 h-3 text-muted-foreground" />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default EditorToolbar
