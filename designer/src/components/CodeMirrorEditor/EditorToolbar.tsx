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
  onRefresh,
  onSave,
  onDiscard,
  isDirty = false,
  isSaving = false,
  saveError = null
}) => {
  return (
    <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
      {/* Main toolbar row */}
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
          {isDirty && (
            <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">
              • Unsaved changes
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
          {!readOnly && onDiscard && (
            <button
              onClick={onDiscard}
              disabled={!isDirty || isSaving}
              className="text-xs px-3 py-1.5 rounded border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Discard changes"
            >
              Discard
            </button>
          )}
          {!readOnly && onSave && (
            <button
              onClick={onSave}
              disabled={!isDirty || isSaving}
              className="text-xs px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
              title="Save changes"
            >
              {isSaving ? (
                <>
                  <FontIcon type="loading" className="w-3 h-3 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <FontIcon type="save" className="w-3 h-3" />
                  Save
                </>
              )}
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

      {/* Error banner */}
      {saveError && (
        <div className="mt-3 px-3 py-2 bg-destructive/10 border border-destructive/30 rounded-md flex items-start gap-2" role="alert">
          <FontIcon type="info" className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs text-destructive break-words">
              {saveError}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default EditorToolbar
