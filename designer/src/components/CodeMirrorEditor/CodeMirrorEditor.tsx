import React from 'react'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useCodeMirror } from '../../hooks/useCodeMirror'
import EditorToolbar from './EditorToolbar'
import EditorContent from './EditorContent'
import EditorFallback from './EditorFallback'
import Loader from '../../common/Loader'
import FontIcon from '../../common/FontIcon'
import type { CodeMirrorEditorProps } from '../../types/codemirror'

/**
 * CodeMirror editor component for displaying project configurations
 * Refactored to use composition of smaller, focused components
 */
const CodeMirrorEditor: React.FC<CodeMirrorEditorProps> = ({
  content,
  className = '',
  readOnly = true,
  language = 'json',
  theme,
  onChange,
  onSave,
  onDiscard,
  onCopy,
  isDirty = false,
  isSaving = false,
  saveError = null,
  copyStatus = 'idle',
  onEditorReady,
}) => {
  const activeProject = useActiveProject()

  // Use the extracted CodeMirror hook
  const { editorRef, isLoading, isInitialized, error, navigationAPI } =
    useCodeMirror(content, {
      readOnly,
      language,
      theme,
      onChange,
    })

  // Notify parent when navigation API is ready
  React.useEffect(() => {
    if (navigationAPI && onEditorReady) {
      onEditorReady(navigationAPI)
    }
  }, [navigationAPI, onEditorReady])

  const copyIconType =
    copyStatus === 'success'
      ? 'checkmark-filled'
      : copyStatus === 'error'
        ? 'info'
        : 'copy'

  const copyIconClass =
    copyStatus === 'success'
      ? 'w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400'
      : copyStatus === 'error'
        ? 'w-3.5 h-3.5 text-destructive'
        : 'w-3.5 h-3.5 text-foreground'

  const copyTitle =
    copyStatus === 'success'
      ? 'Configuration copied to clipboard'
      : copyStatus === 'error'
        ? 'Copy failed. Click to try again.'
        : 'Copy configuration to clipboard'

  const renderCopyButton = () => {
    if (!onCopy) return null

    return (
      <button
        onClick={onCopy}
        className="absolute top-3 right-4 z-10 p-2 rounded-md border border-border bg-card hover:bg-muted transition-colors shadow-sm"
        title={copyTitle}
        aria-label={copyTitle}
      >
        <FontIcon type={copyIconType} className={copyIconClass} />
      </button>
    )
  }

  // Show loading state while CodeMirror modules are loading
  if (isLoading) {
    return (
      <div
        className={`w-full h-full max-h-full rounded-tr-lg rounded-br-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}
      >
        <EditorToolbar
          activeProject={activeProject}
          readOnly={readOnly}
          onSave={onSave}
          onDiscard={onDiscard}
          isDirty={isDirty}
          isSaving={isSaving}
          saveError={saveError}
        />

        {/* Loading content */}
        <div className="flex-1 flex items-center justify-center relative">
          {renderCopyButton()}
          <div className="flex flex-col items-center gap-3">
            <Loader className="w-8 h-8" />
            <span className="text-sm text-muted-foreground">
              Loading code editor...
            </span>
          </div>
        </div>
      </div>
    )
  }

  // Show error state if CodeMirror failed to load
  if (error) {
    return (
      <div
        className={`w-full h-full max-h-full rounded-tr-lg rounded-br-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}
      >
        <EditorToolbar
          activeProject={activeProject}
          readOnly={readOnly}
          onSave={onSave}
          onDiscard={onDiscard}
          isDirty={isDirty}
          isSaving={isSaving}
          saveError={saveError}
        />

        {/* Error content */}
        <div className="flex-1 flex items-center justify-center p-6 relative">
          {renderCopyButton()}
          <div className="max-w-md text-center space-y-4">
            <div className="w-16 h-16 mx-auto rounded-full bg-destructive/10 flex items-center justify-center">
              <FontIcon type="info" className="w-8 h-8 text-destructive" />
            </div>

            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-foreground">
                Editor Failed to Load
              </h3>
              <p className="text-sm text-muted-foreground">
                The code editor could not be initialized.
              </p>
            </div>

            <div className="p-3 bg-destructive/5 border border-destructive/20 rounded-md text-left">
              <p className="text-xs font-mono text-destructive break-words">
                {error}
              </p>
            </div>

            <p className="text-xs text-muted-foreground">
              Please refresh the page to try again.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`w-full h-full max-h-full rounded-tr-lg rounded-br-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}
    >
      <EditorToolbar
        activeProject={activeProject}
        readOnly={readOnly}
        onSave={onSave}
        onDiscard={onDiscard}
        isDirty={isDirty}
        isSaving={isSaving}
        saveError={saveError}
      />

      {/* Editor container with fallback */}
      <div className="flex-1 min-h-0 relative">
        {renderCopyButton()}
        <EditorContent editorRef={editorRef} />
        <EditorFallback content={content} isInitialized={isInitialized} />
      </div>
    </div>
  )
}

export default CodeMirrorEditor
