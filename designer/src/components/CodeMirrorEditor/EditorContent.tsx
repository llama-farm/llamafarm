import React from 'react'
import type { EditorContentProps } from '../../types/config'

/**
 * Main content area for CodeMirror editor
 * Contains the actual editor DOM element
 */
const EditorContent: React.FC<EditorContentProps> = ({ 
  editorRef, 
  className = '' 
}) => {
  return (
    <div 
      ref={editorRef} 
      className={`config-editor-content flex-1 min-h-0 overflow-hidden bg-background dark:bg-[#10182e] scrollbar-custom ${className}`}
      role="textbox"
      aria-label="Project configuration editor"
      aria-readonly="true"
    >
      {/* CodeMirror will be mounted here by the hook */}
    </div>
  )
}

export default EditorContent
