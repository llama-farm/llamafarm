import React from 'react'
import type { EditorFallbackProps } from '../../types/config'

/**
 * Fallback component when CodeMirror fails to initialize
 * Provides plain text rendering with consistent styling
 */
const EditorFallback: React.FC<EditorFallbackProps> = ({ 
  content, 
  isInitialized,
  className = '' 
}) => {
  if (isInitialized) {
    return null
  }

  return (
    <div className={`absolute inset-0 p-4 text-sm font-mono overflow-auto 
      bg-background dark:bg-[#10182e] 
      scrollbar-custom ${className}`}
      role="textbox"
      aria-label="Project configuration (fallback view)"
      aria-readonly="true"
    >
      <pre className="whitespace-pre-wrap text-foreground m-0 leading-normal">{content}</pre>
    </div>
  )
}

export default EditorFallback
