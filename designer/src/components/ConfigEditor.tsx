import { useEffect, useRef, useState, useMemo } from 'react'
import { EditorView, lineNumbers, keymap } from '@codemirror/view'
import { EditorState, StateEffect } from '@codemirror/state'
import { json } from '@codemirror/lang-json'
import { oneDark } from '@codemirror/theme-one-dark'
import { defaultKeymap } from '@codemirror/commands'
import { bracketMatching, indentOnInput, foldGutter, syntaxHighlighting, HighlightStyle } from '@codemirror/language'
import { highlightSelectionMatches } from '@codemirror/search'
import { tags } from '@lezer/highlight'
import { useTheme } from '../contexts/ThemeContext'
import { useProject } from '../hooks/useProjects'
import { useActiveProject } from '../hooks/useActiveProject'
import Loader from '../common/Loader'
import FontIcon from '../common/FontIcon'

// Custom light theme for CodeMirror with proper syntax highlighting
const lightTheme = EditorView.theme({
  '&': {
    color: '#24292f',
    backgroundColor: '#ffffff',
    height: '100%'
  },
  '.cm-content': {
    padding: '16px',
    minHeight: '100%'
  },
  '.cm-focused': {
    outline: 'none'
  },
  '.cm-editor': {
    borderRadius: '8px',
    height: '100%'
  },
  '.cm-scroller': {
    lineHeight: '1.5',
    height: '100%'
  },
  '.cm-cursor': {
    display: 'none' // Hide cursor in read-only mode
  },
  '.cm-gutters': {
    backgroundColor: '#f6f8fa',
    color: '#656d76',
    border: 'none',
    borderRight: '1px solid #d1d9e0'
  },
  '.cm-activeLineGutter': {
    backgroundColor: '#f6f8fa'
  },
  '.cm-foldGutter .cm-gutterElement': {
    color: '#656d76'
  },
  '.cm-selectionBackground': {
    backgroundColor: '#0969da26'
  },
  '.cm-scroller::-webkit-scrollbar': {
    width: '12px'
  },
  '.cm-scroller::-webkit-scrollbar-track': {
    backgroundColor: '#f6f8fa'
  },
  '.cm-scroller::-webkit-scrollbar-thumb': {
    backgroundColor: '#d1d9e0',
    borderRadius: '6px',
    border: '2px solid #f6f8fa'
  },
  '.cm-scroller::-webkit-scrollbar-thumb:hover': {
    backgroundColor: '#afb8c1'
  }
}, { dark: false })

// Custom syntax highlighting for light theme
const lightHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: '#cf222e' },
  { tag: tags.atom, color: '#0969da' },
  { tag: tags.bool, color: '#0969da' },
  { tag: tags.url, color: '#0969da' },
  { tag: tags.labelName, color: '#116329' },
  { tag: tags.inserted, color: '#116329' },
  { tag: tags.deleted, color: '#d1242f' },
  { tag: tags.literal, color: '#0969da' },
  { tag: tags.string, color: '#0a3069' },
  { tag: tags.number, color: '#0969da' },
  { tag: [tags.regexp, tags.escape, tags.special(tags.string)], color: '#bc4c00' },
  { tag: tags.definition(tags.variableName), color: '#24292f' },
  { tag: tags.local(tags.variableName), color: '#24292f' },
  { tag: [tags.typeName, tags.namespace], color: '#8250df' },
  { tag: tags.className, color: '#8250df' },
  { tag: [tags.special(tags.variableName), tags.macroName], color: '#0969da' },
  { tag: tags.definition(tags.propertyName), color: '#0969da' },
  { tag: tags.propertyName, color: '#0969da' },
  { tag: tags.comment, color: '#656d76', fontStyle: 'italic' },
  { tag: tags.meta, color: '#656d76' },
  { tag: tags.invalid, color: '#d1242f' },
  { tag: tags.punctuation, color: '#24292f' },
  { tag: tags.bracket, color: '#24292f' }
])

/**
 * ProjectConfigViewer - A CodeMirror-based read-only configuration viewer
 * 
 * Features:
 * - Loads project configuration from the API using useProject hook
 * - JSON syntax highlighting with CodeMirror 6
 * - Dynamic theme switching (light/dark) matching app theme
 * - Read-only mode with proper error handling and loading states
 * - Responsive design using Tailwind CSS
 */
interface ConfigEditorProps {
  className?: string
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ 
  className = '' 
}) => {
  const { theme } = useTheme()
  const editorRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<EditorView | null>(null)
  const [isInitialized, setIsInitialized] = useState(false)
  
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



  // Create extensions configuration
  const createExtensions = useMemo(() => {
    return () => [
      // Language support
      json(),
      
      // Editor features
      lineNumbers(),
      foldGutter(),
      bracketMatching(),
      indentOnInput(),
      highlightSelectionMatches(),
      
      // Read-only configuration
      EditorView.editable.of(false),
      EditorState.readOnly.of(true),
      
      // Keymaps (limited for read-only)
      keymap.of([
        ...defaultKeymap.filter(binding => {
          // Allow navigation and selection commands only
          const allowedKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']
          const allowedCommands = ['selectAll', 'copy', 'cursorDocStart', 'cursorDocEnd', 'cursorLineStart', 'cursorLineEnd']
          return allowedKeys.includes(binding.key || '') || 
                 allowedCommands.some(cmd => binding.run?.name?.includes(cmd))
        })
      ]),
      
      // Typography theme (applied to both light and dark)
      EditorView.theme({
        '&': {
          fontSize: '14px',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
        }
      }),
      
      // Theme extensions with syntax highlighting
      ...(theme === 'dark' 
        ? [
            oneDark,
            EditorView.theme({
              '&': {
                height: '100%'
              },
              '.cm-editor': {
                height: '100%'
              },
              '.cm-scroller': {
                height: '100%'
              },
              '.cm-content': {
                minHeight: '100%'
              },
              '.cm-scroller::-webkit-scrollbar': {
                width: '12px'
              },
              '.cm-scroller::-webkit-scrollbar-track': {
                backgroundColor: '#1c2028'
              },
              '.cm-scroller::-webkit-scrollbar-thumb': {
                backgroundColor: '#3e4451',
                borderRadius: '6px',
                border: '2px solid #1c2028'
              },
              '.cm-scroller::-webkit-scrollbar-thumb:hover': {
                backgroundColor: '#5c6370'
              }
            }, { dark: true })
          ]
        : [lightTheme, syntaxHighlighting(lightHighlightStyle)]
      )
    ]
  }, [theme])

  // Format project configuration as JSON
  const formattedConfig = useMemo(() => {
    // Always show something - even if no project or data
    if (!activeProject) {
      return JSON.stringify({
        message: "No active project",
        note: "Select a project from the dropdown to view its configuration"
      }, null, 2)
    }
    
    if (!projectResponse) {
      return JSON.stringify({
        message: "Loading project data...",
        project: activeProject.project,
        namespace: activeProject.namespace
      }, null, 2)
    }
    
    if (error) {
      return JSON.stringify({
        message: "Error loading project configuration",
        error: error.message || 'Unknown error',
        project: activeProject.project
      }, null, 2)
    }

    const config = projectResponse.project?.config
    
    // Always show the config, even if empty
    return JSON.stringify({
      name: projectResponse.project?.name || activeProject.project,
      namespace: projectResponse.project?.namespace || activeProject.namespace,
      configuration: config || {},
      message: config && Object.keys(config).length > 2 ? 
        "Project configuration loaded" : 
        "Project has minimal configuration - add data, models, and prompts using the tabs above"
    }, null, 2)
  }, [activeProject, projectResponse, error])

  // Initialize CodeMirror
  useEffect(() => {
    if (!editorRef.current) return
    
    // Clean up previous editor if it exists
    if (viewRef.current) {
      viewRef.current.destroy()
      viewRef.current = null
      setIsInitialized(false)
    }
    
    // DEBUG: Log scrollbar elements
    console.log('🔍 SCROLLBAR DEBUG - Theme:', theme)
    const scrollableElements = document.querySelectorAll('[style*="overflow"], .cm-scroller, .cm-editor')
    scrollableElements.forEach((el, i) => {
      console.log(`Element ${i}:`, el, getComputedStyle(el).overflow)
    })

    const extensions = createExtensions()

    const state = EditorState.create({
      doc: formattedConfig,
      extensions
    })

    const view = new EditorView({
      state,
      parent: editorRef.current
    })

    viewRef.current = view
    setIsInitialized(true)

    return () => {
      view.destroy()
      viewRef.current = null
      setIsInitialized(false)
    }
  }, []) // Only initialize once

  // Update content when project data changes
  useEffect(() => {
    if (!viewRef.current || !isInitialized) return

    const currentDoc = viewRef.current.state.doc.toString()
    if (currentDoc !== formattedConfig) {
      viewRef.current.dispatch({
        changes: {
          from: 0,
          to: viewRef.current.state.doc.length,
          insert: formattedConfig
        }
      })
    }
  }, [formattedConfig, isInitialized])

  // Update theme when it changes
  useEffect(() => {
    if (!viewRef.current || !isInitialized) return
    
    viewRef.current.dispatch({
      effects: StateEffect.reconfigure.of(createExtensions())
    })
  }, [theme, isInitialized, createExtensions])

  // Only show loading on initial load, not for subsequent fetches
  const isActuallyLoading = isLoading && !projectResponse
  
  if (isActuallyLoading) {
    return (
      <div className={`w-full h-full max-h-full rounded-lg bg-card border border-border flex items-center justify-center ${className}`}>
        <div className="flex flex-col items-center gap-4">
          <Loader />
          <p className="text-muted-foreground text-sm">Loading project configuration...</p>
          <p className="text-xs text-muted-foreground">
            Fetching: {activeProject?.namespace || 'unknown'}/{activeProject?.project || 'unknown'}
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`w-full h-full max-h-full rounded-lg bg-card border border-border flex items-center justify-center ${className}`}>
        <div className="flex flex-col items-center gap-4 p-6 text-center">
          <div className="w-12 h-12 rounded-full bg-destructive/10 flex items-center justify-center">
            <FontIcon type="info" className="w-6 h-6 text-destructive" />
          </div>
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">Failed to load configuration</h3>
            <p className="text-muted-foreground text-sm max-w-md">
              {error instanceof Error ? error.message : 'An unknown error occurred while fetching the project configuration.'}
            </p>
            <button
              onClick={() => refetch()}
              className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors text-sm"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    )
  }

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
            <span className="text-xs text-muted-foreground">Read-only</span>
            <FontIcon type="eye-off" className="w-3 h-3 text-muted-foreground" />
          </div>
        </div>
      </div>

      {/* Editor */}
      <div 
        ref={editorRef} 
        className="config-editor-content flex-1 min-h-0 overflow-auto bg-background custom-scrollbar"
      >
        {/* Fallback content if CodeMirror fails to initialize */}
        {!isInitialized && (
          <div className="p-4 text-sm text-muted-foreground font-mono overflow-auto flex-1 custom-scrollbar">
            <pre className="whitespace-pre-wrap">{formattedConfig}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default ConfigEditor
