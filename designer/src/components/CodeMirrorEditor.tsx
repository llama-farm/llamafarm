import { useEffect, useRef, useState, useMemo } from 'react'
import { useTheme } from '../contexts/ThemeContext'
import Loader from '../common/Loader'
import FontIcon from '../common/FontIcon'

// Dynamic imports for CodeMirror packages
const loadCodeMirrorModules = async () => {
  const [
    { EditorView, lineNumbers, keymap },
    { EditorState, StateEffect },
    { json },
    { defaultKeymap },
    { bracketMatching, indentOnInput, foldGutter, syntaxHighlighting, HighlightStyle },
    { highlightSelectionMatches },
    { tags }
  ] = await Promise.all([
    import('@codemirror/view'),
    import('@codemirror/state'),
    import('@codemirror/lang-json'),
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/search'),
    import('@lezer/highlight')
  ])

  return {
    EditorView,
    lineNumbers,
    keymap,
    EditorState,
    StateEffect,
    json,
    defaultKeymap,
    bracketMatching,
    indentOnInput,
    foldGutter,
    syntaxHighlighting,
    HighlightStyle,
    highlightSelectionMatches,
    tags
  }
}

interface CodeMirrorEditorProps {
  content: string
  className?: string
}

const CodeMirrorEditor: React.FC<CodeMirrorEditorProps> = ({ content, className = '' }) => {
  const { theme } = useTheme()
  const editorRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isInitialized, setIsInitialized] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modules, setModules] = useState<any>(null)

  // Load CodeMirror modules
  useEffect(() => {
    let isMounted = true

    const loadModules = async () => {
      try {
        const loadedModules = await loadCodeMirrorModules()
        if (isMounted) {
          setModules(loadedModules)
          setIsLoading(false)
        }
      } catch (err) {
        console.error('Failed to load CodeMirror modules:', err)
        if (isMounted) {
          setError('Failed to load code editor')
          setIsLoading(false)
        }
      }
    }

    loadModules()

    return () => {
      isMounted = false
    }
  }, [])

  // Create extensions configuration
  const createExtensions = useMemo(() => {
    if (!modules) return () => []

    const {
      EditorView,
      lineNumbers,
      keymap,
      EditorState,
      json,
      defaultKeymap,
      bracketMatching,
      indentOnInput,
      foldGutter,
      syntaxHighlighting,
      HighlightStyle,
      highlightSelectionMatches,
      tags
    } = modules

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

    // Custom dark theme for CodeMirror with the specific background color
    const customDarkTheme = EditorView.theme({
      '&': {
        color: '#f0f6fc',
        backgroundColor: '#10182e',
        height: '100%'
      },
      '.cm-content': {
        padding: '16px',
        minHeight: '100%',
        backgroundColor: '#10182e'
      },
      '.cm-focused': {
        outline: 'none',
        backgroundColor: '#10182e'
      },
      '.cm-editor': {
        borderRadius: '8px',
        height: '100%',
        backgroundColor: '#10182e'
      },
      '.cm-scroller': {
        lineHeight: '1.5',
        height: '100%',
        backgroundColor: '#10182e'
      },
      '.cm-cursor': {
        display: 'none' // Hide cursor in read-only mode
      },
      '.cm-gutters': {
        backgroundColor: '#161b22',
        color: '#7d8590',
        border: 'none',
        borderRight: '1px solid #30363d'
      },
      '.cm-activeLineGutter': {
        backgroundColor: '#161b22'
      },
      '.cm-foldGutter .cm-gutterElement': {
        color: '#7d8590'
      },
      '.cm-selectionBackground': {
        backgroundColor: '#264f78'
      },
      '.cm-scroller::-webkit-scrollbar': {
        width: '12px'
      },
      '.cm-scroller::-webkit-scrollbar-track': {
        backgroundColor: '#1c2028'
      },
      '.cm-scroller::-webkit-scrollbar-thumb': {
        backgroundColor: '#30363d',
        borderRadius: '6px',
        border: '2px solid #10182e'
      },
      '.cm-scroller::-webkit-scrollbar-thumb:hover': {
        backgroundColor: '#484f58'
      }
    }, { dark: true })

    // Custom syntax highlighting for light theme
    const lightHighlightStyle = HighlightStyle.define([
      { tag: tags.keyword, color: '#d73a49' },
      { tag: tags.string, color: '#032f62' },
      { tag: tags.number, color: '#005cc5' },
      { tag: tags.bool, color: '#005cc5' },
      { tag: tags.null, color: '#005cc5' },
      { tag: tags.propertyName, color: '#22863a' },
      { tag: tags.comment, color: '#6a737d', fontStyle: 'italic' },
      { tag: tags.bracket, color: '#24292f' },
      { tag: tags.brace, color: '#24292f' },
      { tag: tags.punctuation, color: '#24292f' }
    ])

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
        ...defaultKeymap.filter((binding: any) => {
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
      theme === 'dark' 
        ? [customDarkTheme, syntaxHighlighting(HighlightStyle.define([
            { tag: tags.keyword, color: '#ff7b72' },
            { tag: tags.string, color: '#a5d6ff' },
            { tag: tags.number, color: '#79c0ff' },
            { tag: tags.bool, color: '#79c0ff' },
            { tag: tags.null, color: '#79c0ff' },
            { tag: tags.propertyName, color: '#7ee787' },
            { tag: tags.comment, color: '#8b949e', fontStyle: 'italic' },
            { tag: tags.bracket, color: '#f0f6fc' },
            { tag: tags.brace, color: '#f0f6fc' },
            { tag: tags.punctuation, color: '#f0f6fc' }
          ]))]
        : [lightTheme, syntaxHighlighting(lightHighlightStyle)]
    ]
  }, [modules, theme])

  // Initialize CodeMirror
  useEffect(() => {
    if (!editorRef.current || !modules || isLoading) return
    
    const { EditorView, EditorState } = modules
    
    // Clean up previous editor if it exists
    if (viewRef.current) {
      viewRef.current.destroy()
      viewRef.current = null
      setIsInitialized(false)
    }

    const extensions = createExtensions()

    const state = EditorState.create({
      doc: content,
      extensions
    })

    const view = new EditorView({
      state,
      parent: editorRef.current
    })

    viewRef.current = view
    setIsInitialized(true)

    return () => {
      if (viewRef.current) {
        viewRef.current.destroy()
        viewRef.current = null
        setIsInitialized(false)
      }
    }
  }, [modules, isLoading, createExtensions]) // Only initialize once when modules are loaded

  // Update content when it changes
  useEffect(() => {
    if (!viewRef.current || !isInitialized || !modules) return
    const currentDoc = viewRef.current.state.doc.toString()
    if (currentDoc !== content) {
      viewRef.current.dispatch({
        changes: {
          from: 0,
          to: viewRef.current.state.doc.length,
          insert: content
        }
      })
    }
  }, [content, isInitialized, modules])

  // Update theme when it changes
  useEffect(() => {
    if (!viewRef.current || !isInitialized || !modules) return
    
    const { StateEffect } = modules
    viewRef.current.dispatch({
      effects: StateEffect.reconfigure.of(createExtensions())
    })
  }, [theme, isInitialized, createExtensions, modules])

  if (isLoading) {
    return (
      <div className={`w-full h-full max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}>
        {/* Header */}
        <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FontIcon type="code" className="w-4 h-4 text-foreground" />
              <h2 className="text-sm font-semibold text-foreground">
                Project Configuration
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Loading editor...</span>
            </div>
          </div>
        </div>

        {/* Loading state */}
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader className="w-8 h-8" />
            <span className="text-sm text-muted-foreground">Loading code editor...</span>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`w-full h-full max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}>
        {/* Header */}
        <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FontIcon type="code" className="w-4 h-4 text-foreground" />
              <h2 className="text-sm font-semibold text-foreground">
                Project Configuration
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-red-500">Error loading editor</span>
            </div>
          </div>
        </div>

        {/* Error fallback with plain text */}
        <div className="flex-1 min-h-0 overflow-auto bg-background dark:bg-[#1c2028] custom-scrollbar">
          <div className="p-4 text-sm text-muted-foreground font-mono overflow-auto flex-1 custom-scrollbar">
            <pre className="whitespace-pre-wrap">{content}</pre>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`w-full h-full max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-card flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FontIcon type="code" className="w-4 h-4 text-foreground" />
            <h2 className="text-sm font-semibold text-foreground">
              Project Configuration
            </h2>
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
        className="config-editor-content flex-1 min-h-0 overflow-auto bg-background dark:bg-[#10182e] custom-scrollbar"
      >
        {/* Fallback content if CodeMirror fails to initialize */}
        {!isInitialized && (
          <div className="p-4 text-sm text-muted-foreground font-mono overflow-auto flex-1 custom-scrollbar">
            <pre className="whitespace-pre-wrap">{content}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default CodeMirrorEditor
