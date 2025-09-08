import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useTheme } from '../contexts/ThemeContext'
import type { CodeMirrorModules, CodeMirrorInstance, UseCodeMirrorReturn, CodeMirrorConfig } from '../types/codemirror'

// Dynamic imports for CodeMirror packages
const loadCodeMirrorModules = async (): Promise<CodeMirrorModules> => {
  const [
    { EditorView, lineNumbers, keymap },
    { EditorState, StateEffect },
    { json },
    { defaultKeymap },
    { bracketMatching, indentOnInput, foldGutter },
    { highlightSelectionMatches },
    { oneDark }
  ] = await Promise.all([
    import('@codemirror/view'),
    import('@codemirror/state'),
    import('@codemirror/lang-json'),
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/search'),
    import('@codemirror/theme-one-dark')
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
    highlightSelectionMatches,
    oneDark
  }
}

/**
 * Custom hook for CodeMirror editor initialization and management
 * Handles all the editor lifecycle and provides clean interface
 */
export function useCodeMirror(
  content: string, 
  config: CodeMirrorConfig = {}
): UseCodeMirrorReturn {
  const { theme } = useTheme()
  const editorRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<CodeMirrorInstance | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isInitialized, setIsInitialized] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modules, setModules] = useState<CodeMirrorModules | null>(null)

  // Default configuration
  const defaultConfig: CodeMirrorConfig = {
    readOnly: true,
    lineNumbers: true,
    foldGutter: true,
    highlightSelectionMatches: true,
    theme: theme,
    language: 'json',
    tabSize: 2,
    indentUnit: 2,
    ...config
  }

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
    if (!modules) return []

    const {
      lineNumbers: lineNumbersExt,
      foldGutter: foldGutterExt,
      bracketMatching,
      indentOnInput,
      highlightSelectionMatches,
      json,
      keymap,
      defaultKeymap,
      oneDark
    } = modules

    const extensions = []

    // Theme setup - add dark theme first if needed
    if (defaultConfig.theme === 'dark') {
      extensions.push(oneDark)
    }

    // Language support
    if (defaultConfig.language === 'json') {
      extensions.push(json())
    }

    // Basic editing
    extensions.push(bracketMatching())
    if (!defaultConfig.readOnly) {
      extensions.push(indentOnInput())
    }
    extensions.push(keymap.of(defaultKeymap))

    // Read-only configuration
    if (defaultConfig.readOnly) {
      extensions.push(modules.EditorView.editable.of(false))
    }

    // Optional features
    if (defaultConfig.lineNumbers) {
      extensions.push(lineNumbersExt())
    }

    if (defaultConfig.foldGutter) {
      extensions.push(foldGutterExt())
    }

    if (defaultConfig.highlightSelectionMatches) {
      extensions.push(highlightSelectionMatches())
    }

    // Custom styling extension with proper dark theme background
    extensions.push(
      modules.EditorView.theme({
        '&': {
          fontSize: '14px',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--background))',
        },
        '.cm-content': {
          padding: '16px',
          minHeight: '100%',
          caretColor: defaultConfig.theme === 'dark' ? '#ffffff' : '#000000',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--background))',
        },
        '.cm-focused': {
          outline: 'none'
        },
        '.cm-editor': {
          height: '100%',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--background))',
        },
        '.cm-scroller': {
          height: '100%',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--background))',
          overflow: 'auto !important', // CRITICAL: Force overflow with !important
          fontFamily: 'inherit',
          scrollbarWidth: 'thin',
          scrollbarColor: defaultConfig.theme === 'dark' 
            ? '#3a4a5c #10182e' 
            : 'hsl(var(--border)) hsl(var(--muted))',
        },
        // Fix webkit scrollbar selectors (make them separate, not nested)
        '.cm-scroller::-webkit-scrollbar': {
          width: '12px',
          height: '12px',
        },
        '.cm-scroller::-webkit-scrollbar-track': {
          background: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--muted))',
          borderRadius: '6px',
        },
        '.cm-scroller::-webkit-scrollbar-thumb': {
          background: defaultConfig.theme === 'dark' ? '#3a4a5c' : 'hsl(var(--border))',
          borderRadius: '6px',
          border: defaultConfig.theme === 'dark' ? '2px solid #10182e' : '2px solid hsl(var(--muted))',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:hover': {
          background: defaultConfig.theme === 'dark' ? '#4a5a6c' : 'hsl(var(--muted-foreground) / 0.3)',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:active': {
          background: defaultConfig.theme === 'dark' ? '#5a6a7c' : 'hsl(var(--muted-foreground) / 0.4)',
        },
        '.cm-scroller::-webkit-scrollbar-corner': {
          background: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--muted))',
        }
      })
    )

    return extensions
  }, [modules, defaultConfig])

  // Initialize CodeMirror editor
  useEffect(() => {
    if (!modules || !editorRef.current || isInitialized) return

    const { EditorView, EditorState } = modules

    try {
      const state = EditorState.create({
        doc: content,
        extensions: createExtensions
      })

      const view = new EditorView({
        state,
        parent: editorRef.current
      })

      // Create our instance wrapper
      const instance: CodeMirrorInstance = {
        view,
        state,
        destroy: () => {
          view.destroy()
          setIsInitialized(false)
        },
        reconfigure: (newExtensions: any[]) => {
          view.dispatch({
            effects: modules.StateEffect.reconfigure.of(newExtensions)
          })
        },
        focus: () => view.focus(),
        getContent: () => view.state.doc.toString(),
        setContent: (newContent: string) => {
          view.dispatch({
            changes: {
              from: 0,
              to: view.state.doc.length,
              insert: newContent
            }
          })
        }
      }

      viewRef.current = instance
      setIsInitialized(true)
      setError(null)
    } catch (err) {
      console.error('Failed to initialize CodeMirror:', err)
      setError('Failed to initialize editor')
    }
  }, [modules, createExtensions, content, defaultConfig.readOnly, isInitialized])

  // Update content when it changes
  useEffect(() => {
    if (viewRef.current && isInitialized) {
      const currentContent = viewRef.current.getContent()
      if (currentContent !== content) {
        viewRef.current.setContent(content)
      }
    }
  }, [content, isInitialized])


  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (viewRef.current) {
        viewRef.current.destroy()
        viewRef.current = null
      }
    }
  }, [])

  // Memoized callbacks
  const destroy = useCallback(() => {
    if (viewRef.current) {
      viewRef.current.destroy()
      viewRef.current = null
    }
  }, [])

  const reconfigure = useCallback((newExtensions: any[]) => {
    if (viewRef.current) {
      viewRef.current.reconfigure(newExtensions)
    }
  }, [])

  return {
    editorRef,
    viewRef,
    isLoading,
    isInitialized,
    error,
    modules,
    destroy,
    reconfigure
  }
}
