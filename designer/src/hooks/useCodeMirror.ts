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
    { bracketMatching, indentOnInput, foldGutter, syntaxHighlighting, HighlightStyle },
    { highlightSelectionMatches },
    { tags },
    { oneDark }
  ] = await Promise.all([
    import('@codemirror/view'),
    import('@codemirror/state'),
    import('@codemirror/lang-json'),
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/search'),
    import('@lezer/highlight'),
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
    syntaxHighlighting,
    HighlightStyle,
    highlightSelectionMatches,
    tags,
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
      syntaxHighlighting,
      HighlightStyle,
      highlightSelectionMatches,
      json,
      keymap,
      defaultKeymap,
      tags,
      oneDark
    } = modules

    const extensions = []

    // Theme setup - apply appropriate theme
    if (defaultConfig.theme === 'dark') {
      extensions.push(oneDark)
    }
    // Light theme uses default CodeMirror styling with our custom overrides

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

    // Add syntax highlighting for both themes
    const customHighlightStyle = HighlightStyle.define([
      // JSON-specific highlighting
      { tag: tags.propertyName, color: defaultConfig.theme === 'dark' ? '#e06c75' : '#d73a49' }, // Red for property names
      { tag: tags.string, color: defaultConfig.theme === 'dark' ? '#98c379' : '#032f62' }, // Green/Blue for strings
      { tag: tags.number, color: defaultConfig.theme === 'dark' ? '#d19a66' : '#005cc5' }, // Orange/Blue for numbers
      { tag: tags.bool, color: defaultConfig.theme === 'dark' ? '#56b6c2' : '#005cc5' }, // Cyan/Blue for booleans
      { tag: tags.null, color: defaultConfig.theme === 'dark' ? '#e06c75' : '#d73a49' }, // Red for null
      { tag: tags.keyword, color: defaultConfig.theme === 'dark' ? '#c678dd' : '#d73a49' }, // Purple/Red for keywords
      { tag: tags.bracket, color: defaultConfig.theme === 'dark' ? '#abb2bf' : '#24292e' }, // Gray for brackets
      { tag: tags.punctuation, color: defaultConfig.theme === 'dark' ? '#abb2bf' : '#24292e' }, // Gray for punctuation
    ])

    extensions.push(syntaxHighlighting(customHighlightStyle))

    // Custom styling extension with proper theme backgrounds
    extensions.push(
      modules.EditorView.theme({
        '&': {
          fontSize: '14px',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : 'hsl(var(--background))',
        },
        '.cm-content': {
          padding: '20px 20px 40px 20px', // Add extra bottom padding to show last line
          minHeight: '100%',
          caretColor: defaultConfig.theme === 'dark' ? '#ffffff' : '#000000',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : '#ffffff',
          margin: '0',
          color: defaultConfig.theme === 'dark' ? '#abb2bf' : '#1f2937',
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
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : '#ffffff',
          overflow: 'auto !important', // CRITICAL: Force overflow with !important
          fontFamily: 'inherit',
          scrollbarWidth: 'thin',
          scrollbarColor: defaultConfig.theme === 'dark' 
            ? '#3a4a5c #10182e' 
            : '#cbd5e1 #f8fafc',
          padding: '0',
          margin: '0',
          paddingBottom: '20px', // Extra space at bottom to ensure last line visibility
        },
        '.cm-gutters': {
          paddingRight: '8px',
          backgroundColor: defaultConfig.theme === 'dark' ? '#10182e' : '#ffffff',
          border: 'none',
        },
        '.cm-lineNumbers .cm-gutterElement': {
          paddingRight: '12px',
          paddingLeft: '8px',
          color: defaultConfig.theme === 'dark' ? '#5c6370' : '#64748b',
          fontSize: '14px',
        },
        // Fix webkit scrollbar selectors (make them separate, not nested)
        '.cm-scroller::-webkit-scrollbar': {
          width: '12px',
          height: '12px',
        },
        '.cm-scroller::-webkit-scrollbar-track': {
          background: defaultConfig.theme === 'dark' ? '#10182e' : '#f8fafc',
          borderRadius: '6px',
        },
        '.cm-scroller::-webkit-scrollbar-thumb': {
          background: defaultConfig.theme === 'dark' ? '#3a4a5c' : '#cbd5e1',
          borderRadius: '6px',
          border: defaultConfig.theme === 'dark' ? '2px solid #10182e' : '2px solid #f8fafc',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:hover': {
          background: defaultConfig.theme === 'dark' ? '#4a5a6c' : '#94a3b8',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:active': {
          background: defaultConfig.theme === 'dark' ? '#5a6a7c' : '#64748b',
        },
        '.cm-scroller::-webkit-scrollbar-corner': {
          background: defaultConfig.theme === 'dark' ? '#10182e' : '#f8fafc',
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

  // Update theme when it changes
  useEffect(() => {
    if (viewRef.current && isInitialized && modules) {
      // Reconfigure the editor with new theme extensions
      viewRef.current.reconfigure(createExtensions)
    }
  }, [defaultConfig.theme, viewRef.current, isInitialized, modules, createExtensions])


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
