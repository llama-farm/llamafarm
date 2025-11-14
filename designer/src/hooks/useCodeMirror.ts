import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useTheme } from '../contexts/ThemeContext'
import type {
  CodeMirrorModules,
  CodeMirrorInstance,
  UseCodeMirrorReturn,
  CodeMirrorConfig,
} from '../types/codemirror'
import type { EditorNavigationAPI } from '../types/config-toc'

// Dynamic imports for CodeMirror packages
const loadCodeMirrorModules = async (): Promise<CodeMirrorModules> => {
  const [
    { EditorView, lineNumbers, keymap, Decoration },
    { EditorState, StateEffect, StateField },
    { json },
    { yaml },
    { defaultKeymap },
    {
      bracketMatching,
      indentOnInput,
      foldGutter,
      syntaxHighlighting,
      HighlightStyle,
    },
    { highlightSelectionMatches },
    { tags },
    { oneDark },
  ] = await Promise.all([
    import('@codemirror/view'),
    import('@codemirror/state'),
    import('@codemirror/lang-json'),
    import('@codemirror/lang-yaml'),
    import('@codemirror/commands'),
    import('@codemirror/language'),
    import('@codemirror/search'),
    import('@lezer/highlight'),
    import('@codemirror/theme-one-dark'),
  ])

  return {
    EditorView,
    lineNumbers,
    keymap,
    EditorState,
    StateEffect,
    StateField,
    Decoration,
    json,
    yaml,
    defaultKeymap,
    bracketMatching,
    indentOnInput,
    foldGutter,
    syntaxHighlighting,
    HighlightStyle,
    highlightSelectionMatches,
    tags,
    oneDark,
    linter: null as any, // Not used due to compatibility issues
    lintGutter: null as any, // Not used due to compatibility issues
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
    onChange: config.onChange,
    ...config,
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

  // Create a persistent StateField for managing highlights
  const highlightEffects = useMemo(() => {
    if (!modules) return null

    // Effect to add a highlight
    const addHighlight = modules.StateEffect.define<{
      from: number
      to: number
    }>()
    // Effect to clear highlights
    const clearHighlight = modules.StateEffect.define()

    // StateField to manage highlight decorations
    const highlightField = modules.StateField.define({
      create: () => modules.Decoration.none,
      update: (decorations: any, tr: any) => {
        // Check for our custom effects
        for (const effect of tr.effects) {
          if (effect.is(addHighlight)) {
            const mark = modules.Decoration.mark({ class: 'cm-toc-highlight' })
            const decoration = mark.range(effect.value.from, effect.value.to)
            return modules.Decoration.set([decoration])
          }
          if (effect.is(clearHighlight)) {
            return modules.Decoration.none
          }
        }
        // Map decorations through document changes
        return decorations.map(tr.changes)
      },
      provide: (f: any) => modules.EditorView.decorations.from(f),
    })

    return { highlightField, addHighlight, clearHighlight }
  }, [modules])

  const searchHighlightEffects = useMemo(() => {
    if (!modules) return null

    const setHighlights = modules.StateEffect.define<{
      ranges: Array<{ from: number; to: number }>
      activeIndex?: number | null
    }>()
    const clearHighlights = modules.StateEffect.define<null>()

    const highlightField = modules.StateField.define({
      create: () => modules.Decoration.none,
      update: (decorations: any, tr: any) => {
        let currentDecorations = decorations.map(tr.changes)

        for (const effect of tr.effects) {
          if (effect.is(setHighlights)) {
            const { ranges, activeIndex = null } = effect.value
            if (!Array.isArray(ranges) || ranges.length === 0) {
              return modules.Decoration.none
            }

            const baseDecoration = modules.Decoration.mark({
              class: 'cm-search-highlight',
            })
            const activeDecoration = modules.Decoration.mark({
              class: 'cm-search-highlight-active',
            })

            const decorationsList = ranges
              .filter(
                range =>
                  typeof range?.from === 'number' &&
                  typeof range?.to === 'number' &&
                  range.to > range.from
              )
              .sort((a, b) => a.from - b.from)
              .map((range, index) => {
                const decoration =
                  activeIndex === index ? activeDecoration : baseDecoration
                return decoration.range(range.from, range.to)
              })

            return modules.Decoration.set(decorationsList, true)
          }

          if (effect.is(clearHighlights)) {
            return modules.Decoration.none
          }
        }

        return currentDecorations
      },
      provide: (f: any) => modules.EditorView.decorations.from(f),
    })

    return {
      highlightField,
      setHighlights,
      clearHighlights,
    }
  }, [modules])

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
      yaml,
      keymap,
      defaultKeymap,
      tags,
    } = modules

    const extensions = []

    // Add the persistent highlight field first
    if (highlightEffects) {
      extensions.push(highlightEffects.highlightField)
    }

    if (searchHighlightEffects) {
      extensions.push(searchHighlightEffects.highlightField)
    }

    // We provide our own theme and highlight styles for both modes to ensure
    // consistent, non-red/green-forward palettes. Avoid pushing oneDark so its
    // highlight colors don't override our custom hues.

    // Language support
    if (defaultConfig.language === 'json') {
      extensions.push(json())
    } else if (defaultConfig.language === 'yaml') {
      extensions.push(yaml())

      // Real-time linting disabled due to CodeMirror module compatibility issues
      // Validation happens on save instead (see ConfigEditor component)
      // The linter extension causes: "Unrecognized extension value in extension set"
      // This is a known issue with dynamic module loading in CodeMirror
      //
      // Alternative: Users will see validation errors when they click Save
      // which prevents saving invalid configs
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
      // shadcn/tailwind-inspired palette mapping, favoring lighter blues/teals/pinks/purples
      // property names → light blue
      {
        tag: tags.propertyName,
        color: defaultConfig.theme === 'dark' ? '#93c5fd' : '#60a5fa',
      },
      // strings → light teal
      {
        tag: tags.string,
        color: defaultConfig.theme === 'dark' ? '#5eead4' : '#2dd4bf',
      },
      // numbers → light orange/amber
      {
        tag: tags.number,
        color: defaultConfig.theme === 'dark' ? '#fdba74' : '#fb923c',
      },
      // booleans → violet (avoid strong green)
      {
        tag: tags.bool,
        color: defaultConfig.theme === 'dark' ? '#c4b5fd' : '#a78bfa',
      },
      // null → pink (avoid red)
      {
        tag: tags.null,
        color: defaultConfig.theme === 'dark' ? '#f9a8d4' : '#f472b6',
      },
      // keywords → purple/violet
      {
        tag: tags.keyword,
        color: defaultConfig.theme === 'dark' ? '#a78bfa' : '#8b5cf6',
      },
      // punctuation / brackets → slate
      {
        tag: tags.bracket,
        color: defaultConfig.theme === 'dark' ? '#94a3b8' : '#475569',
      },
      {
        tag: tags.punctuation,
        color: defaultConfig.theme === 'dark' ? '#94a3b8' : '#475569',
      },
    ])

    extensions.push(syntaxHighlighting(customHighlightStyle))

    // Add onChange listener if provided
    if (defaultConfig.onChange && !defaultConfig.readOnly) {
      const updateListener = modules.EditorView.updateListener.of(
        (update: any) => {
          if (update.docChanged) {
            const newContent = update.state.doc.toString()
            defaultConfig.onChange!(newContent)
          }
        }
      )
      extensions.push(updateListener)
    }

    // Custom styling extension with proper theme backgrounds
    extensions.push(
      modules.EditorView.theme({
        '&': {
          fontSize: '14px',
          fontFamily:
            'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
          backgroundColor:
            defaultConfig.theme === 'dark'
              ? '#10182e'
              : 'hsl(var(--background))',
        },
        '.cm-content': {
          padding: '20px 20px 20px 20px',
          paddingBottom: '40vh', // Enough padding so last line can scroll to top
          minHeight: '100%',
          caretColor: defaultConfig.theme === 'dark' ? '#ffffff' : '#000000',
          backgroundColor:
            defaultConfig.theme === 'dark' ? '#10182e' : '#ffffff',
          margin: '0',
          color: defaultConfig.theme === 'dark' ? '#abb2bf' : '#1f2937',
        },
        '.cm-focused': {
          outline: 'none',
        },
        '.cm-editor': {
          height: '100%',
          maxHeight: '100%',
          backgroundColor:
            defaultConfig.theme === 'dark'
              ? '#10182e'
              : 'hsl(var(--background))',
        },
        '.cm-scroller': {
          height: '100%',
          maxHeight: '100%',
          backgroundColor:
            defaultConfig.theme === 'dark' ? '#10182e' : '#ffffff',
          overflow: 'auto !important',
          fontFamily: 'inherit',
          scrollbarWidth: 'thin',
          scrollbarColor:
            defaultConfig.theme === 'dark'
              ? '#3a4a5c #10182e'
              : '#cbd5e1 #f8fafc',
          padding: '0',
          margin: '0',
        },
        '.cm-gutters': {
          paddingRight: '8px',
          backgroundColor:
            defaultConfig.theme === 'dark' ? '#0f172a' : '#e2e8f0', // slate blue theme
          color: defaultConfig.theme === 'dark' ? '#94a3b8' : '#475569', // slate for numbers
          borderRight:
            defaultConfig.theme === 'dark'
              ? '1px solid #1e293b'
              : '1px solid #cbd5e1',
        },
        '.cm-lineNumbers .cm-gutterElement': {
          paddingRight: '12px',
          paddingLeft: '8px',
          color: defaultConfig.theme === 'dark' ? '#94a3b8' : '#475569',
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
          border:
            defaultConfig.theme === 'dark'
              ? '2px solid #10182e'
              : '2px solid #f8fafc',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:hover': {
          background: defaultConfig.theme === 'dark' ? '#4a5a6c' : '#94a3b8',
        },
        '.cm-scroller::-webkit-scrollbar-thumb:active': {
          background: defaultConfig.theme === 'dark' ? '#5a6a7c' : '#64748b',
        },
        '.cm-scroller::-webkit-scrollbar-corner': {
          background: defaultConfig.theme === 'dark' ? '#10182e' : '#f8fafc',
        },
      })
    )

    return extensions
  }, [modules, defaultConfig, highlightEffects, searchHighlightEffects])

  // Initialize CodeMirror editor
  useEffect(() => {
    if (!modules || !editorRef.current || isInitialized) return

    const { EditorView, EditorState } = modules

    try {
      const state = EditorState.create({
        doc: content,
        extensions: createExtensions,
      })

      const view = new EditorView({
        state,
        parent: editorRef.current,
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
            effects: modules.StateEffect.reconfigure.of(newExtensions),
          })
        },
        focus: () => view.focus(),
        getContent: () => view.state.doc.toString(),
        setContent: (newContent: string) => {
          view.dispatch({
            changes: {
              from: 0,
              to: view.state.doc.length,
              insert: newContent,
            },
          })
        },
      }

      viewRef.current = instance
      setIsInitialized(true)
      setError(null)
    } catch (err) {
      console.error('Failed to initialize CodeMirror:', err)
      setError('Failed to initialize editor')
    }
  }, [
    modules,
    createExtensions,
    content,
    defaultConfig.readOnly,
    isInitialized,
  ])

  // Update content when it changes
  useEffect(() => {
    if (viewRef.current && isInitialized) {
      const currentContent = viewRef.current.getContent()
      if (currentContent !== content) {
        try {
          viewRef.current.setContent(content || '')
        } catch (err) {
          console.error('Failed to update editor content:', err)
          setError('Failed to update editor content')
          viewRef.current.destroy()
          viewRef.current = null
        }
      }
    }
  }, [content, isInitialized])

  // Update theme when it changes
  useEffect(() => {
    if (viewRef.current && isInitialized && modules) {
      // Reconfigure the editor with new theme extensions
      viewRef.current.reconfigure(createExtensions)
    }
  }, [
    defaultConfig.theme,
    viewRef.current,
    isInitialized,
    modules,
    createExtensions,
  ])

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

  // Navigation API for TOC and other components
  const navigationAPI: EditorNavigationAPI | null = useMemo(() => {
    if (!viewRef.current || !isInitialized || !modules) return null

    // Track active animation for cancellation
    let activeAnimationId: number | null = null

    return {
      scrollToLine: (lineNumber: number) => {
        const view = viewRef.current?.view
        if (!view) return

        try {
          // Cancel any existing animation
          if (activeAnimationId !== null) {
            cancelAnimationFrame(activeAnimationId)
            activeAnimationId = null
          }

          // Get the line (1-indexed input)
          const lineNum = Math.max(
            1,
            Math.min(lineNumber, view.state.doc.lines)
          )
          const line = view.state.doc.line(lineNum)
          const pos = line.from

          // Update selection immediately (cursor follows immediately)
          view.dispatch({
            selection: { anchor: pos },
          })

          // Get scroll container and calculate positions
          const scrollContainer = view.scrollDOM
          const startScrollTop = scrollContainer.scrollTop

          // Calculate target position using CodeMirror's coordinate system
          const lineBlock = view.lineBlockAt(pos)
          const targetScrollTop = Math.max(0, lineBlock.top - 20) // 20px margin from top

          // If already at target (or very close), no need to animate
          if (Math.abs(startScrollTop - targetScrollTop) < 2) {
            return
          }

          // Animation parameters
          const duration = 350 // milliseconds
          const startTime = performance.now()

          // Ease-in-out cubic easing function
          const easeInOutCubic = (t: number): number => {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
          }

          // Animation loop
          const animate = (currentTime: number) => {
            const elapsed = currentTime - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easedProgress = easeInOutCubic(progress)

            // Calculate current scroll position
            const currentScrollTop =
              startScrollTop +
              (targetScrollTop - startScrollTop) * easedProgress

            // Apply scroll position
            scrollContainer.scrollTop = currentScrollTop

            // Continue animation if not complete
            if (progress < 1) {
              activeAnimationId = requestAnimationFrame(animate)
            } else {
              activeAnimationId = null
            }
          }

          // Start animation
          activeAnimationId = requestAnimationFrame(animate)
        } catch (err) {
          console.error('Failed to scroll to line:', err)
          // Cleanup animation on error
          if (activeAnimationId !== null) {
            cancelAnimationFrame(activeAnimationId)
            activeAnimationId = null
          }
        }
      },

      highlightLines: (start: number, end: number, duration = 2500) => {
        const view = viewRef.current?.view
        if (!view || !modules || !highlightEffects) return () => {}

        const maxLine = view.state.doc.lines
        const startLine = Math.min(Math.max(1, start), maxLine)
        const endLine = Math.min(Math.max(startLine, end), maxLine)

        try {
          const fromPos = view.state.doc.line(startLine).from
          const toPos = view.state.doc.line(
            Math.min(endLine, view.state.doc.lines)
          ).to

          // Add highlight using our persistent StateField
          view.dispatch({
            effects: highlightEffects.addHighlight.of({
              from: fromPos,
              to: toPos,
            }),
          })

          // Clear highlight after duration (let CSS animation complete)
          const timeoutId = setTimeout(() => {
            if (viewRef.current?.view && highlightEffects) {
              viewRef.current.view.dispatch({
                effects: highlightEffects.clearHighlight.of(null),
              })
            }
          }, duration)
          return () => {
            clearTimeout(timeoutId)
            if (viewRef.current?.view && highlightEffects) {
              viewRef.current.view.dispatch({
                effects: highlightEffects.clearHighlight.of(null),
              })
            }
          }
        } catch (err) {
          console.error('Failed to highlight lines:', err)
          return () => {}
        }
      },

      getCurrentLine: () => {
        const view = viewRef.current?.view
        if (!view) return 1

        try {
          // Get the current visible range
          const { scrollTop } = view.scrollDOM
          const lineHeight = view.defaultLineHeight
          const approxLine = Math.floor(scrollTop / lineHeight) + 1

          return Math.max(1, Math.min(approxLine, view.state.doc.lines))
        } catch {
          return 1
        }
      },

      highlightSearchMatches: (
        matches: Array<{ from: number; to: number }>,
        activeIndex: number | null = null
      ) => {
        const view = viewRef.current?.view
        if (!view || !modules || !searchHighlightEffects) return

        if (!Array.isArray(matches) || matches.length === 0) {
          view.dispatch({
            effects: searchHighlightEffects.clearHighlights.of(null),
          })
          return
        }

        const validMatches = matches
          .filter(
            match =>
              typeof match?.from === 'number' &&
              typeof match?.to === 'number' &&
              match.to > match.from
          )
          .sort((a, b) => a.from - b.from)

        if (validMatches.length === 0) {
          view.dispatch({
            effects: searchHighlightEffects.clearHighlights.of(null),
          })
          return
        }

        const clampedActive =
          typeof activeIndex === 'number'
            ? Math.max(0, Math.min(activeIndex, validMatches.length - 1))
            : null

        view.dispatch({
          effects: searchHighlightEffects.setHighlights.of({
            ranges: validMatches,
            activeIndex: clampedActive,
          }),
        })
      },

      clearSearchMatches: () => {
        const view = viewRef.current?.view
        if (!view || !modules || !searchHighlightEffects) return

        view.dispatch({
          effects: searchHighlightEffects.clearHighlights.of(null),
        })
      },

      revealSearchMatch: (match: { from: number; to: number }) => {
        const view = viewRef.current?.view
        if (!view || !modules) return

        try {
          const from = Math.max(
            0,
            Math.min(match.from ?? 0, view.state.doc.length)
          )
          const to = Math.max(
            from,
            Math.min(match.to ?? from, view.state.doc.length)
          )

          // Update selection immediately
          view.dispatch({
            selection: { anchor: from, head: to },
          })

          // Cancel any existing animation
          if (activeAnimationId !== null) {
            cancelAnimationFrame(activeAnimationId)
            activeAnimationId = null
          }

          // Get scroll container and calculate positions
          const scrollContainer = view.scrollDOM
          const startScrollTop = scrollContainer.scrollTop
          const viewportHeight = scrollContainer.clientHeight

          // Calculate target position - center the match in viewport
          const matchBlock = view.lineBlockAt(from)
          const targetScrollTop = Math.max(
            0,
            matchBlock.top - viewportHeight / 2 + matchBlock.height / 2
          )

          // If already at target (or very close), no need to animate
          if (Math.abs(startScrollTop - targetScrollTop) < 2) {
            return
          }

          // Animation parameters (same as scrollToLine)
          const duration = 350 // milliseconds
          const startTime = performance.now()

          // Ease-in-out cubic easing function
          const easeInOutCubic = (t: number): number => {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
          }

          // Animation loop
          const animate = (currentTime: number) => {
            const elapsed = currentTime - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easedProgress = easeInOutCubic(progress)

            // Calculate current scroll position
            const currentScrollTop =
              startScrollTop +
              (targetScrollTop - startScrollTop) * easedProgress

            // Apply scroll position
            scrollContainer.scrollTop = currentScrollTop

            // Continue animation if not complete
            if (progress < 1) {
              activeAnimationId = requestAnimationFrame(animate)
            } else {
              activeAnimationId = null
            }
          }

          // Start animation
          activeAnimationId = requestAnimationFrame(animate)
        } catch (err) {
          console.error('Failed to reveal search match:', err)
          // Cleanup animation on error
          if (activeAnimationId !== null) {
            cancelAnimationFrame(activeAnimationId)
            activeAnimationId = null
          }
        }
      },

      focusEditor: () => {
        viewRef.current?.focus()
      },
    }
  }, [
    viewRef,
    isInitialized,
    modules,
    highlightEffects,
    searchHighlightEffects,
  ])

  return {
    editorRef,
    viewRef,
    isLoading,
    isInitialized,
    error,
    modules,
    destroy,
    reconfigure,
    navigationAPI,
  }
}
