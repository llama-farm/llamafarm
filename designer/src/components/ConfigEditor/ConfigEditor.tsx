import { lazy, Suspense, useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useFormattedConfig } from '../../hooks/useFormattedConfig'
import { useUpdateProject } from '../../hooks/useProjects'
import { useTheme } from '../../contexts/ThemeContext'
import yaml from 'yaml'
import ConfigLoading from './ConfigLoading'
import ConfigError from './ConfigError'
import ConfigTableOfContents from './ConfigTableOfContents'
import Loader from '../../common/Loader'
import FontIcon from '../../common/FontIcon'
import type { EditorNavigationAPI } from '../../types/config-toc'
import { useConfigStructure } from '../../hooks/useConfigStructure'
import type { TOCNode } from '../../types/config-toc'
import { normalisePointer, parentPointer } from '../../utils/configNavigation'
import { findSearchMatches, type SearchMatch } from '../../utils/searchUtils'

// Lazy load the CodeMirror editor
const CodeMirrorEditor = lazy(
  () => import('../CodeMirrorEditor/CodeMirrorEditor')
)

interface ConfigEditorProps {
  className?: string
  initialPointer?: string | null
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ className = '', initialPointer = null }) => {
  // Get current project info using reactive hook
  const activeProject = useActiveProject()
  const { theme } = useTheme()

  // Use the extracted hook for all data fetching and formatting
  const { formattedConfig, isLoading, error, refetch, projectData } =
    useFormattedConfig()

  // Edit state
  const [editedContent, setEditedContent] = useState<string>(formattedConfig)
  const [isDirty, setIsDirty] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [navigationAPI, setNavigationAPI] = useState<EditorNavigationAPI | null>(null)
  const [pendingPointer, setPendingPointer] = useState<string | null>(null)
  const [resolvedPointer, setResolvedPointer] = useState<string | null>(null)
  const [copyStatus, setCopyStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMatches, setSearchMatches] = useState<SearchMatch[]>([])
  const [activeMatchIndex, setActiveMatchIndex] = useState<number>(-1)
  const searchInputRef = useRef<HTMLInputElement | null>(null)

  const { nodes } = useConfigStructure(editedContent, !isDirty)

  const pointerMap = useMemo(() => {
    const map = new Map<string, TOCNode>()
    const visit = (list: TOCNode[]) => {
      list.forEach(node => {
        if (node.jsonPointer) {
          map.set(normalisePointer(node.jsonPointer), node)
        }
        if (node.children && node.children.length > 0) {
          visit(node.children)
        }
      })
    }
    visit(nodes)
    return map
  }, [nodes])

  useEffect(() => {
    if (typeof initialPointer === 'string') {
      setPendingPointer(initialPointer)
      setResolvedPointer(null)
    } else if (initialPointer == null) {
      setPendingPointer(null)
      setResolvedPointer(null)
    }
  }, [initialPointer])

  useEffect(() => {
    if (!navigationAPI || !pendingPointer) return

    const resolveNode = (pointer: string): TOCNode | null => {
      if (pointerMap.size === 0) return null
      let current: string | null = normalisePointer(pointer)
      const tried = new Set<string>()
      while (current) {
        if (tried.has(current)) break
        tried.add(current)
        const node = pointerMap.get(current)
        if (node) return node
        current = parentPointer(current)
      }
      return pointerMap.get('/') ?? null
    }

    const targetNode = resolveNode(pendingPointer)
    if (!targetNode) {
      return
    }

    const startLine = targetNode.lineStart || 1
    const endLine = targetNode.lineEnd >= startLine ? targetNode.lineEnd : startLine

    navigationAPI.scrollToLine(startLine)
    navigationAPI.highlightLines(startLine, endLine, 2000)
    const pointerToStore = targetNode.jsonPointer
      ? normalisePointer(targetNode.jsonPointer)
      : normalisePointer(pendingPointer)
    setResolvedPointer(pointerToStore)
    setPendingPointer(null)
  }, [navigationAPI, pendingPointer, pointerMap])

  // Update edited content when formatted config changes
  // Guard: Don't reset if user has unsaved changes
  useEffect(() => {
    if (!isDirty) {
      setEditedContent(formattedConfig)
      setSaveError(null) // Clear errors when config reloads
    }
  }, [formattedConfig, isDirty])

  useEffect(() => {
    const handleGlobalFind = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'f') {
        event.preventDefault()
        if (searchInputRef.current) {
          searchInputRef.current.focus()
          searchInputRef.current.select()
        }
      }
    }

    window.addEventListener('keydown', handleGlobalFind)
    return () => window.removeEventListener('keydown', handleGlobalFind)
  }, [])

  // Update project mutation
  const updateProject = useUpdateProject()

  // Only show loading on initial load, not for subsequent fetches
  const isActuallyLoading = isLoading && !projectData

  // Handle content changes
  const handleChange = (newContent: string) => {
    setEditedContent(newContent)
    setIsDirty(newContent !== formattedConfig)
    setSaveError(null) // Clear error on change
  }

  const handleCopy = useCallback(async () => {
    const textToCopy = String(editedContent ?? '')

    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(textToCopy)
      } else if (typeof document !== 'undefined') {
        const textarea = document.createElement('textarea')
        textarea.setAttribute('readonly', '')
        textarea.style.position = 'fixed'
        textarea.style.top = '-1000px'
        textarea.style.left = '-1000px'
        textarea.style.opacity = '0'
        textarea.value = textToCopy

        document.body.appendChild(textarea)
        try {
          textarea.select()
          const successful = document.execCommand('copy')
          if (!successful) {
            throw new Error('execCommand copy failed')
          }
        } finally {
          textarea.remove()
        }
      } else {
        throw new Error('Clipboard API unavailable')
      }

      setCopyStatus('success')
    } catch (copyError) {
      console.error('Failed to copy config:', copyError)
      setCopyStatus('error')
    }
  }, [editedContent])

  useEffect(() => {
    if (copyStatus === 'idle') return

    const timeout = window.setTimeout(() => setCopyStatus('idle'), copyStatus === 'success' ? 2000 : 4000)
    return () => window.clearTimeout(timeout)
  }, [copyStatus])

  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value)
  }, [])

  const handleSearchClear = useCallback(() => {
    setSearchQuery('')
    setSearchMatches([])
    setActiveMatchIndex(-1)
    navigationAPI?.clearSearchMatches?.()
    if (searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [navigationAPI, searchInputRef])

  const focusNextMatch = useCallback(() => {
    if (searchMatches.length === 0) return
    setActiveMatchIndex(prevIndex => {
      if (prevIndex < 0) {
        return 0
      }
      return (prevIndex + 1) % searchMatches.length
    })
  }, [searchMatches.length])

  const focusPreviousMatch = useCallback(() => {
    if (searchMatches.length === 0) return
    setActiveMatchIndex(prevIndex => {
      if (prevIndex < 0) {
        return searchMatches.length - 1
      }
      return (prevIndex - 1 + searchMatches.length) % searchMatches.length
    })
  }, [searchMatches.length])

  const handleSearchKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        event.preventDefault()
        if (event.shiftKey) {
          focusPreviousMatch()
        } else {
          focusNextMatch()
        }
      } else if (event.key === 'Escape') {
        event.preventDefault()
        handleSearchClear()
      }
    },
    [focusNextMatch, focusPreviousMatch, handleSearchClear]
  )

  useEffect(() => {
    const trimmedQuery = searchQuery.trim()
    if (!trimmedQuery) {
      setSearchMatches([])
      setActiveMatchIndex(-1)
      navigationAPI?.clearSearchMatches?.()
      return
    }

    const handle = window.setTimeout(() => {
      const matches = findSearchMatches(editedContent, trimmedQuery)
      setSearchMatches(matches)
      setActiveMatchIndex(prevIndex => {
        if (matches.length === 0) {
          return -1
        }
        if (prevIndex < 0) {
          return 0
        }
        return Math.min(prevIndex, matches.length - 1)
      })
    }, 160)

    return () => window.clearTimeout(handle)
  }, [searchQuery, editedContent, navigationAPI])

  useEffect(() => {
    if (!navigationAPI) return

    if (searchMatches.length === 0) {
      navigationAPI.clearSearchMatches?.()
      return
    }

    const ranges = searchMatches.map(match => ({ from: match.from, to: match.to }))
    navigationAPI.highlightSearchMatches?.(
      ranges,
      activeMatchIndex >= 0 ? activeMatchIndex : null
    )
  }, [navigationAPI, searchMatches, activeMatchIndex])

  useEffect(() => {
    if (!navigationAPI) return
    if (activeMatchIndex < 0 || activeMatchIndex >= searchMatches.length) return

    const match = searchMatches[activeMatchIndex]
    navigationAPI.revealSearchMatch?.({ from: match.from, to: match.to })
  }, [activeMatchIndex, searchMatches, navigationAPI])

  // Handle save
  const handleSave = async () => {
    if (!activeProject || !isDirty) return

    setSaveError(null) // Clear previous errors

    try {
      // Parse YAML first to check for syntax errors
      let configObj
      try {
        configObj = yaml.parse(editedContent)
      } catch (parseError) {
        setSaveError(`YAML syntax error: ${parseError instanceof Error ? parseError.message : 'Invalid YAML syntax'}. Please fix the syntax before saving.`)
        return
      }

      // Update project via API
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: {
          config: configObj
        }
      })

      // Refetch to get latest data
      await refetch()

      setIsDirty(false)
      setSaveError(null) // Clear any previous errors on success
    } catch (err: any) {
      console.error('Failed to save config:', err)

      // Parse and sanitize backend validation errors
      let errorMessage = 'Failed to save configuration'

      if (err.response?.data) {
        const {data} = err.response

        // Handle backend validation errors - sanitize to avoid exposing internal details
        if (data.detail) {
          if (Array.isArray(data.detail)) {
            // Pydantic validation errors - only show user-relevant information
            const backendErrors = data.detail
              .slice(0, 5) // Limit to first 5 errors for UX
              .map((e: any) => {
                const location = e.loc ? e.loc.slice(1).join('.') : 'configuration' // Remove 'body' prefix
                const message = e.msg || e.message || 'Invalid value'
                return `${location}: ${message}`
              })
              .join('; ')

            const moreErrors = data.detail.length > 5 ? ` (and ${data.detail.length - 5} more)` : ''
            errorMessage = `Validation failed: ${backendErrors}${moreErrors}`
          } else if (typeof data.detail === 'string') {
            // Sanitize string errors to remove potential internal paths/traces
            const sanitizedDetail = data.detail
              .replace(/\/[^\s]+\/llamafarm/g, 'llamafarm') // Remove absolute paths
              .replace(/File "[^"]+", /g, '') // Remove file references
            errorMessage = `Error: ${sanitizedDetail}`
          } else {
            errorMessage = 'Invalid configuration format'
          }
        } else if (data.message) {
          errorMessage = data.message
        }
      } else if (err.message) {
        // Client-side error
        errorMessage = `Network error: ${err.message}`
      }

      setSaveError(errorMessage)
    }
  }

  // Handle discard
  const handleDiscard = () => {
    setEditedContent(formattedConfig)
    setIsDirty(false)
    setSaveError(null) // Clear errors on discard
  }

  if (isActuallyLoading) {
    return <ConfigLoading activeProject={activeProject} className={className} />
  }

  if (error) {
    return (
      <ConfigError
        error={error}
        activeProject={activeProject}
        onRetry={refetch}
        className={className}
      />
    )
  }

  return (
    <div className={`flex flex-row h-full w-full overflow-hidden ${className}`}>
      {/* TOC Panel - hidden on mobile */}
      <div className="hidden md:block w-[240px] flex-shrink-0 h-full overflow-hidden">
        <ConfigTableOfContents
          configContent={editedContent}
          navigationAPI={navigationAPI}
          shouldUpdate={!isDirty}
          activePointer={resolvedPointer}
          searchQuery={searchQuery}
          onSearchChange={handleSearchChange}
          onSearchKeyDown={handleSearchKeyDown}
          onNavigatePrevious={focusPreviousMatch}
          onNavigateNext={focusNextMatch}
          onClearSearch={handleSearchClear}
          searchInputRef={searchInputRef}
          searchSummary={{
            total: searchMatches.length,
            activeIndex: activeMatchIndex,
          }}
        />
      </div>

      {/* Editor Panel */}
      <div className="flex-1 min-w-0 h-full">
        <Suspense
          fallback={
            <div className="config-editor w-full h-full min-h-0 max-h-full rounded-lg bg-card border border-border overflow-hidden flex flex-col">
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
                    <span className="text-xs text-muted-foreground">
                      Loading editor...
                    </span>
                  </div>
                </div>
              </div>

              {/* Loading editor */}
              <div className="flex-1 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <Loader className="w-8 h-8" />
                  <span className="text-sm text-muted-foreground">
                    Loading code editor...
                  </span>
                </div>
              </div>
            </div>
          }
        >
          <CodeMirrorEditor
            key={`${activeProject?.namespace}-${activeProject?.project}`}
            content={editedContent}
            className="w-full h-full min-h-0"
            language="yaml"
            theme={theme}
            readOnly={false}
            onChange={handleChange}
            onSave={handleSave}
            onDiscard={handleDiscard}
            onCopy={handleCopy}
            isDirty={isDirty}
            isSaving={updateProject.isPending}
            saveError={saveError}
            copyStatus={copyStatus}
            onEditorReady={setNavigationAPI}
          />
        </Suspense>
      </div>
    </div>
  )
}

export default ConfigEditor

