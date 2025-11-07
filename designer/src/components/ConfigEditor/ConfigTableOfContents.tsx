import React, { useState, useEffect, useRef } from 'react'
import { useConfigStructure } from '../../hooks/useConfigStructure'
import TOCNode from './TOCNode'
import type {
  TOCNode as TOCNodeType,
  EditorNavigationAPI,
} from '../../types/config-toc'
import Loader from '../../common/Loader'
import FontIcon from '../../common/FontIcon'
import { normalisePointer } from '../../utils/configNavigation'

interface ConfigTableOfContentsProps {
  /** YAML config content */
  configContent: string

  /** Navigation API from CodeMirror editor */
  navigationAPI: EditorNavigationAPI | null

  /** Whether to update the TOC structure (usually when not dirty) */
  shouldUpdate?: boolean

  /** Pointer for section that should be highlighted/active */
  activePointer?: string | null

  /** Search query value */
  searchQuery?: string

  /** Handler for search query changes */
  onSearchChange?: (value: string) => void

  /** Handler for search input keyboard shortcuts */
  onSearchKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void

  /** Handler for moving to previous match */
  onNavigatePrevious?: () => void

  /** Handler for moving to next match */
  onNavigateNext?: () => void

  /** Clear the search query */
  onClearSearch?: () => void

  /** Ref for the search input */
  searchInputRef?: React.RefObject<HTMLInputElement>

  /** Summary information for search results */
  searchSummary?: {
    total: number
    activeIndex: number
  }
}

/**
 * Table of contents component for config editor
 * Displays hierarchical structure and allows navigation
 */
const ConfigTableOfContents: React.FC<ConfigTableOfContentsProps> = ({
  configContent,
  navigationAPI,
  shouldUpdate = true,
  activePointer = null,
  searchQuery = '',
  onSearchChange,
  onSearchKeyDown,
  onNavigatePrevious,
  onNavigateNext,
  onClearSearch,
  searchInputRef,
  searchSummary,
}) => {
  // Parse config structure
  const { nodes, success, error } = useConfigStructure(
    configContent,
    shouldUpdate
  )

  // Track the active/selected node
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null)
  const activePointerRef = useRef<string | null>(null)

  // Track if we're in a manual navigation (to prevent auto-scroll from overriding)
  const isManualNavigating = useRef(false)

  const findNodeByPointer = (
    nodeList: TOCNodeType[],
    pointer: string
  ): TOCNodeType | null => {
    for (const node of nodeList) {
      if (node.jsonPointer && normalisePointer(node.jsonPointer) === pointer) {
        return node
      }
      if (node.children) {
        const found = findNodeByPointer(node.children, pointer)
        if (found) return found
      }
    }
    return null
  }

  const findClosestNode = (pointer: string): TOCNodeType | null => {
    let current = pointer
    const seen = new Set<string>()
    while (!seen.has(current)) {
      seen.add(current)
      const found = findNodeByPointer(nodes, current)
      if (found) return found
      if (current === '/') break
      const parts = current.split('/')
      parts.pop()
      current = parts.length <= 1 ? '/' : parts.join('/')
      if (!current.startsWith('/')) current = `/${current}`
    }
    return null
  }

  // Handle navigation to a node
  const handleNavigate = (node: TOCNodeType) => {
    if (!navigationAPI) return

    // Set manual navigation flag
    isManualNavigating.current = true

    // Set as active
    setActiveNodeId(node.id)
    if (node.jsonPointer) {
      activePointerRef.current = normalisePointer(node.jsonPointer)
    } else {
      activePointerRef.current = null
    }

    // Scroll to the line
    navigationAPI.scrollToLine(node.lineStart)

    // Highlight only the first line
    navigationAPI.highlightLines(node.lineStart, node.lineStart, 2500)

    // Clear the manual navigation flag after scroll + animation completes
    setTimeout(() => {
      isManualNavigating.current = false
    }, 3000) // 3 seconds: enough for scroll (500ms) + settle (2500ms)
  }

  useEffect(() => {
    if (!activePointer) {
      activePointerRef.current = null
      return
    }

    const pointer = normalisePointer(activePointer)
    const node = findClosestNode(pointer)
    if (!node) return

    isManualNavigating.current = true
    setActiveNodeId(node.id)
    activePointerRef.current = pointer

    const timeout = setTimeout(() => {
      isManualNavigating.current = false
    }, 3000)

    return () => clearTimeout(timeout)
  }, [activePointer, nodes])

  // Track scroll position and update active section
  useEffect(() => {
    if (!navigationAPI?.getCurrentLine || nodes.length === 0) return

    const handleScroll = () => {
      // Don't update if user manually clicked (let the click handler manage it)
      if (isManualNavigating.current) return

      const currentLine = navigationAPI.getCurrentLine?.() || 1

      if (activePointerRef.current) {
        const node = findClosestNode(activePointerRef.current)
        if (
          node &&
          currentLine >= node.lineStart &&
          currentLine <= node.lineEnd
        ) {
          return
        }
        activePointerRef.current = null
      }

      // Find which node contains the current line
      const findActiveNode = (nodeList: TOCNodeType[]): string | null => {
        for (const node of nodeList) {
          // Check if current line is within this node's range
          if (currentLine >= node.lineStart && currentLine <= node.lineEnd) {
            // If it has children, check them first (more specific)
            if (node.children && node.children.length > 0) {
              const childActive = findActiveNode(node.children)
              if (childActive) return childActive
            }
            return node.id
          }
        }
        return null
      }

      const activeId = findActiveNode(nodes)
      if (activeId && activeId !== activeNodeId) {
        setActiveNodeId(activeId)
      }
    }

    // Set up scroll listener with throttling
    let scrollTimeout: NodeJS.Timeout
    const throttledScroll = () => {
      clearTimeout(scrollTimeout)
      scrollTimeout = setTimeout(handleScroll, 200)
    }

    // Initial check (but only if not manually navigating)
    if (!isManualNavigating.current) {
      handleScroll()
    }

    // Listen to editor scroll events (check less frequently)
    const checkInterval = setInterval(throttledScroll, 1000)

    return () => {
      clearInterval(checkInterval)
      clearTimeout(scrollTimeout)
    }
  }, [navigationAPI, nodes, activeNodeId])

  const renderHeader = () => {
    const trimmedQuery = searchQuery.trim()
    const totalMatches = searchSummary?.total ?? 0
    const activeIndex = searchSummary?.activeIndex ?? -1
    const hasMatches = totalMatches > 0
    const activeDisplay = hasMatches && activeIndex >= 0 ? activeIndex + 1 : 0
    const summaryText = !trimmedQuery
      ? ''
      : hasMatches
        ? `${activeDisplay} of ${totalMatches} matches`
        : 'No matches found'

    const disableNavigation = !hasMatches || activeIndex < 0
    const showSummary = trimmedQuery.length > 0
    const showNavButtons = showSummary && hasMatches

    return (
      <div className="px-4 py-4 border-b border-border bg-card flex-shrink-0">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">Contents</h3>
        </div>

        {onSearchChange && (
          <div className="mt-3 space-y-2">
            <div className="relative flex items-center">
              <FontIcon
                type="search"
                className="absolute left-3 w-3.5 h-3.5 text-muted-foreground"
              />
              <input
                ref={searchInputRef}
                value={searchQuery}
                onChange={event => onSearchChange(event.target.value)}
                onKeyDown={onSearchKeyDown}
                placeholder="Search config"
                className="w-full pl-9 pr-9 py-2 text-xs rounded-md bg-muted border border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 text-foreground placeholder:text-muted-foreground transition"
              />
              {trimmedQuery && onClearSearch && (
                <button
                  type="button"
                  onClick={onClearSearch}
                  className="absolute right-3 p-1.5 rounded-md text-muted-foreground hover:text-foreground"
                  aria-label="Clear search"
                >
                  <FontIcon type="close" className="w-3 h-3" />
                </button>
              )}
            </div>
            {showSummary && (
              <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                <span>{summaryText}</span>
                <span>Enter / Shift+Enter</span>
              </div>
            )}
            {showNavButtons && (
              <div className="flex items-center justify-between gap-2 text-[11px]">
                <button
                  type="button"
                  onClick={onNavigatePrevious}
                  disabled={disableNavigation}
                  className="flex-1 py-1 rounded-md border border-border text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                  aria-label="Previous match"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={onNavigateNext}
                  disabled={disableNavigation}
                  className="flex-1 py-1 rounded-md border border-border text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
                  aria-label="Next match"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // Loading or error states
  if (!success && error) {
    return (
      <div className="h-full w-full bg-card border-l border-t border-b border-border rounded-tl-lg rounded-bl-lg flex flex-col overflow-hidden">
        {renderHeader()}
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-center">
            <p className="text-xs text-muted-foreground">
              Unable to parse config
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (nodes.length === 0) {
    return (
      <div className="h-full w-full bg-card border-l border-t border-b border-border rounded-tl-lg rounded-bl-lg flex flex-col overflow-hidden">
        {renderHeader()}
        <div className="flex-1 flex items-center justify-center p-4">
          <Loader className="w-6 h-6" />
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full bg-card border-l border-t border-b border-border rounded-tl-lg rounded-bl-lg flex flex-col overflow-hidden">
      {/* Header */}
      {renderHeader()}

      {/* TOC tree */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        <nav aria-label="Config sections" className="py-2 pb-24">
          {nodes.map(node => (
            <TOCNode
              key={node.id}
              node={node}
              onNavigate={handleNavigate}
              defaultCollapsed={false} // Top-level nodes start expanded
              isActive={activeNodeId === node.id}
              activeNodeId={activeNodeId}
            />
          ))}
        </nav>
      </div>
    </div>
  )
}

export default ConfigTableOfContents
