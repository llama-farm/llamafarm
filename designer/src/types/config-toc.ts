/**
 * Table of Contents types for config editor navigation
 */

export interface TOCNode {
  /** Unique identifier for this node */
  id: string

  /** Display label for the TOC item */
  label: string

  /** JSON Pointer (RFC 6902) path to this section */
  jsonPointer: string

  /** Starting line number in YAML (1-indexed) */
  lineStart: number

  /** Ending line number in YAML (1-indexed) */
  lineEnd: number

  /** Nesting depth level (0 = root, max 2 for 3 levels total) */
  level: number

  /** Child nodes */
  children?: TOCNode[]

  /** Whether this node can be collapsed */
  isCollapsible: boolean
}

export interface EditorNavigationAPI {
  /** Scroll to a specific line in the editor */
  scrollToLine: (lineNumber: number) => void

  /** Highlight lines with a fade effect */
  highlightLines: (start: number, end: number, duration?: number) => () => void

  /** Get current scroll position */
  getScrollPosition?: () => number

  /** Get the currently visible line number */
  getCurrentLine?: () => number

  /** Highlight search matches with optional active range */
  highlightSearchMatches?: (
    matches: Array<{ from: number; to: number }>,
    activeIndex?: number | null
  ) => void

  /** Clear all search match highlights */
  clearSearchMatches?: () => void

  /** Reveal a specific search match range */
  revealSearchMatch?: (match: { from: number; to: number }) => void

  /** Focus the editor instance */
  focusEditor?: () => void
}

export interface ConfigStructureResult {
  /** Root TOC nodes */
  nodes: TOCNode[]

  /** Whether parsing was successful */
  success: boolean

  /** Error message if parsing failed */
  error?: string
}
