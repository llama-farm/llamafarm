import type { EditorView } from '@codemirror/view'
import type { EditorState, StateEffect } from '@codemirror/state'
import type { HighlightStyle } from '@codemirror/language'
import type { tags } from '@lezer/highlight'
import type { EditorNavigationAPI } from './config-toc'

/**
 * Type definition for dynamically loaded CodeMirror modules
 */
export interface CodeMirrorModules {
  EditorView: typeof EditorView
  lineNumbers: any // Extension function
  keymap: any // Extension function
  EditorState: typeof EditorState
  StateEffect: typeof StateEffect
  StateField: any // StateField for decorations
  Decoration: any // Decoration for highlighting
  json: any // Language extension
  yaml: any // Language extension
  defaultKeymap: any // Keymap array
  bracketMatching: any // Extension function
  indentOnInput: any // Extension function
  foldGutter: any // Extension function
  syntaxHighlighting: any // Extension function
  HighlightStyle: typeof HighlightStyle
  highlightSelectionMatches: any // Extension function
  tags: typeof tags
  oneDark: any // Dark theme extension
  linter: any // Linting function
  lintGutter: any // Lint gutter extension
}

/**
 * Type definition for a CodeMirror editor instance
 */
export interface CodeMirrorInstance {
  view: EditorView
  state: EditorState
  destroy: () => void
  reconfigure: (extensions: any[]) => void
  focus: () => void
  getContent: () => string
  setContent: (content: string) => void
}

/**
 * Props for CodeMirror editor components
 */
export interface CodeMirrorEditorProps {
  content: string
  className?: string
  onChange?: (content: string) => void
  readOnly?: boolean
  language?: 'json' | 'yaml' | 'javascript' | 'typescript'
  theme?: 'light' | 'dark'
  onSave?: () => void
  onDiscard?: () => void
  onCopy?: () => void
  isDirty?: boolean
  isSaving?: boolean
  saveError?: string | null
  copyStatus?: 'idle' | 'success' | 'error'
  onEditorReady?: (api: EditorNavigationAPI) => void
}

/**
 * Return type for useCodeMirror hook
 */
export interface UseCodeMirrorReturn {
  editorRef: React.RefObject<HTMLDivElement>
  viewRef: React.MutableRefObject<CodeMirrorInstance | null>
  isLoading: boolean
  isInitialized: boolean
  error: string | null
  modules: CodeMirrorModules | null
  destroy: () => void
  reconfigure: (extensions: any[]) => void
  navigationAPI: EditorNavigationAPI | null
}

/**
 * Configuration options for CodeMirror editor
 */
export interface CodeMirrorConfig {
  readOnly?: boolean
  lineNumbers?: boolean
  foldGutter?: boolean
  highlightSelectionMatches?: boolean
  theme?: 'light' | 'dark'
  language?: string
  tabSize?: number
  indentUnit?: number
  onChange?: (content: string) => void
}
