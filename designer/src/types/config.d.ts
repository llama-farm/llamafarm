/**
 * Type definitions for configuration-related components and hooks
 */

export interface ProjectConfig {
  [key: string]: any
}

export interface FormattedConfigData {
  project: string
  namespace: string
  config: ProjectConfig
  metadata: {
    lastUpdated?: string
    source: string
  }
}

export interface UseFormattedConfigReturn {
  formattedConfig: string
  isLoading: boolean
  error: Error | null
  refetch: () => void
  projectData: any | null
}

export interface ConfigEditorState {
  isActuallyLoading: boolean
  hasError: boolean
  hasData: boolean
}

export interface ConfigLoadingProps {
  activeProject: { project: string; namespace: string } | null
  className?: string
}

export interface ConfigErrorProps {
  error: Error | null
  activeProject: { project: string; namespace: string } | null
  onRetry?: () => void
  className?: string
}

export interface EditorToolbarProps {
  activeProject: { project: string; namespace: string } | null
  readOnly?: boolean
  onRefresh?: () => void
  onSave?: () => void
  onDiscard?: () => void
  isDirty?: boolean
  isSaving?: boolean
  saveError?: string | null
}

export interface EditorContentProps {
  editorRef: React.RefObject<HTMLDivElement>
  className?: string
}

export interface EditorFallbackProps {
  content: string
  isInitialized: boolean
  className?: string
}
