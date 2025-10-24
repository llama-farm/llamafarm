import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useUpdateProject, projectKeys } from './useProjects'

/**
 * Types for parser and extractor rows in the UI
 */
export type ParserRow = {
  id: string
  name: string
  priority: number
  include: string
  exclude: string
  summary: string
  config?: Record<string, unknown>
}

export type ExtractorRow = {
  id: string
  name: string
  priority: number
  applyTo: string
  summary: string
  config?: Record<string, unknown>
}

/**
 * Map legacy high-number priorities to new low-number scale
 */
const migratePriority = (value: unknown): number => {
  if (typeof value !== 'number') return 50
  if (value >= 100) return Math.max(1, Math.floor(value / 10))
  return value
}

/**
 * Conversion functions between YAML config and UI row formats
 */
export const ragStrategyConverters = {
  /**
   * Convert YAML parser config to UI ParserRow format
   */
  yamlToParserRow: (yamlParser: any, index: number): ParserRow => {
    const patterns = yamlParser.file_include_patterns || []
    const includeStr = Array.isArray(patterns) ? patterns.join(', ') : ''

    return {
      id: `parser-${index}-${yamlParser.type}`,
      name: yamlParser.type || 'UnknownParser',
      priority: migratePriority(yamlParser.priority),
      include: includeStr,
      exclude: '', // Not stored in YAML
      summary: '', // UI-generated
      config: yamlParser.config || {},
    }
  },

  /**
   * Convert YAML extractor config to UI ExtractorRow format
   */
  yamlToExtractorRow: (yamlExtractor: any, index: number): ExtractorRow => {
    const patterns = yamlExtractor.file_include_patterns || []
    const applyToStr = Array.isArray(patterns) ? patterns.join(', ') : ''

    return {
      id: `extractor-${index}-${yamlExtractor.type}`,
      name: yamlExtractor.type || 'UnknownExtractor',
      priority: migratePriority(yamlExtractor.priority),
      applyTo: applyToStr,
      summary: '', // UI-generated
      config: yamlExtractor.config || {},
    }
  },

  /**
   * Convert ParserRow to YAML-compliant parser config
   */
  parserRowToYaml: (row: ParserRow) => {
    // Parse include patterns string to array
    const patterns = row.include
      .split(',')
      .map(p => p.trim())
      .filter(Boolean)

    return {
      type: row.name,
      config: row.config || {},
      file_include_patterns: patterns.length > 0 ? patterns : undefined,
      priority: row.priority ?? 50,
    }
  },

  /**
   * Convert ExtractorRow to YAML-compliant extractor config
   */
  extractorRowToYaml: (row: ExtractorRow) => {
    const yamlConfig: any = {
      type: row.name,
      config: row.config || {},
      priority: row.priority ?? 50,
    }

    // Parse applyTo patterns if present
    if (row.applyTo && row.applyTo.trim()) {
      const patterns = row.applyTo
        .split(',')
        .map(p => p.trim())
        .filter(Boolean)
      if (patterns.length > 0) {
        yamlConfig.file_include_patterns = patterns
      }
    }

    return yamlConfig
  },
}

/**
 * Hook to update parsers and extractors for a specific RAG strategy
 * Provides mutations that update the llamafarm.yaml config via API
 *
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param strategyName - The name of the data processing strategy
 * @returns Mutation hooks for updating parsers and extractors
 */
export const useRagStrategy = (
  namespace: string,
  projectId: string,
  strategyName: string
) => {
  const queryClient = useQueryClient()
  const updateProjectMutation = useUpdateProject()

  /**
   * Update parsers for the specified strategy
   */
  const updateParsers = useMutation({
    mutationFn: async ({
      parserRows,
      projectConfig,
    }: {
      parserRows: ParserRow[]
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for parser update')
      }

      // Get or initialize rag.data_processing_strategies
      const rag = projectConfig.rag || {}
      const strategies = rag.data_processing_strategies || []

      // Find strategy by name
      const strategyIndex = strategies.findIndex(
        (s: any) => s.name === strategyName
      )

      if (strategyIndex === -1) {
        throw new Error(`Strategy "${strategyName}" not found in config`)
      }

      // Convert rows to YAML format
      const yamlParsers = parserRows.map(ragStrategyConverters.parserRowToYaml)

      // Update the strategy's parsers
      const updatedStrategies = [...strategies]
      updatedStrategies[strategyIndex] = {
        ...updatedStrategies[strategyIndex],
        parsers: yamlParsers,
      }

      // Build updated config
      const nextConfig = {
        ...projectConfig,
        rag: {
          ...rag,
          data_processing_strategies: updatedStrategies,
        },
      }

      // Update project via API
      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    onSuccess: () => {
      // Invalidate project config to trigger refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error) => {
      console.error('Failed to update parsers:', error)
    },
  })

  /**
   * Update extractors for the specified strategy
   */
  const updateExtractors = useMutation({
    mutationFn: async ({
      extractorRows,
      projectConfig,
    }: {
      extractorRows: ExtractorRow[]
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for extractor update')
      }

      // Get or initialize rag.data_processing_strategies
      const rag = projectConfig.rag || {}
      const strategies = rag.data_processing_strategies || []

      // Find strategy by name
      const strategyIndex = strategies.findIndex(
        (s: any) => s.name === strategyName
      )

      if (strategyIndex === -1) {
        throw new Error(`Strategy "${strategyName}" not found in config`)
      }

      // Convert rows to YAML format
      const yamlExtractors = extractorRows.map(
        ragStrategyConverters.extractorRowToYaml
      )

      // Update the strategy's extractors
      const updatedStrategies = [...strategies]
      updatedStrategies[strategyIndex] = {
        ...updatedStrategies[strategyIndex],
        extractors: yamlExtractors,
      }

      // Build updated config
      const nextConfig = {
        ...projectConfig,
        rag: {
          ...rag,
          data_processing_strategies: updatedStrategies,
        },
      }

      // Update project via API
      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    onSuccess: () => {
      // Invalidate project config to trigger refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error) => {
      console.error('Failed to update extractors:', error)
    },
  })

  return {
    updateParsers,
    updateExtractors,
    isUpdating: updateParsers.isPending || updateExtractors.isPending,
    converters: ragStrategyConverters,
  }
}
