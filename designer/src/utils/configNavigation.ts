import type { ProjectConfig } from '../types/config'

/**
 * Union describing sections within the project configuration that can be
 * targeted by the ConfigEditor when switching to code mode.
 */
export type ConfigLocation =
  | { type: 'root' }
  | { type: 'prompts' }
  | { type: 'datasets' }
  | { type: 'dataset'; datasetName: string }
  | { type: 'rag' }
  | { type: 'rag.dataProcessingStrategies' }
  | { type: 'rag.dataProcessingStrategy'; strategyName: string }
  | { type: 'rag.databases' }
  | { type: 'rag.database'; databaseName: string }
  | {
      type: 'rag.database.embedding'
      embeddingName: string
      databaseName?: string
    }
  | {
      type: 'rag.database.retrieval'
      retrievalName: string
      databaseName?: string
    }
  | { type: 'runtime' }
  | { type: 'runtime.models' }
  | { type: 'runtime.model'; modelName: string }

/**
 * Normalises a JSON pointer string so it always starts with a leading slash
 * and has no trailing slash (except for the root pointer "/").
 */
export const normalisePointer = (pointer: string): string => {
  if (!pointer || pointer === '/') return '/'
  const trimmed = pointer.endsWith('/') ? pointer.slice(0, -1) : pointer
  return trimmed.startsWith('/') ? trimmed : `/${trimmed}`
}

/**
 * Returns the parent pointer of a given JSON pointer
 * @param pointer - The JSON pointer to get the parent of
 * @returns The parent pointer, or null if at root
 */
export const parentPointer = (pointer: string): string | null => {
  if (!pointer || pointer === '/') return null
  const parts = pointer.split('/')
  parts.pop()
  if (parts.length <= 1) {
    return '/'
  }
  const joined = parts.join('/')
  return joined.startsWith('/') ? joined : `/${joined}`
}

const pointerOrFallback = (pointer: string | null | undefined, fallback: string): string => {
  if (!pointer) return fallback
  return normalisePointer(pointer)
}

const findIndexByName = (list: any[] | undefined, name: string): number => {
  if (!Array.isArray(list)) return -1
  return list.findIndex(item => {
    if (!item) return false
    const itemName = typeof item === 'string' ? item : item.name
    return typeof itemName === 'string' && itemName === name
  })
}

/**
 * Attempts to resolve the JSON pointer for a given location within the
 * project configuration. When the exact location cannot be found the function
 * falls back to the closest available parent pointer so navigation still lands
 * in a relevant section.
 */
export const findConfigPointer = (
  config: ProjectConfig | undefined,
  location: ConfigLocation
): string => {
  if (!config) {
    return '/'
  }

  switch (location.type) {
    case 'root':
      return '/'
    case 'prompts':
      return Array.isArray((config as any).prompts) ? '/prompts' : '/'
    case 'datasets':
      return Array.isArray((config as any).datasets) ? '/datasets' : '/'
    case 'dataset': {
      const { datasets } = (config as any) ?? {}
      const index = findIndexByName(datasets, location.datasetName)
      return index >= 0 ? `/datasets/${index}` : findConfigPointer(config, { type: 'datasets' })
    }
    case 'rag':
      return (config as any).rag ? '/rag' : '/'
    case 'rag.dataProcessingStrategies':
      return Array.isArray((config as any)?.rag?.data_processing_strategies)
        ? '/rag/data_processing_strategies'
        : findConfigPointer(config, { type: 'rag' })
    case 'rag.dataProcessingStrategy': {
      const { rag } = (config as any) ?? {}
      const strategies = rag?.data_processing_strategies
      const index = findIndexByName(strategies, location.strategyName)
      return index >= 0
        ? `/rag/data_processing_strategies/${index}`
        : findConfigPointer(config, { type: 'rag.dataProcessingStrategies' })
    }
    case 'rag.databases':
      return Array.isArray((config as any)?.rag?.databases)
        ? '/rag/databases'
        : findConfigPointer(config, { type: 'rag' })
    case 'rag.database': {
      const { rag } = (config as any) ?? {}
      const databases = rag?.databases
      const index = findIndexByName(databases, location.databaseName)
      return index >= 0
        ? `/rag/databases/${index}`
        : findConfigPointer(config, { type: 'rag.databases' })
    }
    case 'rag.database.embedding': {
      const pointer = findEmbeddingPointer(config, location.embeddingName, location.databaseName)
      return pointerOrFallback(pointer, findConfigPointer(config, { type: 'rag.databases' }))
    }
    case 'rag.database.retrieval': {
      const pointer = findRetrievalPointer(config, location.retrievalName, location.databaseName)
      return pointerOrFallback(pointer, findConfigPointer(config, { type: 'rag.databases' }))
    }
    case 'runtime':
      return (config as any).runtime ? '/runtime' : '/'
    case 'runtime.models':
      return Array.isArray((config as any)?.runtime?.models)
        ? '/runtime/models'
        : findConfigPointer(config, { type: 'runtime' })
    case 'runtime.model': {
      const { runtime } = (config as any) ?? {}
      const models = runtime?.models
      const index = findIndexByName(models, location.modelName)
      return index >= 0
        ? `/runtime/models/${index}`
        : findConfigPointer(config, { type: 'runtime.models' })
    }
    default:
      return '/'
  }
}

const findEmbeddingPointer = (
  config: ProjectConfig,
  embeddingName: string,
  databaseName?: string
): string | null => {
  const databases = (config as any)?.rag?.databases
  if (!Array.isArray(databases)) return null

  const narrowed = typeof databaseName === 'string'
    ? databases.filter((db: any) => db?.name === databaseName)
    : databases

  for (let dbIndex = 0; dbIndex < narrowed.length; dbIndex += 1) {
    const database = narrowed[dbIndex]
    if (!database) continue
    const embeddings = database.embedding_strategies
    const index = findIndexByName(embeddings, embeddingName)
    if (index >= 0) {
      // When databaseName was provided we need the actual index in the full list
      const resolvedDbIndex = databaseName
        ? findIndexByName(databases, databaseName)
        : dbIndex
      if (resolvedDbIndex >= 0) {
        return `/rag/databases/${resolvedDbIndex}/embedding_strategies/${index}`
      }
    }
  }

  return null
}

const findRetrievalPointer = (
  config: ProjectConfig,
  retrievalName: string,
  databaseName?: string
): string | null => {
  const databases = (config as any)?.rag?.databases
  if (!Array.isArray(databases)) return null

  const narrowed = typeof databaseName === 'string'
    ? databases.filter((db: any) => db?.name === databaseName)
    : databases

  for (let dbIndex = 0; dbIndex < narrowed.length; dbIndex += 1) {
    const database = narrowed[dbIndex]
    if (!database) continue
    const retrievals = database.retrieval_strategies
    const index = findIndexByName(retrievals, retrievalName)
    if (index >= 0) {
      const resolvedDbIndex = databaseName
        ? findIndexByName(databases, databaseName)
        : dbIndex
      if (resolvedDbIndex >= 0) {
        return `/rag/databases/${resolvedDbIndex}/retrieval_strategies/${index}`
      }
    }
  }

  return null
}
