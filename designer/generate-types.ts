#!/usr/bin/env tsx
/**
 * Generate TypeScript types from rag/schema.yaml
 *
 * Uses json-schema-to-typescript to do all the heavy lifting.
 * We just extract enum values and generate const arrays.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import yaml from 'js-yaml'
import { compile } from 'json-schema-to-typescript'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

async function main() {
  console.log('Generating TypeScript types from rag/schema.yaml...')

  // Load schema
  const schemaPath = join(__dirname, '..', 'rag', 'schema.yaml')
  const schema = yaml.load(readFileSync(schemaPath, 'utf-8')) as any

  const outputDir = join(__dirname, 'src', 'types')
  mkdirSync(outputDir, { recursive: true })

  // ============================================================================
  // Generate ragTypes.ts
  // ============================================================================

  const parsers = schema.definitions?.parsers || {}
  const extractors = schema.definitions?.extractors || {}
  const extractorConfig = schema.definitions?.extractorConfig

  // Known acronyms that should be all caps
  const acronyms = new Set(['csv', 'pdf', 'msg', 'xml', 'html', 'json', 'yaml'])

  // Extract parser type names from config keys
  const parserTypes: string[] = []
  for (const key of Object.keys(parsers)) {
    if (key === 'autoParserConfig') {
      parserTypes.push('auto')
    } else if (key.endsWith('Config') && key.includes('Parser')) {
      const base = key.slice(0, -6)
      const idx = base.indexOf('Parser')
      const formatLower = base.slice(0, idx)
      // Use all caps for known acronyms, title case otherwise
      const format = acronyms.has(formatLower.toLowerCase())
        ? formatLower.toUpperCase()
        : formatLower.charAt(0).toUpperCase() + formatLower.slice(1)
      const tool = base.slice(idx + 6)
      parserTypes.push(`${format}Parser_${tool.charAt(0).toUpperCase() + tool.slice(1)}`)
    }
  }

  // Extract extractor types from enum
  const extractorTypes = extractorConfig?.properties?.type?.enum || []

  // Generate all parser config interfaces using json-schema-to-typescript
  const parserInterfaces: string[] = []
  for (const [key, def] of Object.entries(parsers)) {
    try {
      const interfaceName = key.charAt(0).toUpperCase() + key.slice(1)
      const ts = await compile(def as any, interfaceName, {
        bannerComment: '',
        style: { singleQuote: true },
      })
      parserInterfaces.push(ts.trim())
    } catch (err) {
      console.warn(`Failed to generate interface for ${key}:`, err)
    }
  }

  // Generate all extractor config interfaces
  const extractorInterfaces: string[] = []
  for (const [key, def] of Object.entries(extractors)) {
    try {
      const interfaceName = key.charAt(0).toUpperCase() + key.slice(1)
      const ts = await compile(def as any, interfaceName, {
        bannerComment: '',
        style: { singleQuote: true },
      })
      extractorInterfaces.push(ts.trim())
    } catch (err) {
      console.warn(`Failed to generate interface for ${key}:`, err)
    }
  }

  // Build parser type -> config key mapping
  const parserTypeToConfigKey: Record<string, string> = {}
  for (const key of Object.keys(parsers)) {
    if (key === 'autoParserConfig') {
      parserTypeToConfigKey['auto'] = key
    } else if (key.endsWith('Config') && key.includes('Parser')) {
      const base = key.slice(0, -6)
      const idx = base.indexOf('Parser')
      const formatLower = base.slice(0, idx)
      // Use all caps for known acronyms, title case otherwise
      const format = acronyms.has(formatLower.toLowerCase())
        ? formatLower.toUpperCase()
        : formatLower.charAt(0).toUpperCase() + formatLower.slice(1)
      const tool = base.slice(idx + 6)
      const parserType = `${format}Parser_${tool.charAt(0).toUpperCase() + tool.slice(1)}`
      parserTypeToConfigKey[parserType] = key
    }
  }

  // Build extractor type -> config key mapping
  const extractorTypeToConfigKey: Record<string, string> = {}
  for (const extractorType of extractorTypes) {
    // Strategy 1: lowercase first + Config
    const potential1 = extractorType.charAt(0).toLowerCase() + extractorType.slice(1) + 'Config'
    if (potential1 in extractors) {
      extractorTypeToConfigKey[extractorType] = potential1
      continue
    }

    // Strategy 2: Remove "Extractor" suffix
    if (extractorType.endsWith('Extractor')) {
      const base = extractorType.slice(0, -9)
      const potential2 = base.charAt(0).toLowerCase() + base.slice(1) + 'ExtractorConfig'
      if (potential2 in extractors) {
        extractorTypeToConfigKey[extractorType] = potential2
        continue
      }
    }

    // Strategy 3: Check algorithm enum
    for (const [configKey, configDef] of Object.entries(extractors)) {
      const algorithm = (configDef as any).properties?.algorithm?.enum || []
      const normalized = extractorType.toLowerCase().replace('extractor', '')
      if (algorithm.map((a: string) => a.toLowerCase()).includes(normalized)) {
        extractorTypeToConfigKey[extractorType] = configKey
        break
      }
    }
  }

  // Build schema metadata objects using ParserType as keys (for UI forms)
  const parserSchemas: Record<string, any> = {}
  for (const [parserType, configKey] of Object.entries(parserTypeToConfigKey)) {
    const def = parsers[configKey]
    parserSchemas[parserType] = {
      properties: (def as any).properties || {},
      title: (def as any).title || configKey,
      description: (def as any).description || '',
      defaultExtensions: (def as any).defaultExtensions || [],
    }
  }

  const extractorSchemas: Record<string, any> = {}
  for (const [extractorType, configKey] of Object.entries(extractorTypeToConfigKey)) {
    const def = extractors[configKey]
    extractorSchemas[extractorType] = {
      properties: (def as any).properties || {},
      title: (def as any).title || configKey,
      description: (def as any).description || '',
    }
  }

  const ragTypesContent = `/**
 * AUTO-GENERATED - DO NOT EDIT
 * Generated from rag/schema.yaml by designer/generate-types.ts
 */

// ============================================================================
// Parser Config Interfaces (generated by json-schema-to-typescript)
// ============================================================================

${parserInterfaces.join('\n\n')}

// ============================================================================
// Extractor Config Interfaces (generated by json-schema-to-typescript)
// ============================================================================

${extractorInterfaces.join('\n\n')}

// ============================================================================
// Type Constants
// ============================================================================

export const PARSER_TYPES = ${JSON.stringify(parserTypes.sort())} as const
export type ParserType = typeof PARSER_TYPES[number]

export const EXTRACTOR_TYPES = ${JSON.stringify(extractorTypes)} as const
export type ExtractorType = typeof EXTRACTOR_TYPES[number]

// ============================================================================
// Helper Types for UI Forms
// ============================================================================

export interface SchemaField {
  type?: string
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: { type?: string }
  nullable?: boolean
}

export interface ParserSchema {
  properties: Record<string, SchemaField>
  title: string
  description: string
  defaultExtensions?: string[]
}

export interface ExtractorSchema {
  properties: Record<string, SchemaField>
  title: string
  description: string
}

/**
 * Schema metadata indexed by ParserType (e.g., "PDFParser_PyPDF2")
 * Dynamically generated from rag/schema.yaml - no hardcoding needed.
 */
export const PARSER_SCHEMAS = ${JSON.stringify(parserSchemas, null, 2)} as any

/**
 * Schema metadata indexed by ExtractorType (e.g., "KeywordExtractor")
 * Dynamically generated from rag/schema.yaml - no hardcoding needed.
 */
export const EXTRACTOR_SCHEMAS = ${JSON.stringify(extractorSchemas, null, 2)} as any

export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {
  // Extract defaults from schema
  const schema = PARSER_SCHEMAS[parserType]
  if (!schema) return {}

  const defaults: Record<string, any> = {}
  for (const [key, field] of Object.entries(schema.properties as Record<string, SchemaField>)) {
    if (field && 'default' in field) {
      defaults[key] = field.default
    }
  }
  return defaults
}

export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {
  // Extract defaults from schema
  const schema = EXTRACTOR_SCHEMAS[extractorType]
  if (!schema) return {}

  const defaults: Record<string, any> = {}
  for (const [key, field] of Object.entries(schema.properties as Record<string, SchemaField>)) {
    if (field && 'default' in field) {
      defaults[key] = field.default
    }
  }
  return defaults
}
`

  writeFileSync(join(outputDir, 'ragTypes.ts'), ragTypesContent)
  console.log(`✓ Generated ragTypes.ts (${parserTypes.length} parsers, ${extractorTypes.length} extractors)`)

  // ============================================================================
  // Generate databaseTypes.ts
  // ============================================================================

  const vectorStores = schema.definitions?.vectorStores || {}
  const embedders = schema.definitions?.embedders || {}
  const retrievalStrategies = schema.definitions?.retrievalStrategies || {}

  const vectorStoreConfig = schema.definitions?.vectorStoreConfig
  const embedderConfig = schema.definitions?.embedderConfig
  const dbDef = schema.definitions?.databaseDefinition

  // Extract types from enums
  const vectorStoreTypes = vectorStoreConfig?.properties?.type?.enum || []
  const embedderTypes = embedderConfig?.properties?.type?.enum || []
  const retrievalStrategyTypes =
    dbDef?.properties?.retrieval_strategies?.items?.properties?.type?.enum || []

  // Generate vector store interfaces
  const vectorStoreInterfaces: string[] = []
  for (const [key, def] of Object.entries(vectorStores)) {
    try {
      const interfaceName = key.charAt(0).toUpperCase() + key.slice(1)
      const ts = await compile(def as any, interfaceName, {
        bannerComment: '',
        style: { singleQuote: true },
      })
      vectorStoreInterfaces.push(ts.trim())
    } catch (err) {
      console.warn(`Failed to generate interface for ${key}:`, err)
    }
  }

  // Generate embedder interfaces
  const embedderInterfaces: string[] = []
  for (const [key, def] of Object.entries(embedders)) {
    try {
      const interfaceName = key.charAt(0).toUpperCase() + key.slice(1)
      const ts = await compile(def as any, interfaceName, {
        bannerComment: '',
        style: { singleQuote: true },
      })
      embedderInterfaces.push(ts.trim())
    } catch (err) {
      console.warn(`Failed to generate interface for ${key}:`, err)
    }
  }

  // Generate retrieval strategy interfaces
  const strategyInterfaces: string[] = []
  for (const [key, def] of Object.entries(retrievalStrategies)) {
    try {
      const interfaceName = key.charAt(0).toUpperCase() + key.slice(1)
      const ts = await compile(def as any, interfaceName, {
        bannerComment: '',
        style: { singleQuote: true },
      })
      strategyInterfaces.push(ts.trim())
    } catch (err) {
      console.warn(`Failed to generate interface for ${key}:`, err)
    }
  }

  const databaseTypesContent = `/**
 * AUTO-GENERATED - DO NOT EDIT
 * Generated from rag/schema.yaml by designer/generate-types.ts
 */

// ============================================================================
// Vector Store Config Interfaces (generated by json-schema-to-typescript)
// ============================================================================

${vectorStoreInterfaces.join('\n\n')}

// ============================================================================
// Embedder Config Interfaces (generated by json-schema-to-typescript)
// ============================================================================

${embedderInterfaces.join('\n\n')}

// ============================================================================
// Retrieval Strategy Config Interfaces (generated by json-schema-to-typescript)
// ============================================================================

${strategyInterfaces.join('\n\n')}

// ============================================================================
// Type Constants
// ============================================================================

export const VECTOR_STORE_TYPES = ${JSON.stringify(vectorStoreTypes)} as const
export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]

export const EMBEDDER_TYPES = ${JSON.stringify(embedderTypes)} as const
export type EmbedderType = typeof EMBEDDER_TYPES[number]

export const RETRIEVAL_STRATEGY_TYPES = ${JSON.stringify(retrievalStrategyTypes)} as const
export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]
`

  writeFileSync(join(outputDir, 'databaseTypes.ts'), databaseTypesContent)
  console.log(
    `✓ Generated databaseTypes.ts (${vectorStoreTypes.length} stores, ${embedderTypes.length} embedders, ${retrievalStrategyTypes.length} strategies)`
  )

  console.log('\nDone! Import from @/types/ragTypes and @/types/databaseTypes')
}

main()
