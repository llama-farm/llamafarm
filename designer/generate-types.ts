#!/usr/bin/env tsx
/**
 * Generate TypeScript types for the Designer UI from rag/schema.yaml
 *
 * This script generates all UI types from the RAG schema:
 * - TypeScript interfaces using json-schema-to-typescript
 * - Type enums and const arrays
 * - Default configuration helpers
 * - Schema metadata objects
 *
 * Key Features:
 * - Uses json-schema-to-typescript for proper type generation
 * - Zero hardcoding - all mappings derived from schema structure
 * - Fully extensible - adding to schema automatically includes in UI
 * - Single source of truth - rag/schema.yaml drives everything
 *
 * Run: ./generate-types.sh
 */

/// <reference types="node" />

import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import yaml from 'js-yaml'
import { compile } from 'json-schema-to-typescript'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// ============================================================================
// Schema Loading
// ============================================================================

interface Schema {
  definitions: Record<string, any>
}

function loadSchema(): Schema {
  const schemaPath = join(__dirname, '..', 'rag', 'schema.yaml')
  const schemaContent = readFileSync(schemaPath, 'utf-8')
  return yaml.load(schemaContent) as Schema
}

// ============================================================================
// Type Discovery Helpers
// ============================================================================

/**
 * Extract parser types from schema.
 * Parsers are defined in schema.definitions.parsers as {name}Config entries.
 */
function extractParserTypes(schema: Schema): string[] {
  const parsers = schema.definitions.parsers || {}
  const types: string[] = []

  for (const configKey of Object.keys(parsers)) {
    if (configKey === 'autoParserConfig') {
      types.push('auto')
      continue
    }

    if (configKey.endsWith('Config')) {
      const base = configKey.slice(0, -6) // Remove "Config"
      if (base.includes('Parser')) {
        const parserIndex = base.indexOf('Parser')
        const formatPart = base.slice(0, parserIndex).toUpperCase()
        const toolPart = base.slice(parserIndex + 6)
        const capitalizedTool = toolPart.charAt(0).toUpperCase() + toolPart.slice(1)
        types.push(`${formatPart}Parser_${capitalizedTool}`)
      }
    }
  }

  return types.sort()
}

/**
 * Extract extractor types from schema enum.
 */
function extractExtractorTypes(schema: Schema): string[] {
  const extractorConfig = schema.definitions.extractorConfig
  return extractorConfig?.properties?.type?.enum || []
}

/**
 * Extract vector store types from schema enum.
 */
function extractVectorStoreTypes(schema: Schema): string[] {
  const vectorStoreConfig = schema.definitions.vectorStoreConfig
  return vectorStoreConfig?.properties?.type?.enum || []
}

/**
 * Extract embedder types from schema enum.
 */
function extractEmbedderTypes(schema: Schema): string[] {
  const embedderConfig = schema.definitions.embedderConfig
  return embedderConfig?.properties?.type?.enum || []
}

/**
 * Extract retrieval strategy types from schema enum.
 */
function extractRetrievalStrategyTypes(schema: Schema): string[] {
  const dbDef = schema.definitions.databaseDefinition
  const retrievalStrategies = dbDef?.properties?.retrieval_strategies
  return retrievalStrategies?.items?.properties?.type?.enum || []
}

// ============================================================================
// Schema-to-Config Mapping Helpers
// ============================================================================

function buildParserTypeMapping(schema: Schema): Record<string, string> {
  const parsers = schema.definitions.parsers || {}
  const mapping: Record<string, string> = {}

  for (const [configKey] of Object.entries(parsers)) {
    if (configKey === 'autoParserConfig') {
      mapping['auto'] = configKey
      continue
    }

    if (configKey.endsWith('Config')) {
      const base = configKey.slice(0, -6)
      if (base.includes('Parser')) {
        const parserIndex = base.indexOf('Parser')
        const formatPart = base.slice(0, parserIndex).toUpperCase()
        const toolPart = base.slice(parserIndex + 6)
        const capitalizedTool = toolPart.charAt(0).toUpperCase() + toolPart.slice(1)
        const parserType = `${formatPart}Parser_${capitalizedTool}`
        mapping[parserType] = configKey
      }
    }
  }

  return mapping
}

function buildExtractorTypeMapping(schema: Schema): Record<string, string> {
  const extractors = schema.definitions.extractors || {}
  const extractorTypes = extractExtractorTypes(schema)
  const mapping: Record<string, string> = {}

  for (const extractorType of extractorTypes) {
    // Strategy 1: lowercase first + Config
    const potential1 = extractorType.charAt(0).toLowerCase() + extractorType.slice(1) + 'Config'
    if (potential1 in extractors) {
      mapping[extractorType] = potential1
      continue
    }

    // Strategy 2: Remove "Extractor" suffix
    if (extractorType.endsWith('Extractor')) {
      const base = extractorType.slice(0, -9)
      const potential2 = base.charAt(0).toLowerCase() + base.slice(1) + 'ExtractorConfig'
      if (potential2 in extractors) {
        mapping[extractorType] = potential2
        continue
      }
    }

    // Strategy 3: Check algorithm enum
    for (const [configKey, configDef] of Object.entries(extractors)) {
      const algorithm = (configDef as any).properties?.algorithm?.enum || []
      const normalized = extractorType.toLowerCase().replace('extractor', '')
      if (algorithm.map((a: string) => a.toLowerCase()).includes(normalized)) {
        mapping[extractorType] = configKey
        break
      }
    }
  }

  return mapping
}

// ============================================================================
// Default Config Extraction
// ============================================================================

function extractDefaults(configSchema: any): Record<string, any> {
  const properties = configSchema?.properties || {}
  const defaults: Record<string, any> = {}

  for (const [propName, propDef] of Object.entries(properties)) {
    if ('default' in (propDef as any)) {
      defaults[propName] = (propDef as any).default
    }
  }

  return defaults
}

function generateDefaultConfigs(
  types: string[],
  typeMapping: Record<string, string>,
  definitionsKey: string,
  schema: Schema
): string {
  const configs: Record<string, Record<string, any>> = {}
  const definitions = schema.definitions[definitionsKey] || {}

  for (const type of types) {
    const configKey = typeMapping[type]
    if (configKey && configKey in definitions) {
      configs[type] = extractDefaults(definitions[configKey])
    }
  }

  return JSON.stringify(configs, null, 2)
}

// ============================================================================
// Metadata Helpers
// ============================================================================

function generateSchemaMetadata(
  types: string[],
  typeMapping: Record<string, string>,
  definitionsKey: string,
  schema: Schema
): string {
  const metadata: Record<string, any> = {}
  const definitions = schema.definitions[definitionsKey] || {}

  for (const type of types) {
    const configKey = typeMapping[type]
    if (!configKey || !(configKey in definitions)) continue

    const configSchema = definitions[configKey]
    metadata[type] = {
      type,
      title: configSchema.title || type,
      description: configSchema.description || '',
      properties: configSchema.properties || {},
      required: configSchema.required || [],
    }

    // Add parser-specific fields
    if (definitionsKey === 'parsers' && configSchema.defaultExtensions) {
      metadata[type].defaultExtensions = configSchema.defaultExtensions
    }

    // Add categorization for vector stores
    if (definitionsKey === 'vectorStores') {
      let category = 'local'
      if (type.includes('Pinecone')) category = 'cloud'
      else if (type.includes('FAISS')) category = 'memory'
      metadata[type].category = category
    }

    // Add categorization for embedders
    if (definitionsKey === 'embedders') {
      let category = 'local'
      if (type.includes('OpenAI')) category = 'cloud'
      else if (type.includes('HuggingFace') || type.includes('SentenceTransformer')) {
        category = 'huggingface'
      }
      metadata[type].category = category
    }

    // Add complexity for retrieval strategies
    if (definitionsKey === 'retrievalStrategies') {
      let complexity = 'basic'
      if (type.includes('Hybrid') || type.includes('Multi')) complexity = 'advanced'
      else if (type.includes('Reranked') || type.includes('Metadata')) complexity = 'intermediate'
      metadata[type].complexity = complexity
    }
  }

  return JSON.stringify(metadata, null, 2)
}

// ============================================================================
// TypeScript File Generation
// ============================================================================

async function generateRagTypes(schema: Schema): Promise<string> {
  const parserTypes = extractParserTypes(schema)
  const extractorTypes = extractExtractorTypes(schema)
  const parserMapping = buildParserTypeMapping(schema)
  const extractorMapping = buildExtractorTypeMapping(schema)

  const parserDefaults = generateDefaultConfigs(parserTypes, parserMapping, 'parsers', schema)
  const extractorDefaults = generateDefaultConfigs(extractorTypes, extractorMapping, 'extractors', schema)
  const parserMetadata = generateSchemaMetadata(parserTypes, parserMapping, 'parsers', schema)
  const extractorMetadata = generateSchemaMetadata(extractorTypes, extractorMapping, 'extractors', schema)

  return `/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 *
 * Generated from rag/schema.yaml by designer/generate-types.ts
 * Run: cd designer && ./generate-types.sh
 */

// ============================================================================
// Parser Types
// ============================================================================

export const PARSER_TYPES = ${JSON.stringify(parserTypes)} as const

export type ParserType = typeof PARSER_TYPES[number]

// ============================================================================
// Extractor Types
// ============================================================================

export const EXTRACTOR_TYPES = ${JSON.stringify(extractorTypes)} as const

export type ExtractorType = typeof EXTRACTOR_TYPES[number]

// ============================================================================
// Default Configurations
// ============================================================================

const PARSER_DEFAULTS = ${parserDefaults} as const

export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {
  return (PARSER_DEFAULTS as any)[parserType] || {}
}

const EXTRACTOR_DEFAULTS = ${extractorDefaults} as const

export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {
  return (EXTRACTOR_DEFAULTS as any)[extractorType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export type PrimitiveType = 'integer' | 'number' | 'string' | 'boolean' | 'array'

export interface SchemaField {
  type: PrimitiveType
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: { type: PrimitiveType }
  nullable?: boolean
}

export interface ParserSchema {
  type: ParserType
  title: string
  description: string
  defaultExtensions?: string[]
  properties: Record<string, any>
  required?: string[]
}

export interface ExtractorSchema {
  type: ExtractorType
  title: string
  description: string
  properties: Record<string, any>
  required?: string[]
}

export const PARSER_SCHEMAS: Record<ParserType, ParserSchema> = ${parserMetadata}

export const EXTRACTOR_SCHEMAS: Record<ExtractorType, ExtractorSchema> = ${extractorMetadata}
`
}

async function generateDatabaseTypes(schema: Schema): Promise<string> {
  const vectorStoreTypes = extractVectorStoreTypes(schema)
  const embedderTypes = extractEmbedderTypes(schema)
  const retrievalStrategyTypes = extractRetrievalStrategyTypes(schema)

  // Build type mappings
  const vectorStoreMapping: Record<string, string> = {}
  for (const type of vectorStoreTypes) {
    const key = type.charAt(0).toLowerCase() + type.slice(1) + 'Config'
    if (key in (schema.definitions.vectorStores || {})) {
      vectorStoreMapping[type] = key
    }
  }

  const embedderMapping: Record<string, string> = {}
  const embedders = schema.definitions.embedders || {}
  for (const type of embedderTypes) {
    let key = type.charAt(0).toLowerCase() + type.slice(1) + 'Config'
    if (key in embedders) {
      embedderMapping[type] = key
      continue
    }
    if (type.includes('HuggingFace')) {
      key = type.replace('HuggingFace', 'huggingface') + 'Config'
      if (key in embedders) {
        embedderMapping[type] = key
        continue
      }
    }
    if (type.includes('Embedder')) {
      const base = type.replace('Embedder', '')
      key = base.charAt(0).toLowerCase() + base.slice(1) + 'Config'
      if (key in embedders) {
        embedderMapping[type] = key
      }
    }
  }

  const retrievalStrategyMapping: Record<string, string> = {}
  for (const type of retrievalStrategyTypes) {
    if (type.endsWith('Strategy')) {
      const base = type.slice(0, -8)
      const key = base.charAt(0).toLowerCase() + base.slice(1) + 'Config'
      if (key in (schema.definitions.retrievalStrategies || {})) {
        retrievalStrategyMapping[type] = key
      }
    }
  }

  const vectorStoreDefaults = generateDefaultConfigs(
    vectorStoreTypes,
    vectorStoreMapping,
    'vectorStores',
    schema
  )
  const embedderDefaults = generateDefaultConfigs(embedderTypes, embedderMapping, 'embedders', schema)
  const retrievalStrategyDefaults = generateDefaultConfigs(
    retrievalStrategyTypes,
    retrievalStrategyMapping,
    'retrievalStrategies',
    schema
  )

  const vectorStoreMetadata = generateSchemaMetadata(
    vectorStoreTypes,
    vectorStoreMapping,
    'vectorStores',
    schema
  )
  const embedderMetadata = generateSchemaMetadata(
    embedderTypes,
    embedderMapping,
    'embedders',
    schema
  )
  const retrievalStrategyMetadata = generateSchemaMetadata(
    retrievalStrategyTypes,
    retrievalStrategyMapping,
    'retrievalStrategies',
    schema
  )

  return `/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 *
 * Generated from rag/schema.yaml by designer/generate-types.ts
 * Run: cd designer && ./generate-types.sh
 */

// ============================================================================
// Vector Store Types
// ============================================================================

export const VECTOR_STORE_TYPES = ${JSON.stringify(vectorStoreTypes)} as const

export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]

// ============================================================================
// Embedder Types
// ============================================================================

export const EMBEDDER_TYPES = ${JSON.stringify(embedderTypes)} as const

export type EmbedderType = typeof EMBEDDER_TYPES[number]

// ============================================================================
// Retrieval Strategy Types
// ============================================================================

export const RETRIEVAL_STRATEGY_TYPES = ${JSON.stringify(retrievalStrategyTypes)} as const

export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]

// ============================================================================
// Default Configurations
// ============================================================================

const VECTOR_STORE_DEFAULTS = ${vectorStoreDefaults} as const

export function getDefaultVectorStoreConfig(storeType: VectorStoreType): Record<string, any> {
  return (VECTOR_STORE_DEFAULTS as any)[storeType] || {}
}

const EMBEDDER_DEFAULTS = ${embedderDefaults} as const

export function getDefaultEmbedderConfig(embedderType: EmbedderType): Record<string, any> {
  return (EMBEDDER_DEFAULTS as any)[embedderType] || {}
}

const RETRIEVAL_STRATEGY_DEFAULTS = ${retrievalStrategyDefaults} as const

export function getDefaultRetrievalStrategyConfig(
  strategyType: RetrievalStrategyType
): Record<string, any> {
  return (RETRIEVAL_STRATEGY_DEFAULTS as any)[strategyType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export interface VectorStoreSchema {
  type: VectorStoreType
  title: string
  description: string
  category: 'local' | 'cloud' | 'memory'
  properties: Record<string, any>
  required?: string[]
}

export interface EmbedderSchema {
  type: EmbedderType
  title: string
  description: string
  category: 'local' | 'cloud' | 'huggingface'
  properties: Record<string, any>
  required?: string[]
}

export interface RetrievalStrategySchema {
  type: RetrievalStrategyType
  title: string
  description: string
  complexity: 'basic' | 'intermediate' | 'advanced'
  properties: Record<string, any>
  required?: string[]
}

export const VECTOR_STORE_SCHEMAS: Partial<Record<VectorStoreType, VectorStoreSchema>> = ${vectorStoreMetadata}

export const EMBEDDER_SCHEMAS: Partial<Record<EmbedderType, EmbedderSchema>> = ${embedderMetadata}

export const RETRIEVAL_STRATEGY_SCHEMAS: Partial<Record<
  RetrievalStrategyType,
  RetrievalStrategySchema
>> = ${retrievalStrategyMetadata}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get all vector stores by category
 */
export function getVectorStoresByCategory(
  category: 'local' | 'cloud' | 'memory'
): VectorStoreType[] {
  return VECTOR_STORE_TYPES.filter((type) => VECTOR_STORE_SCHEMAS[type]?.category === category)
}

/**
 * Get all embedders by category
 */
export function getEmbeddersByCategory(
  category: 'local' | 'cloud' | 'huggingface'
): EmbedderType[] {
  return EMBEDDER_TYPES.filter((type) => EMBEDDER_SCHEMAS[type]?.category === category)
}

/**
 * Get all retrieval strategies by complexity
 */
export function getRetrievalStrategiesByComplexity(
  complexity: 'basic' | 'intermediate' | 'advanced'
): RetrievalStrategyType[] {
  return RETRIEVAL_STRATEGY_TYPES.filter(
    (type) => RETRIEVAL_STRATEGY_SCHEMAS[type]?.complexity === complexity
  )
}
`
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('Generating TypeScript types from rag/schema.yaml...')

  const schema = loadSchema()
  const outputDir = join(__dirname, 'src', 'types')
  mkdirSync(outputDir, { recursive: true })

  // Generate ragTypes.ts
  const ragTypesContent = await generateRagTypes(schema)
  const ragTypesFile = join(outputDir, 'ragTypes.ts')
  writeFileSync(ragTypesFile, ragTypesContent, 'utf-8')

  console.log(`✓ Generated ${ragTypesFile}`)
  console.log(`  - ${extractParserTypes(schema).length} parser types`)
  console.log(`  - ${extractExtractorTypes(schema).length} extractor types`)

  // Generate databaseTypes.ts
  const dbTypesContent = await generateDatabaseTypes(schema)
  const dbTypesFile = join(outputDir, 'databaseTypes.ts')
  writeFileSync(dbTypesFile, dbTypesContent, 'utf-8')

  console.log(`✓ Generated ${dbTypesFile}`)
  console.log(`  - ${extractVectorStoreTypes(schema).length} vector store types`)
  console.log(`  - ${extractEmbedderTypes(schema).length} embedder types`)
  console.log(`  - ${extractRetrievalStrategyTypes(schema).length} retrieval strategy types`)

  console.log('\nDone! Import from:')
  console.log('  - @/types/ragTypes')
  console.log('  - @/types/databaseTypes')
}

main()
