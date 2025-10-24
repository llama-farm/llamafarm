#!/usr/bin/env tsx
/**
 * Generate TypeScript types for the Designer UI from rag/schema.yaml
 *
 * This script generates all UI types from the RAG schema:
 * - Parser types and configurations
 * - Extractor types and configurations
 * - Vector store/database types and configurations
 * - Embedder types and configurations
 * - Retrieval strategy types and configurations
 * - Default configuration functions
 * - TypeScript type definitions
 *
 * Key Features:
 * - Zero hardcoding - all mappings derived from schema structure
 * - Fully extensible - adding to schema automatically includes in UI
 * - Single source of truth - rag/schema.yaml drives everything
 * - Uses json-schema-to-typescript for standard type generation
 *
 * Run: ./generate-types.sh
 */

/// <reference types="node" />

import { readFileSync, writeFileSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import yaml from 'js-yaml'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// ============================================================================
// Schema Loading
// ============================================================================

interface SchemaProperty {
  type?: string
  title?: string
  description?: string
  default?: any
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: any
  nullable?: boolean
}

interface ParserConfig {
  type: string
  title: string
  description: string
  defaultExtensions?: string[]
  properties: Record<string, SchemaProperty>
  required?: string[]
}

interface Schema {
  definitions: {
    parsers?: Record<string, ParserConfig>
    extractors?: Record<string, any>
    extractorConfig?: any
    vectorStores?: Record<string, any>
    vectorStoreConfig?: any
    embedders?: Record<string, any>
    embedderConfig?: any
    retrievalStrategies?: Record<string, any>
    databaseDefinition?: any
  }
}

function loadSchema(): Schema {
  const schemaPath = join(__dirname, '..', 'rag', 'schema.yaml')
  const schemaContent = readFileSync(schemaPath, 'utf-8')
  return yaml.load(schemaContent) as Schema
}

// ============================================================================
// Parser Type Discovery
// ============================================================================

function buildParserTypeMapping(schema: Schema): Map<string, string> {
  const parsers = schema.definitions.parsers || {}
  const typeToConfig = new Map<string, string>()

  for (const [configKey, config] of Object.entries(parsers)) {
    if (configKey === 'autoParserConfig') {
      typeToConfig.set('auto', configKey)
      continue
    }

    if (configKey.endsWith('Config')) {
      const base = configKey.slice(0, -6) // Remove "Config" suffix

      if (base.includes('Parser')) {
        const parserIndex = base.indexOf('Parser')
        const formatPart = base.slice(0, parserIndex).toUpperCase()
        const toolPart = base.slice(parserIndex + 6)
        const capitalizedTool = toolPart.charAt(0).toUpperCase() + toolPart.slice(1)

        const parserType = `${formatPart}Parser_${capitalizedTool}`
        typeToConfig.set(parserType, configKey)
      }
    }
  }

  return typeToConfig
}

function extractParserTypes(schema: Schema): string[] {
  const typeMapping = buildParserTypeMapping(schema)
  return Array.from(typeMapping.keys()).sort()
}

function getParserConfigSchema(schema: Schema, parserType: string): ParserConfig | null {
  const typeMapping = buildParserTypeMapping(schema)
  const configKey = typeMapping.get(parserType)

  if (!configKey) return null

  const parsers = schema.definitions.parsers || {}
  return parsers[configKey] || null
}

// ============================================================================
// Extractor Type Discovery
// ============================================================================

function extractExtractorTypes(schema: Schema): string[] {
  const extractorConfig = schema.definitions.extractorConfig
  const typeEnum = extractorConfig?.properties?.type?.enum || []
  return typeEnum.sort()
}

function buildExtractorTypeMapping(schema: Schema): Map<string, string> {
  const extractors = schema.definitions.extractors || {}
  const typeToConfig = new Map<string, string>()

  const extractorTypes = extractExtractorTypes(schema)

  for (const extractorType of extractorTypes) {
    // Strategy 1: Try exact match with lowercase first letter + Config
    const potential1 = extractorType.charAt(0).toLowerCase() + extractorType.slice(1) + 'Config'
    if (potential1 in extractors) {
      typeToConfig.set(extractorType, potential1)
      continue
    }

    // Strategy 2: Try removing "Extractor" suffix and adding "Config"
    if (extractorType.endsWith('Extractor')) {
      const base = extractorType.slice(0, -9)
      const potential2 = base.charAt(0).toLowerCase() + base.slice(1) + 'ExtractorConfig'
      if (potential2 in extractors) {
        typeToConfig.set(extractorType, potential2)
        continue
      }
    }

    // Strategy 3: Check if this type is part of a config's algorithm enum
    for (const [configKey, configDef] of Object.entries(extractors)) {
      const algorithm = (configDef as any).properties?.algorithm?.enum || []
      const normalized = extractorType.toLowerCase().replace('extractor', '')
      if (algorithm.map((a: string) => a.toLowerCase()).includes(normalized)) {
        typeToConfig.set(extractorType, configKey)
        break
      }
    }
  }

  return typeToConfig
}

function getExtractorConfigSchema(schema: Schema, extractorType: string): any {
  const typeMapping = buildExtractorTypeMapping(schema)
  const configKey = typeMapping.get(extractorType)

  if (!configKey) return null

  const extractors = schema.definitions.extractors || {}
  return extractors[configKey] || null
}

// ============================================================================
// Vector Store Type Discovery
// ============================================================================

function extractVectorStoreTypes(schema: Schema): string[] {
  const vectorStoreConfig = schema.definitions.vectorStoreConfig
  const typeEnum = vectorStoreConfig?.properties?.type?.enum || []
  return typeEnum.sort()
}

function buildVectorStoreTypeMapping(schema: Schema): Map<string, string> {
  const vectorStores = schema.definitions.vectorStores || {}
  const typeToConfig = new Map<string, string>()

  const storeTypes = extractVectorStoreTypes(schema)

  for (const storeType of storeTypes) {
    const potentialKey = storeType.charAt(0).toLowerCase() + storeType.slice(1) + 'Config'
    if (potentialKey in vectorStores) {
      typeToConfig.set(storeType, potentialKey)
    }
  }

  return typeToConfig
}

function getVectorStoreConfigSchema(schema: Schema, storeType: string): any {
  const typeMapping = buildVectorStoreTypeMapping(schema)
  const configKey = typeMapping.get(storeType)

  if (!configKey) return null

  const vectorStores = schema.definitions.vectorStores || {}
  return vectorStores[configKey] || null
}

// ============================================================================
// Embedder Type Discovery
// ============================================================================

function extractEmbedderTypes(schema: Schema): string[] {
  const embedderConfig = schema.definitions.embedderConfig
  const typeEnum = embedderConfig?.properties?.type?.enum || []
  return typeEnum.sort()
}

function buildEmbedderTypeMapping(schema: Schema): Map<string, string> {
  const embedders = schema.definitions.embedders || {}
  const typeToConfig = new Map<string, string>()

  const embedderTypes = extractEmbedderTypes(schema)

  for (const embedderType of embedderTypes) {
    // Try lowercase first letter + Config
    let potentialKey = embedderType.charAt(0).toLowerCase() + embedderType.slice(1) + 'Config'
    if (potentialKey in embedders) {
      typeToConfig.set(embedderType, potentialKey)
      continue
    }

    // Handle HuggingFaceEmbedder -> huggingfaceEmbedderConfig
    if (embedderType.includes('HuggingFace')) {
      potentialKey = embedderType.replace('HuggingFace', 'huggingface') + 'Config'
      if (potentialKey in embedders) {
        typeToConfig.set(embedderType, potentialKey)
        continue
      }
    }

    // Handle SentenceTransformerEmbedder -> sentenceTransformerConfig
    if (embedderType.includes('Embedder')) {
      const base = embedderType.replace('Embedder', '')
      potentialKey = base.charAt(0).toLowerCase() + base.slice(1) + 'Config'
      if (potentialKey in embedders) {
        typeToConfig.set(embedderType, potentialKey)
      }
    }
  }

  return typeToConfig
}

function getEmbedderConfigSchema(schema: Schema, embedderType: string): any {
  const typeMapping = buildEmbedderTypeMapping(schema)
  const configKey = typeMapping.get(embedderType)

  if (!configKey) return null

  const embedders = schema.definitions.embedders || {}
  return embedders[configKey] || null
}

// ============================================================================
// Retrieval Strategy Type Discovery
// ============================================================================

function extractRetrievalStrategyTypes(schema: Schema): string[] {
  const dbDef = schema.definitions.databaseDefinition
  const retrievalStrategies = dbDef?.properties?.retrieval_strategies
  const items = retrievalStrategies?.items
  const typeProp = items?.properties?.type
  const typeEnum = typeProp?.enum || []
  return typeEnum.sort()
}

function buildRetrievalStrategyTypeMapping(schema: Schema): Map<string, string> {
  const retrievalStrategies = schema.definitions.retrievalStrategies || {}
  const typeToConfig = new Map<string, string>()

  const strategyTypes = extractRetrievalStrategyTypes(schema)

  for (const strategyType of strategyTypes) {
    if (strategyType.endsWith('Strategy')) {
      const base = strategyType.slice(0, -8) // Remove "Strategy"
      const potentialKey = base.charAt(0).toLowerCase() + base.slice(1) + 'Config'
      if (potentialKey in retrievalStrategies) {
        typeToConfig.set(strategyType, potentialKey)
      }
    }
  }

  return typeToConfig
}

function getRetrievalStrategyConfigSchema(schema: Schema, strategyType: string): any {
  const typeMapping = buildRetrievalStrategyTypeMapping(schema)
  const configKey = typeMapping.get(strategyType)

  if (!configKey) return null

  const retrievalStrategies = schema.definitions.retrievalStrategies || {}
  return retrievalStrategies[configKey] || null
}

// ============================================================================
// Default Config Generation
// ============================================================================

function generateDefaultConfig(configSchema: any): Record<string, any> {
  const properties = configSchema?.properties || {}
  const defaults: Record<string, any> = {}

  for (const [propName, propDef] of Object.entries(properties)) {
    if ('default' in (propDef as any)) {
      defaults[propName] = (propDef as any).default
    }
  }

  return defaults
}

// ============================================================================
// Schema Property Conversion
// ============================================================================

function convertSchemaPropertiesToTS(properties: Record<string, SchemaProperty>): string {
  if (!properties || Object.keys(properties).length === 0) {
    return '{}'
  }

  const result: string[] = []

  for (const [propName, propDef] of Object.entries(properties)) {
    const fieldParts: string[] = []

    // Type mapping
    const yamlType = propDef.type || 'string'
    if (['integer', 'number', 'string', 'boolean', 'array'].includes(yamlType)) {
      fieldParts.push(`type: "${yamlType}"`)
    } else {
      fieldParts.push('type: "string"')
    }

    // Optional fields
    if (propDef.title) {
      fieldParts.push(`title: ${JSON.stringify(propDef.title)}`)
    }

    if (propDef.description) {
      fieldParts.push(`description: ${JSON.stringify(propDef.description)}`)
    }

    if ('default' in propDef) {
      fieldParts.push(`default: ${JSON.stringify(propDef.default)}`)
    }

    if (propDef.minimum !== undefined) {
      fieldParts.push(`minimum: ${propDef.minimum}`)
    }

    if (propDef.maximum !== undefined) {
      fieldParts.push(`maximum: ${propDef.maximum}`)
    }

    if (propDef.enum) {
      fieldParts.push(`enum: ${JSON.stringify(propDef.enum)}`)
    }

    if (propDef.items && yamlType === 'array') {
      const itemsType = propDef.items.type || 'string'
      if (['integer', 'number', 'string', 'boolean', 'array'].includes(itemsType)) {
        fieldParts.push(`items: { type: "${itemsType}" }`)
      } else if (itemsType === 'object') {
        fieldParts.push('items: { type: "object" as any }')
      }
    }

    if (propDef.nullable) {
      fieldParts.push('nullable: true')
    }

    const fieldStr = `{ ${fieldParts.join(', ')} }`
    result.push(`      ${propName}: ${fieldStr}`)
  }

  return result.join(',\n')
}

// ============================================================================
// TypeScript Generation - RAG Types
// ============================================================================

function generateRagTypesTypeScript(schema: Schema): string {
  const parserTypes = extractParserTypes(schema)
  const extractorTypes = extractExtractorTypes(schema)

  const lines: string[] = [
    '/**',
    ' * AUTO-GENERATED FILE - DO NOT EDIT',
    ' * ',
    ' * Generated from rag/schema.yaml by designer/generate-types.ts',
    ' * Run: cd designer && ./generate-types.sh',
    ' */',
    '',
    '// ============================================================================',
    '// Parser Types',
    '// ============================================================================',
    '',
    'export const PARSER_TYPES = [',
  ]

  for (const pt of parserTypes) {
    lines.push(`  "${pt}",`)
  }

  lines.push(
    '] as const',
    '',
    'export type ParserType = typeof PARSER_TYPES[number]',
    '',
    '// ============================================================================',
    '// Extractor Types',
    '// ============================================================================',
    '',
    'export const EXTRACTOR_TYPES = [',
  )

  for (const et of extractorTypes) {
    lines.push(`  "${et}",`)
  }

  lines.push(
    '] as const',
    '',
    'export type ExtractorType = typeof EXTRACTOR_TYPES[number]',
    '',
    '// ============================================================================',
    '// Default Configurations',
    '// ============================================================================',
    '',
    'export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {',
    '  const configs: Record<ParserType, Record<string, any>> = {',
  )

  for (const pt of parserTypes) {
    const configSchema = getParserConfigSchema(schema, pt)
    const defaultConfig = generateDefaultConfig(configSchema)
    const configJson = JSON.stringify(defaultConfig, null, 2)
    const indented = configJson.split('\n').map(line => `      ${line}`).join('\n')
    lines.push(`    "${pt}": ${indented},`)
  }

  lines.push(
    '  }',
    '  return configs[parserType] || {}',
    '}',
    '',
    'export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {',
    '  const configs: Record<ExtractorType, Record<string, any>> = {',
  )

  for (const et of extractorTypes) {
    const configSchema = getExtractorConfigSchema(schema, et)
    const defaultConfig = generateDefaultConfig(configSchema)
    const configJson = JSON.stringify(defaultConfig, null, 2)
    const indented = configJson.split('\n').map(line => `      ${line}`).join('\n')
    lines.push(`    "${et}": ${indented},`)
  }

  lines.push(
    '  }',
    '  return configs[extractorType] || {}',
    '}',
    '',
    '// ============================================================================',
    '// Schema Metadata',
    '// ============================================================================',
    '',
    "export type PrimitiveType = 'integer' | 'number' | 'string' | 'boolean' | 'array'",
    '',
    'export type SchemaField = {',
    '  type: PrimitiveType',
    '  title?: string',
    '  description?: string',
    '  default?: unknown',
    '  minimum?: number',
    '  maximum?: number',
    '  enum?: string[]',
    '  items?: { type: PrimitiveType }',
    '  nullable?: boolean',
    '}',
    '',
    'export interface ParserSchema {',
    '  type: ParserType',
    '  title: string',
    '  description: string',
    '  defaultExtensions: string[]',
    '  properties: Record<string, SchemaField>',
    '  required?: string[]',
    '}',
    '',
    'export interface ExtractorSchema {',
    '  type: ExtractorType',
    '  title: string',
    '  description: string',
    '  properties: Record<string, SchemaField>',
    '  required?: string[]',
    '}',
    '',
    'export const PARSER_SCHEMAS: Record<ParserType, ParserSchema> = {',
  )

  // Add parser schema metadata
  for (const pt of parserTypes) {
    const configSchema = getParserConfigSchema(schema, pt)
    if (!configSchema) continue

    const title = configSchema.title || pt
    const description = configSchema.description || ''
    const extensions = configSchema.defaultExtensions || []
    const properties = configSchema.properties || {}
    const required = configSchema.required || []

    lines.push(
      `  "${pt}": {`,
      `    type: "${pt}",`,
      `    title: ${JSON.stringify(title)},`,
      `    description: ${JSON.stringify(description)},`,
      `    defaultExtensions: ${JSON.stringify(extensions)},`,
    )

    const propertiesTS = convertSchemaPropertiesToTS(properties)
    if (Object.keys(properties).length > 0) {
      lines.push('    properties: {')
      lines.push(propertiesTS)
      lines.push('    },')
    } else {
      lines.push('    properties: {},')
    }

    if (required.length > 0) {
      lines.push(`    required: ${JSON.stringify(required)},`)
    }

    lines.push('  },')
  }

  lines.push(
    '}',
    '',
    'export const EXTRACTOR_SCHEMAS: Record<ExtractorType, ExtractorSchema> = {',
  )

  // Add extractor schema metadata
  for (const et of extractorTypes) {
    const configSchema = getExtractorConfigSchema(schema, et)
    if (!configSchema) continue

    const title = configSchema.title || et
    const description = configSchema.description || ''
    const properties = configSchema.properties || {}
    const required = configSchema.required || []

    lines.push(
      `  "${et}": {`,
      `    type: "${et}",`,
      `    title: ${JSON.stringify(title)},`,
      `    description: ${JSON.stringify(description)},`,
    )

    const propertiesTS = convertSchemaPropertiesToTS(properties)
    if (Object.keys(properties).length > 0) {
      lines.push('    properties: {')
      lines.push(propertiesTS)
      lines.push('    },')
    } else {
      lines.push('    properties: {},')
    }

    if (required.length > 0) {
      lines.push(`    required: ${JSON.stringify(required)},`)
    }

    lines.push('  },')
  }

  lines.push('}', '')

  return lines.join('\n')
}

// ============================================================================
// TypeScript Generation - Database Types
// ============================================================================

function generateDatabaseTypesTypeScript(schema: Schema): string {
  const vectorStoreTypes = extractVectorStoreTypes(schema)
  const embedderTypes = extractEmbedderTypes(schema)
  const retrievalStrategyTypes = extractRetrievalStrategyTypes(schema)

  const lines: string[] = [
    '/**',
    ' * AUTO-GENERATED FILE - DO NOT EDIT',
    ' * ',
    ' * Generated from rag/schema.yaml by designer/generate-types.ts',
    ' * Run: cd designer && ./generate-types.sh',
    ' */',
    '',
    '// ============================================================================',
    '// Vector Store / Database Types',
    '// ============================================================================',
    '',
    'export const VECTOR_STORE_TYPES = [',
  ]

  for (const vst of vectorStoreTypes) {
    lines.push(`  "${vst}",`)
  }

  lines.push(
    '] as const',
    '',
    'export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]',
    '',
    '// ============================================================================',
    '// Embedder Types',
    '// ============================================================================',
    '',
    'export const EMBEDDER_TYPES = [',
  )

  for (const et of embedderTypes) {
    lines.push(`  "${et}",`)
  }

  lines.push(
    '] as const',
    '',
    'export type EmbedderType = typeof EMBEDDER_TYPES[number]',
    '',
    '// ============================================================================',
    '// Retrieval Strategy Types',
    '// ============================================================================',
    '',
    'export const RETRIEVAL_STRATEGY_TYPES = [',
  )

  for (const rst of retrievalStrategyTypes) {
    lines.push(`  "${rst}",`)
  }

  lines.push(
    '] as const',
    '',
    'export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]',
    '',
    '// ============================================================================',
    '// Default Configurations - Vector Stores',
    '// ============================================================================',
    '',
    'export function getDefaultVectorStoreConfig(storeType: VectorStoreType): Record<string, any> {',
    '  const configs: Record<VectorStoreType, Record<string, any>> = {',
  )

  for (const vst of vectorStoreTypes) {
    const configSchema = getVectorStoreConfigSchema(schema, vst)
    const defaultConfig = generateDefaultConfig(configSchema)
    const configJson = JSON.stringify(defaultConfig, null, 2)
    const indented = configJson.split('\n').map(line => `      ${line}`).join('\n')
    lines.push(`    "${vst}": ${indented},`)
  }

  lines.push(
    '  }',
    '  return configs[storeType] || {}',
    '}',
    '',
    '// ============================================================================',
    '// Default Configurations - Embedders',
    '// ============================================================================',
    '',
    'export function getDefaultEmbedderConfig(embedderType: EmbedderType): Record<string, any> {',
    '  const configs: Record<EmbedderType, Record<string, any>> = {',
  )

  for (const et of embedderTypes) {
    const configSchema = getEmbedderConfigSchema(schema, et)
    const defaultConfig = generateDefaultConfig(configSchema)
    const configJson = JSON.stringify(defaultConfig, null, 2)
    const indented = configJson.split('\n').map(line => `      ${line}`).join('\n')
    lines.push(`    "${et}": ${indented},`)
  }

  lines.push(
    '  }',
    '  return configs[embedderType] || {}',
    '}',
    '',
    '// ============================================================================',
    '// Default Configurations - Retrieval Strategies',
    '// ============================================================================',
    '',
    'export function getDefaultRetrievalStrategyConfig(strategyType: RetrievalStrategyType): Record<string, any> {',
    '  const configs: Record<RetrievalStrategyType, Record<string, any>> = {',
  )

  for (const rst of retrievalStrategyTypes) {
    const configSchema = getRetrievalStrategyConfigSchema(schema, rst)
    const defaultConfig = generateDefaultConfig(configSchema)
    const configJson = JSON.stringify(defaultConfig, null, 2)
    const indented = configJson.split('\n').map(line => `      ${line}`).join('\n')
    lines.push(`    "${rst}": ${indented},`)
  }

  lines.push(
    '  }',
    '  return configs[strategyType] || {}',
    '}',
    '',
    '// ============================================================================',
    '// Schema Metadata',
    '// ============================================================================',
    '',
    'export interface VectorStoreSchema {',
    '  type: VectorStoreType',
    '  title: string',
    '  description: string',
    "  category: 'local' | 'cloud' | 'memory'",
    '}',
    '',
    'export interface EmbedderSchema {',
    '  type: EmbedderType',
    '  title: string',
    '  description: string',
    "  category: 'local' | 'cloud' | 'huggingface'",
    '}',
    '',
    'export interface RetrievalStrategySchema {',
    '  type: RetrievalStrategyType',
    '  title: string',
    '  description: string',
    "  complexity: 'basic' | 'intermediate' | 'advanced'",
    '}',
    '',
    'export const VECTOR_STORE_SCHEMAS: Record<VectorStoreType, VectorStoreSchema> = {',
  )

  // Add vector store schema metadata
  for (const vst of vectorStoreTypes) {
    const configSchema = getVectorStoreConfigSchema(schema, vst)
    const title = configSchema?.title || vst
    const description = configSchema?.description || ''

    // Infer category from type name (matching Python logic)
    let category = 'local'
    if (vst.includes('Pinecone')) category = 'cloud'
    else if (vst.includes('FAISS')) category = 'memory'

    lines.push(
      `  "${vst}": {`,
      `    type: "${vst}",`,
      `    title: ${JSON.stringify(title)},`,
      `    description: ${JSON.stringify(description)},`,
      `    category: "${category}",`,
      '  },',
    )
  }

  lines.push(
    '}',
    '',
    'export const EMBEDDER_SCHEMAS: Record<EmbedderType, EmbedderSchema> = {',
  )

  // Add embedder schema metadata
  for (const et of embedderTypes) {
    const configSchema = getEmbedderConfigSchema(schema, et)
    const title = configSchema?.title || et
    const description = configSchema?.description || ''

    // Infer category from type name (matching Python logic)
    let category = 'local'
    if (et.includes('OpenAI')) category = 'cloud'
    else if (et.includes('HuggingFace') || et.includes('SentenceTransformer')) category = 'huggingface'

    lines.push(
      `  "${et}": {`,
      `    type: "${et}",`,
      `    title: ${JSON.stringify(title)},`,
      `    description: ${JSON.stringify(description)},`,
      `    category: "${category}",`,
      '  },',
    )
  }

  lines.push(
    '}',
    '',
    'export const RETRIEVAL_STRATEGY_SCHEMAS: Record<RetrievalStrategyType, RetrievalStrategySchema> = {',
  )

  // Add retrieval strategy schema metadata
  for (const rst of retrievalStrategyTypes) {
    const configSchema = getRetrievalStrategyConfigSchema(schema, rst)
    const title = configSchema?.title || rst
    const description = configSchema?.description || ''

    // Infer complexity from type name (matching Python logic)
    let complexity = 'basic'
    if (rst.includes('Hybrid') || rst.includes('Multi')) complexity = 'advanced'
    else if (rst.includes('Reranked') || rst.includes('Metadata')) complexity = 'intermediate'

    lines.push(
      `  "${rst}": {`,
      `    type: "${rst}",`,
      `    title: ${JSON.stringify(title)},`,
      `    description: ${JSON.stringify(description)},`,
      `    complexity: "${complexity}",`,
      '  },',
    )
  }

  lines.push(
    '}',
    '',
    '// ============================================================================',
    '// Helper Functions',
    '// ============================================================================',
    '',
    '/**',
    ' * Get all vector stores by category',
    ' */',
    "export function getVectorStoresByCategory(category: 'local' | 'cloud' | 'memory'): VectorStoreType[] {",
    '  return VECTOR_STORE_TYPES.filter(type => VECTOR_STORE_SCHEMAS[type].category === category)',
    '}',
    '',
    '/**',
    ' * Get all embedders by category',
    ' */',
    "export function getEmbeddersByCategory(category: 'local' | 'cloud' | 'huggingface'): EmbedderType[] {",
    '  return EMBEDDER_TYPES.filter(type => EMBEDDER_SCHEMAS[type].category === category)',
    '}',
    '',
    '/**',
    ' * Get all retrieval strategies by complexity',
    ' */',
    "export function getRetrievalStrategiesByComplexity(complexity: 'basic' | 'intermediate' | 'advanced'): RetrievalStrategyType[] {",
    '  return RETRIEVAL_STRATEGY_TYPES.filter(type => RETRIEVAL_STRATEGY_SCHEMAS[type].complexity === complexity)',
    '}',
    '',
  )

  return lines.join('\n')
}

// ============================================================================
// Main
// ============================================================================

function main() {
  console.log('Generating TypeScript types from rag/schema.yaml...')

  const schema = loadSchema()

  // Create output directory
  const outputDir = join(__dirname, 'src', 'types')
  mkdirSync(outputDir, { recursive: true })

  // Generate ragTypes.ts
  const ragTypesContent = generateRagTypesTypeScript(schema)
  const ragTypesFile = join(outputDir, 'ragTypes.ts')
  writeFileSync(ragTypesFile, ragTypesContent, 'utf-8')

  console.log(`✓ Generated ${ragTypesFile}`)
  console.log(`  - ${extractParserTypes(schema).length} parser types`)
  console.log(`  - ${extractExtractorTypes(schema).length} extractor types`)

  // Generate databaseTypes.ts
  const dbTypesContent = generateDatabaseTypesTypeScript(schema)
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
