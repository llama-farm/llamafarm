/**
 * Utilities for transforming Model Catalog data to UI format
 */

import {
  MODEL_CATALOG,
  type ModelVariant as CatalogVariant,
} from '@/types/modelCatalog'

/**
 * UI format for model variants (matches existing Models.tsx interface)
 */
export interface LocalModelVariant {
  id: number
  label: string
  parameterSize: string
  downloadSize: string
  modelId?: string // HuggingFace model ID for universal provider
}

/**
 * UI format for model groups (matches existing Models.tsx interface)
 */
export interface LocalModelGroup {
  id: number
  name: string
  parameterSummary: string
  downloadSummary: string
  variants: LocalModelVariant[]
}

/**
 * Generate parameter summary from variants (e.g., "1b, 7b, 70b")
 */
function generateParameterSummary(variants: CatalogVariant[]): string {
  return variants.map(v => v.parameters).join(', ')
}

/**
 * Generate download size summary (e.g., "2–45 GB")
 */
function generateDownloadSummary(variants: CatalogVariant[]): string {
  const sizes = variants.map(v => {
    const match = v.download_size.match(/^([\d.]+)\s*([KMGT]?B)$/i)
    if (!match) return 0
    const [, num, unit] = match
    const multiplier = { KB: 0.001, MB: 0.001, GB: 1, TB: 1000 }[
      unit.toUpperCase()
    ] || 1
    return parseFloat(num) * multiplier
  })

  const min = Math.min(...sizes)
  const max = Math.max(...sizes)

  if (min === max) {
    return variants[0].download_size
  }

  // Format as range
  const formatSize = (gb: number) => {
    if (gb < 1) return `${Math.round(gb * 1000)} MB`
    return `${Math.round(gb)} GB`
  }

  return `${formatSize(min)}–${formatSize(max)}`
}

/**
 * Transform Model Catalog families to LocalModelGroups for UI
 */
export function transformCatalogToLocalGroups(): LocalModelGroup[] {
  const families = MODEL_CATALOG['text-generation'] as any
  let groupId = 1
  let variantId = 1

  return families.map((family: any) => {
    const id = groupId++

    const variants: LocalModelVariant[] = family.variants.map((variant: any) => {
      const vid = variantId++

      // Get universal provider model_id for downloads
      const universalProvider = variant.providers?.universal
      const modelId = universalProvider?.model_id

      // Create label from variant id (e.g., "qwen3:4b" → "qwen3,4b")
      const label = variant.id.replace(':', ',')

      return {
        id: vid,
        label,
        parameterSize: variant.parameters,
        downloadSize: variant.download_size,
        modelId, // Add HuggingFace model ID
      }
    })

    return {
      id,
      name: family.family_id,
      parameterSummary: generateParameterSummary(family.variants),
      downloadSummary: generateDownloadSummary(family.variants),
      variants,
    }
  })
}

/**
 * Get model metadata for a given variant label
 */
export function getModelMetadata(label: string): {
  familyName: string
  variantName: string
  modelId: string | undefined
  description: string | undefined
} | null {
  const families = MODEL_CATALOG['text-generation'] as any

  // Convert label back to id format (e.g., "qwen3,4b" → "qwen3:4b")
  const variantId = label.replace(',', ':')

  for (const family of families) {
    const variant = family.variants.find((v: any) => v.id === variantId)
    if (variant) {
      return {
        familyName: family.family_name,
        variantName: variant.display_name,
        modelId: variant.providers?.universal?.model_id,
        description: variant.description,
      }
    }
  }

  return null
}

/**
 * Search model groups by query
 */
export function searchModelGroups(
  groups: LocalModelGroup[],
  query: string
): LocalModelGroup[] {
  if (!query.trim()) return groups

  const lowerQuery = query.toLowerCase()

  return groups.filter(g =>
    [g.name, g.parameterSummary].some(v =>
      v.toLowerCase().includes(lowerQuery)
    )
  )
}

/**
 * Runtime type for filtering
 */
export type Runtime = 'universal' | 'ollama' | 'lemonade' | 'openai' | 'all'

/**
 * Recommended model with enriched data
 */
export interface RecommendedModelInfo {
  category: string
  categoryDescription?: string
  priority: number
  variantId: string
  displayName: string
  description?: string
  parameters: string
  downloadSize: string
  familyId: string
  familyName: string
  providers: Record<string, any>
}

/**
 * Get all recommended models from catalog, organized by category
 */
export function getRecommendedModels(): RecommendedModelInfo[] {
  const families = MODEL_CATALOG['text-generation'] as any
  const recommended: RecommendedModelInfo[] = []

  families.forEach((family: any) => {
    if (!family.recommended) return

    family.recommended.forEach((category: any) => {
      category.models.forEach((recModel: any) => {
        const variant = family.variants.find(
          (v: any) => v.id === recModel.variant_id
        )
        if (!variant) return

        recommended.push({
          category: category.category,
          categoryDescription: category.description,
          priority: recModel.priority,
          variantId: variant.id,
          displayName: variant.display_name,
          description: variant.description,
          parameters: variant.parameters,
          downloadSize: variant.download_size,
          familyId: family.family_id,
          familyName: family.family_name,
          providers: variant.providers || {},
        })
      })
    })
  })

  // Sort by priority within categories
  return recommended.sort((a, b) => a.priority - b.priority)
}

/**
 * Group recommended models by category
 */
export function getRecommendedByCategory(): Record<
  string,
  RecommendedModelInfo[]
> {
  const recommended = getRecommendedModels()
  const grouped: Record<string, RecommendedModelInfo[]> = {}

  recommended.forEach(model => {
    if (!grouped[model.category]) {
      grouped[model.category] = []
    }
    grouped[model.category].push(model)
  })

  return grouped
}

/**
 * Check if a variant supports a specific runtime
 */
export function variantSupportsRuntime(
  providers: Record<string, any>,
  runtime: Runtime
): boolean {
  if (runtime === 'all') return true
  return !!providers[runtime]
}

/**
 * Filter model groups by runtime
 */
export function filterGroupsByRuntime(
  groups: LocalModelGroup[],
  runtime: Runtime
): LocalModelGroup[] {
  if (runtime === 'all') return groups

  const families = MODEL_CATALOG['text-generation'] as any

  return groups
    .map(group => {
      // Find the family in catalog
      const family = families.find((f: any) => f.family_id === group.name)
      if (!family) return null

      // Filter variants that support the runtime
      const filteredVariants = group.variants.filter(variant => {
        const variantId = variant.label.replace(',', ':')
        const catalogVariant = family.variants.find(
          (v: any) => v.id === variantId
        )
        if (!catalogVariant) return false

        return variantSupportsRuntime(catalogVariant.providers, runtime)
      })

      if (filteredVariants.length === 0) return null

      return {
        ...group,
        variants: filteredVariants,
      }
    })
    .filter((g): g is LocalModelGroup => g !== null)
}

/**
 * Get provider info for a variant
 */
export interface ProviderInfo {
  provider: string
  runtime: string
  format: string
  modelId: string
  downloadCommand?: string
  notes?: string
  baseUrl?: string
}

/**
 * Get all providers for a variant ID
 */
export function getVariantProviders(variantId: string): ProviderInfo[] {
  const families = MODEL_CATALOG['text-generation'] as any

  for (const family of families) {
    const variant = family.variants.find((v: any) => v.id === variantId)
    if (variant && variant.providers) {
      return Object.entries(variant.providers).map(([key, config]: [string, any]) => ({
        provider: key,
        runtime: config.runtime || key,
        format: config.format || 'unknown',
        modelId: config.model_id,
        downloadCommand: config.download_command,
        notes: config.notes,
        baseUrl: config.base_url,
      }))
    }
  }

  return []
}

/**
 * Filter model groups to only show cloud models (API-based)
 */
export function filterCloudModels(groups: LocalModelGroup[]): LocalModelGroup[] {
  const families = MODEL_CATALOG['text-generation'] as any

  return groups
    .map(group => {
      const family = families.find((f: any) => f.family_id === group.name)
      if (!family) return null

      // Filter variants that have at least one cloud provider (runtime === 'openai')
      const filteredVariants = group.variants.filter(variant => {
        const variantId = variant.label.replace(',', ':')
        const catalogVariant = family.variants.find(
          (v: any) => v.id === variantId
        )
        if (!catalogVariant?.providers) return false

        // Check if any provider is cloud-based
        return Object.values(catalogVariant.providers).some(
          (p: any) => p.runtime === 'openai' || p.format === 'api'
        )
      })

      if (filteredVariants.length === 0) return null

      return {
        ...group,
        variants: filteredVariants,
      }
    })
    .filter((g): g is LocalModelGroup => g !== null)
}

/**
 * Filter model groups to only show local models (non-cloud)
 */
export function filterLocalModels(groups: LocalModelGroup[]): LocalModelGroup[] {
  const families = MODEL_CATALOG['text-generation'] as any

  return groups
    .map(group => {
      const family = families.find((f: any) => f.family_id === group.name)
      if (!family) return null

      // Filter variants that have at least one local provider
      const filteredVariants = group.variants.filter(variant => {
        const variantId = variant.label.replace(',', ':')
        const catalogVariant = family.variants.find(
          (v: any) => v.id === variantId
        )
        if (!catalogVariant?.providers) return false

        // Check if any provider is local (not cloud)
        return Object.values(catalogVariant.providers).some(
          (p: any) => p.runtime !== 'openai' && p.format !== 'api'
        )
      })

      if (filteredVariants.length === 0) return null

      return {
        ...group,
        variants: filteredVariants,
      }
    })
    .filter((g): g is LocalModelGroup => g !== null)
}
