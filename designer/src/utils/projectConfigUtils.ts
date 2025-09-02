/**
 * Project configuration utilities for validation and debugging
 */

/**
 * Simple config validation for UX
 * @param config - The project config to validate
 * @returns true if basic structure is ok, false otherwise
 */
export function validateProjectConfig(config: any): boolean {
  // Only basic structure check
  if (!config || typeof config !== 'object') {
    return false
  }
  
  return true
}

/**
 * Creates a minimal valid config structure for testing
 * @param name - Project name
 * @param namespace - Project namespace
 * @returns A minimal valid config
 */
export function createMinimalConfig(name: string, namespace: string): Record<string, any> {
  return {
    version: 'v1',
    name,
    namespace,
    prompts: [],
    datasets: [],
    rag: {
      strategies: [],
      strategy_templates: {}
    },
    runtime: {
      provider: 'ollama',
      model: 'granite3-moe'
    }
  }
}

/**
 * Merges an existing config with updates while preserving structure
 * @param existingConfig - The current project config
 * @param updates - Updates to apply
 * @returns The merged config
 */
export function mergeProjectConfig(
  existingConfig: Record<string, any>, 
  updates: Partial<Record<string, any>>
): Record<string, any> {
  return {
    ...existingConfig,
    ...updates,
    // Ensure nested objects are preserved if not being updated
    rag: updates.rag || existingConfig.rag,
    runtime: updates.runtime || existingConfig.runtime,
    prompts: updates.prompts || existingConfig.prompts || [],
    datasets: updates.datasets || existingConfig.datasets || []
  }
}
