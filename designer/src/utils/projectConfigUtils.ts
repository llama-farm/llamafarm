/**
 * Project configuration utilities for validation and debugging
 */

/**
 * Validates that a project config has the required structure
 * @param config - The project config to validate
 * @returns true if valid, false otherwise
 */
export function validateProjectConfig(config: any): boolean {
  if (!config || typeof config !== 'object') {
    console.error('Project config validation failed: config is not an object', config)
    return false
  }

  // Check for required top-level properties
  const requiredFields = ['version', 'name', 'namespace']
  for (const field of requiredFields) {
    if (!(field in config)) {
      console.error(`Project config validation failed: missing required field '${field}'`, config)
      return false
    }
  }

  // Validate nested structures exist (they can be empty but should be defined)
  if (config.rag && typeof config.rag !== 'object') {
    console.error('Project config validation failed: rag must be an object', config)
    return false
  }

  if (config.runtime && typeof config.runtime !== 'object') {
    console.error('Project config validation failed: runtime must be an object', config)
    return false
  }

  if (config.prompts && !Array.isArray(config.prompts)) {
    console.error('Project config validation failed: prompts must be an array', config)
    return false
  }

  if (config.datasets && !Array.isArray(config.datasets)) {
    console.error('Project config validation failed: datasets must be an array', config)
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
