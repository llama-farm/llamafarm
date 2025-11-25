import { Project, ListProjectsResponse } from '../../types/project'

/**
 * Factory function to create mock Project objects
 * Use this to generate test data for project-related tests
 * 
 * @example
 * ```tsx
 * const project = createMockProject({ name: 'test-project' })
 * const projects = createMockProjectsList('default', 3) // Creates 3 projects
 * ```
 */

interface MockProjectOptions {
  namespace?: string
  name?: string
  config?: Record<string, any>
  validation_error?: string | null
  last_modified?: string | null
}

/**
 * Create a single mock project
 */
export function createMockProject(
  options: MockProjectOptions = {}
): Project {
  const {
    namespace = 'default',
    name = 'test-project',
    config = {
      version: 'v1',
      name,
      namespace,
      runtime: {
        provider: 'ollama',
        model: 'llama3.2:3b',
      },
      prompts: [],
    },
    validation_error = null,
    last_modified = new Date().toISOString(),
  } = options

  return {
    namespace,
    name,
    config,
    validation_error,
    last_modified,
  }
}

/**
 * Create a list of mock projects
 */
export function createMockProjectsList(
  namespace: string = 'default',
  count: number = 2
): ListProjectsResponse {
  const projects: Project[] = []

  for (let i = 0; i < count; i++) {
    projects.push(
      createMockProject({
        namespace,
        name: `project-${i + 1}`,
      })
    )
  }

  return {
    total: projects.length,
    projects,
  }
}

/**
 * Create a mock project with specific configuration sections
 */
export function createMockProjectWithConfig(
  name: string,
  configOverrides: Record<string, any> = {}
): Project {
  const baseConfig = {
    version: 'v1',
    name,
    namespace: 'default',
    runtime: {
      provider: 'ollama',
      model: 'llama3.2:3b',
    },
    prompts: [
      {
        role: 'system',
        content: 'You are a helpful assistant.',
      },
    ],
    rag: {
      databases: [],
    },
    ...configOverrides,
  }

  return createMockProject({
    name,
    config: baseConfig,
  })
}

/**
 * Create a mock project with validation error
 */
export function createMockProjectWithError(
  name: string,
  errorMessage: string
): Project {
  return createMockProject({
    name,
    validation_error: errorMessage,
  })
}

