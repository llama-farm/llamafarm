/**
 * Project API Types - aligned with server/api/routers/projects/
 * 
 * This file contains types for Project API communication.
 * These types should remain stable and aligned with the API contract.
 */

/**
 * Core Project entity structure for API communication
 * Used in responses from the projects service
 * 
 * @example
 * const project: Project = {
 *   namespace: 'user123',
 *   name: 'my-project',
 *   config: { template: 'default', parameters: {} }
 * }
 */
export interface Project {
  /** Project namespace (typically user/org identifier) */
  namespace: string
  /** Unique project name within the namespace */
  name: string
  /** Project configuration object */
  config: Record<string, any>
  /** Validation error message if config has validation issues */
  validation_error?: string | null
  /** Last modified timestamp of the project config */
  last_modified?: string | null
}

/**
 * Request payload for creating a new project
 */
export interface CreateProjectRequest {
  /** Project name */
  name: string
  /** Configuration template to use for the project */
  config_template?: string
}

/**
 * Response from creating a new project
 */
export interface CreateProjectResponse {
  /** The created project */
  project: Project
}

/**
 * Response from listing projects in a namespace
 */
export interface ListProjectsResponse {
  /** Total number of projects */
  total: number
  /** Array of projects */
  projects: Project[]
}

/**
 * Response from getting a single project
 */
export interface GetProjectResponse {
  /** The requested project */
  project: Project
}

/**
 * Request payload for updating a project
 */
export interface UpdateProjectRequest {
  /** Updated project configuration */
  config: Record<string, any>
}

/**
 * Response from updating a project
 */
export interface UpdateProjectResponse {
  /** The updated project */
  project: Project
}

/**
 * Response from deleting a project
 */
export interface DeleteProjectResponse {
  /** The deleted project */
  project: Project
}

/**
 * Standard error response structure
 */
export interface ProjectApiError {
  /** Error detail message */
  detail?: string
}

/**
 * Base error classes for Project API operations
 */
export class ProjectError extends Error {
  constructor(message: string, public statusCode?: number, public data?: any) {
    super(message)
    this.name = 'ProjectError'
  }
}

export class ProjectValidationError extends ProjectError {
  constructor(message: string, data?: any) {
    super(message, 422, data)
    this.name = 'ProjectValidationError'
  }
}

export class ProjectNotFoundError extends ProjectError {
  constructor(message: string, data?: any) {
    super(message, 404, data)
    this.name = 'ProjectNotFoundError'
  }
}

export class ProjectNetworkError extends ProjectError {
  constructor(message: string, originalError?: any) {
    super(message, undefined, originalError)
    this.name = 'ProjectNetworkError'
  }
}
