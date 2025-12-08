/**
 * Project constants and shared data structures
 * Centralized location for project utilities
 */

import type { Project } from '../types/project'
import { getModelDisplayName } from './projectHelpers'

/**
 * Extract project names from API response
 * @param apiResponse - The API response containing projects
 * @returns Array of project names (empty if no projects)
 */
export const getProjectsList = (apiResponse?: {
  projects?: Project[]
}): string[] => {
  // All projects come from server - no localStorage fallback
  return (apiResponse?.projects || []).map(p => p.name)
}


/**
 * Convert API projects to UI ProjectItem format
 * @param apiResponse - The API response containing projects
 * @returns Array of ProjectItem objects for UI display (empty if no projects)
 */
export const getProjectsForUI = (
  apiResponse?: { projects?: Project[] }
) => {
  // All projects come from server - no localStorage fallback
  const api = apiResponse?.projects || []
  
  return api.map((project, idx) => ({
    id: idx + 1,
    name: project.name,
    model: getModelDisplayName(project.config),
    lastEdited: 'N/A',
    description: project.config?.description || '',
    validationError: project.validation_error,
  }))
}

/**
 * Filter projects by search term
 * @param projects - Array of projects to filter
 * @param search - Search term
 * @returns Filtered array of projects
 */
export const filterProjectsBySearch = <T extends { name: string }>(
  projects: T[],
  search: string
): T[] => {
  if (!search || search.trim() === '') return projects

  const searchTerm = search.toLowerCase().trim()
  return projects.filter(project =>
    project.name.toLowerCase().includes(searchTerm)
  )
}
