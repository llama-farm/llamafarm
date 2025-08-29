/**
 * Project constants and shared data structures
 * Centralized location for default projects and common project utilities
 */

import type { Project } from '../types/project'

export const DEFAULT_PROJECT_NAMES = [
  'aircraft-mx-flow',
  'customer-support',
  'financial-analysis',
  'equipment-monitoring',
  'data-pipeline'
]

export const DEFAULT_PROJECTS = DEFAULT_PROJECT_NAMES.map((name, index) => ({
  id: index + 1,
  name,
  model: 'TinyLama',
  lastEdited: '8/15/2025',
  description: `Default ${name} project`
}))

/**
 * Extract project names from API response with fallback to defaults
 * @param apiResponse - The API response containing projects
 * @returns Array of project names
 */
export const getProjectsList = (apiResponse?: { projects?: Project[] }): string[] => {
  if (apiResponse?.projects && apiResponse.projects.length > 0) {
    return apiResponse.projects.map(p => p.name)
  }
  return DEFAULT_PROJECT_NAMES
}

/**
 * Convert API projects to UI ProjectItem format
 * @param apiResponse - The API response containing projects
 * @returns Array of ProjectItem objects for UI display
 */
export const getProjectsForUI = (apiResponse?: { projects?: Project[] }) => {
  if (apiResponse?.projects && apiResponse.projects.length > 0) {
    return apiResponse.projects.map((project, idx) => ({
      id: idx + 1,
      name: project.name,
      model: 'TinyLama', // TODO: Extract from project.config when available
      lastEdited: '8/15/2025', // TODO: Use actual lastModified from API
      description: project.config?.description || ''
    }))
  }
  return DEFAULT_PROJECTS
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
