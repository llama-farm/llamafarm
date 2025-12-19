// Shared project management utilities
// NOTE: This file provides backward compatibility and localStorage management
// while the main API operations use the new Project API hooks

import { Project } from '../types/project'

export interface ProjectItem {
  id: number
  name: string
  model: string
  lastEdited: string
  description?: string
}

/**
 * Get project list from localStorage (legacy function - returns empty if none found)
 */
export const getProjectsList = (): string[] => {
  try {
    const stored = localStorage.getItem('projectsList')
    if (stored) return JSON.parse(stored) as string[]
  } catch (error) {
    console.error('Failed to read projectsList from localStorage:', error)
  }
  return []
}

/**
 * Save project list to localStorage
 */
export const saveProjectsList = (projects: string[]): void => {
  try {
    localStorage.setItem('projectsList', JSON.stringify(projects))
  } catch (error) {
    console.error('Failed to save projectsList to localStorage:', error)
  }
}

/**
 * Get active project from localStorage
 */
export const getActiveProject = (): string | null => {
  try {
    return localStorage.getItem('activeProject')
  } catch (error) {
    console.error('Failed to get active project:', error)
    return null
  }
}

/**
 * Set active project in localStorage and dispatch event
 * @param projectName - The project name to set as active
 */
export const setActiveProject = (projectName: string): void => {
  try {
    localStorage.setItem('activeProject', projectName)
    window.dispatchEvent(
      new CustomEvent<string>('lf-active-project', { detail: projectName })
    )
  } catch (error) {
    console.error('Failed to set active project:', error)
  }
}



/**
 * Convert API Project objects to UI ProjectItem objects
 */
export const apiProjectsToProjectItems = (projects: Project[]): ProjectItem[] => {
  return projects.map((project, idx) => {
    const defaultModel = project.config?.runtime?.default_model
    const firstModel = project.config?.runtime?.models?.[0]?.name
    const model = defaultModel || firstModel || 'No model'
    
    const lastEdited = project.last_modified 
      ? new Date(project.last_modified).toLocaleDateString()
      : new Date().toLocaleDateString()

    return {
      id: idx + 1,
      name: project.name,
      model,
      lastEdited,
    }
  })
}

/**
 * Convert project names to ProjectItem objects (legacy compatibility)
 */
export const namesToProjectItems = (names: string[]): ProjectItem[] => {
  return names.map((name, idx) => ({
    id: idx + 1,
    name,
    model: 'TinyLama',
    lastEdited: '8/15/2025',
  }))
}

/**
 * Filter projects by search term
 */
export const filterProjectsBySearch = <T extends { name: string }>(
  projects: T[],
  search: string
): T[] => {
  if (!search) return projects
  return projects.filter(p => p.name.toLowerCase().includes(search.toLowerCase()))
}

/**
 * Update project name in list
 */
export const updateProjectInList = (
  projects: string[],
  oldName: string,
  newName: string
): string[] => {
  return projects.map(name => (name === oldName ? newName : name))
}

/**
 * Remove project from list
 */
export const removeProjectFromList = (
  projects: string[],
  projectName: string
): string[] => {
  return projects.filter(name => name !== projectName)
}

/**
 * Add project to list if it doesn't exist
 */
export const addProjectToList = (
  projects: string[],
  projectName: string
): string[] => {
  return projects.includes(projectName) ? projects : [...projects, projectName]
}
