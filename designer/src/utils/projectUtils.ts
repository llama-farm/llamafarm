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

// Legacy getProjectsList and saveProjectsList removed - all projects now come from server API

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

// Legacy functions removed: namesToProjectItems, filterProjectsBySearch, 
// updateProjectInList, removeProjectFromList, addProjectToList
// All project management now goes through server API
