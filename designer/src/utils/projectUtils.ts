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

// Legacy default project names for backward compatibility
export const DEFAULT_PROJECT_NAMES = [
  'aircraft-mx-flow',
  'Option 1',
  'Option 2',
  'Option 3',
  'Option 4',
]

// Legacy default projects for UI fallback when API is not available
export const DEFAULT_PROJECTS: ProjectItem[] = [
  { id: 1, name: 'Aircraft MX', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 2, name: 'SkyGuard', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 3, name: 'FalconEye', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 4, name: 'EagleVision', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 5, name: 'ThunderStrike', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 6, name: 'ViperWatch', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 7, name: 'HawkEye', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 8, name: 'StealthOps MX', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 9, name: 'JetStream', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 10, name: 'RaptorControl', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 11, name: 'AeroSentinel', model: 'TinyLama', lastEdited: '8/15/2025' },
  { id: 12, name: 'CloudSurge', model: 'TinyLama', lastEdited: '8/15/2025' },
]

/**
 * Get project list from localStorage with fallback to defaults
 */
export const getProjectsList = (): string[] => {
  try {
    const stored = localStorage.getItem('projectsList')
    if (stored) return JSON.parse(stored) as string[]
  } catch (error) {
    console.error('Failed to read projectsList from localStorage:', error)
  }
  return DEFAULT_PROJECT_NAMES
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
export const getActiveProject = (): string => {
  try {
    return localStorage.getItem('activeProject') ?? DEFAULT_PROJECT_NAMES[0]
  } catch (error) {
    console.error('Failed to get active project:', error)
    return DEFAULT_PROJECT_NAMES[0]
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
  return projects.map((project, idx) => ({
    id: idx + 1,
    name: project.name,
    model: 'TinyLama', // TODO: Extract from project.config when available
    lastEdited: '8/15/2025', // TODO: Use actual lastModified from API
  }))
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
