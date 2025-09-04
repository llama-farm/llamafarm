import { useState, useEffect } from 'react'
import { getCurrentNamespace } from '../utils/namespaceUtils'
import { getActiveProject as getActiveProjectName } from '../utils/projectUtils'

export interface ActiveProject {
  namespace: string
  project: string
}

/**
 * Hook to manage active project state with proper namespace/project structure
 * Automatically syncs with localStorage and handles storage events
 */
export function useActiveProject(): ActiveProject | null {
  const [activeProject, setActiveProject] = useState<ActiveProject | null>(null)

  // Initialize active project from localStorage
  useEffect(() => {
    const updateActiveProject = () => {
      try {
        const namespace = getCurrentNamespace()
        const projectName = getActiveProjectName()
        
        if (namespace && projectName) {
          setActiveProject({
            namespace,
            project: projectName
          })
        } else {
          setActiveProject(null)
        }
      } catch (error) {
        console.error('Failed to get active project:', error)
        setActiveProject(null)
      }
    }

    updateActiveProject()
  }, [])

  // Listen for localStorage changes to update when active project changes
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'activeProject' || e.key === 'userNamespace') {
        try {
          const namespace = getCurrentNamespace()
          const projectName = getActiveProjectName()
          
          if (namespace && projectName) {
            setActiveProject({
              namespace,
              project: projectName
            })
          } else {
            setActiveProject(null)
          }
        } catch (error) {
          console.error('Failed to parse active project from storage event:', error)
          setActiveProject(null)
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [])

  // Listen for custom events dispatched by setActiveProject
  useEffect(() => {
    const handleActiveProjectChange = (event: Event) => {
      if (event instanceof CustomEvent && event.detail) {
        try {
          const projectData = event.detail
          if (projectData && typeof projectData === 'string') {
            const namespace = getCurrentNamespace()
            const projectName = projectData
            
            if (namespace && projectName) {
              setActiveProject({
                namespace,
                project: projectName
              })
            } else {
              setActiveProject(null)
            }
          }
        } catch (error) {
          console.error('Failed to handle active project change:', error)
        }
      }
    }

    window.addEventListener('lf-active-project', handleActiveProjectChange)
    return () => window.removeEventListener('lf-active-project', handleActiveProjectChange)
  }, [])

  return activeProject
}

/**
 * Hook variant that returns individual namespace and project values
 * Useful when you need them as separate variables
 */
export function useActiveProjectValues(): { namespace: string; project: string } | null {
  const activeProject = useActiveProject()
  return activeProject
}

export default useActiveProject
