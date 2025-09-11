import { useNavigate, useLocation } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import { useActiveProject } from './useActiveProject'

/**
 * Hook that automatically navigates users away from dataset-specific pages
 * when they switch to a different project using the project switcher.
 * 
 * This prevents users from being stuck on dataset pages that don't exist
 * in the new project and provides a smooth project switching experience.
 */
export function useProjectSwitchNavigation() {
  const navigate = useNavigate()
  const location = useLocation()
  const prevProjectRef = useRef<string | null>(null)
  const currentProject = useActiveProject()
  
  // Create a unique key for the current project to detect changes
  const currentProjectKey = currentProject 
    ? `${currentProject.namespace}/${currentProject.project}` 
    : null

  useEffect(() => {
    // Check if we're on a dataset-specific page (e.g., /chat/data/DatasetName)
    const isOnDatasetPage = location.pathname.match(/^\/chat\/data\/[^/]+$/)
    
    // If we're on a dataset page and the project has changed (not initial load)
    if (isOnDatasetPage && 
        currentProjectKey && 
        prevProjectRef.current && 
        prevProjectRef.current !== currentProjectKey) {
      
      // Navigate to the new project's data overview page
      // Using replace: true so back button works correctly
      navigate('/chat/data', { replace: true })
    }
    
    // Update the previous project reference for next comparison
    prevProjectRef.current = currentProjectKey
  }, [currentProjectKey, location.pathname, navigate])
}
