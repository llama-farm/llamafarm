/**
 * Context provider for onboarding wizard and checklist state
 * Follows the pattern established by ThemeContext.tsx and ProjectModalContext.tsx
 * State is stored per-project so each project has its own onboarding progress
 */

import React, { createContext, useContext, ReactNode, useState, useEffect } from 'react'
import { useOnboarding } from '../hooks/useOnboarding'
import type { UseOnboardingReturn } from '../types/onboarding'

const OnboardingContext = createContext<UseOnboardingReturn | undefined>(
  undefined
)

export const useOnboardingContext = (): UseOnboardingReturn => {
  const context = useContext(OnboardingContext)
  if (context === undefined) {
    throw new Error(
      'useOnboardingContext must be used within an OnboardingProvider'
    )
  }
  return context
}

interface OnboardingProviderProps {
  children: ReactNode
}

export const OnboardingProvider: React.FC<OnboardingProviderProps> = ({
  children,
}) => {
  // Track the active project to make onboarding state project-specific
  const [activeProjectId, setActiveProjectId] = useState<string | null>(() => {
    try {
      return localStorage.getItem('activeProject')
    } catch {
      return null
    }
  })

  // Listen for active project changes
  useEffect(() => {
    const handleProjectChange = (event: Event) => {
      const projectName = (event as CustomEvent<string>).detail
      setActiveProjectId(projectName || null)
    }

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'activeProject') {
        setActiveProjectId(e.newValue)
      }
    }

    window.addEventListener('lf-active-project', handleProjectChange)
    window.addEventListener('storage', handleStorageChange)

    return () => {
      window.removeEventListener('lf-active-project', handleProjectChange)
      window.removeEventListener('storage', handleStorageChange)
    }
  }, [])

  const onboarding = useOnboarding(activeProjectId)

  return (
    <OnboardingContext.Provider value={onboarding}>
      {children}
    </OnboardingContext.Provider>
  )
}
