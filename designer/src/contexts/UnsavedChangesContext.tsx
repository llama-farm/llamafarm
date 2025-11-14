import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react'
import { useNavigate, useLocation, UNSAFE_NavigationContext } from 'react-router-dom'

interface UnsavedChangesContextType {
  isDirty: boolean
  setIsDirty: (dirty: boolean) => void
  showModal: boolean
  setShowModal: (show: boolean) => void
  pendingNavigation: string | null
  setPendingNavigation: (path: string | null) => void
  pendingAction: (() => void) | null
  setPendingAction: (action: (() => void) | null) => void
  attemptNavigation: (path: string) => void
  attemptAction: (action: () => void) => void
  confirmNavigation: () => void
  cancelNavigation: () => void
}

const UnsavedChangesContext = createContext<UnsavedChangesContextType | undefined>(undefined)

export function UnsavedChangesProvider({ children }: { children: React.ReactNode }) {
  const [isDirty, setIsDirty] = useState(false)
  const [showModal, setShowModal] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null)
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null)
  const navigate = useNavigate()
  const location = useLocation()
  const navigationContext = useContext(UNSAFE_NavigationContext)
  const currentPathRef = useRef(location.pathname)
  const isNavigatingRef = useRef(false) // Flag to bypass dirty check during confirmed navigation

  // Track current path
  useEffect(() => {
    currentPathRef.current = location.pathname
  }, [location.pathname])

  // Intercept navigator push and replace to check for unsaved changes
  useEffect(() => {
    if (!navigationContext?.navigator) return

    const originalPush = navigationContext.navigator.push
    const originalReplace = navigationContext.navigator.replace

    // Wrap push to check for unsaved changes
    navigationContext.navigator.push = (to: any, state?: any) => {
      const targetPath = typeof to === 'string' ? to : to.pathname
      
      // Bypass dirty check if we're in the middle of confirmed navigation
      if (isNavigatingRef.current) {
        originalPush.call(navigationContext.navigator, to, state)
        return
      }
      
      // If we have unsaved changes and we're changing routes
      if (isDirty && targetPath !== currentPathRef.current) {
        setPendingNavigation(targetPath)
        setShowModal(true)
        return
      }
      
      originalPush.call(navigationContext.navigator, to, state)
    }

    // Wrap replace too
    navigationContext.navigator.replace = (to: any, state?: any) => {
      const targetPath = typeof to === 'string' ? to : to.pathname
      
      // Bypass dirty check if we're in the middle of confirmed navigation
      if (isNavigatingRef.current) {
        originalReplace.call(navigationContext.navigator, to, state)
        return
      }
      
      if (isDirty && targetPath !== currentPathRef.current) {
        setPendingNavigation(targetPath)
        setShowModal(true)
        return
      }
      
      originalReplace.call(navigationContext.navigator, to, state)
    }

    return () => {
      navigationContext.navigator.push = originalPush
      navigationContext.navigator.replace = originalReplace
    }
  }, [isDirty, navigationContext])

  const attemptNavigation = useCallback((path: string) => {
    if (isDirty && path !== currentPathRef.current) {
      setPendingNavigation(path)
      setPendingAction(null)
      setShowModal(true)
    } else {
      navigate(path)
    }
  }, [isDirty, navigate])

  const attemptAction = useCallback((action: () => void) => {
    if (isDirty) {
      setPendingAction(() => action)
      setPendingNavigation(null)
      setShowModal(true)
    } else {
      action()
    }
  }, [isDirty])

  const confirmNavigation = useCallback(() => {
    // Note: Don't set isDirty here - let the save handler manage it
    // This prevents redundant state updates that cause flicker
    
    // Close modal and execute action immediately
    setShowModal(false)
    
    // Set flag to bypass dirty check during navigation
    isNavigatingRef.current = true
    
    if (pendingNavigation) {
      const path = pendingNavigation
      setPendingNavigation(null)
      // Small delay to let modal close animation start
      requestAnimationFrame(() => {
        navigate(path)
        // Reset the flag immediately after navigation is triggered
        isNavigatingRef.current = false
      })
    } else if (pendingAction) {
      const action = pendingAction
      setPendingAction(null)
      // Execute action after modal starts closing
      requestAnimationFrame(() => {
        action()
        // Reset the flag immediately after action is triggered
        isNavigatingRef.current = false
      })
    } else {
      // No pending action or navigation, just reset flag
      isNavigatingRef.current = false
    }
  }, [pendingNavigation, pendingAction, navigate])

  const cancelNavigation = useCallback(() => {
    setShowModal(false)
    setPendingNavigation(null)
    setPendingAction(null)
    // Reset navigation flag in case it was set
    isNavigatingRef.current = false
  }, [])

  return (
    <UnsavedChangesContext.Provider
      value={{
        isDirty,
        setIsDirty,
        showModal,
        setShowModal,
        pendingNavigation,
        setPendingNavigation,
        pendingAction,
        setPendingAction,
        attemptNavigation,
        attemptAction,
        confirmNavigation,
        cancelNavigation,
      }}
    >
      {children}
    </UnsavedChangesContext.Provider>
  )
}

export function useUnsavedChanges() {
  const context = useContext(UnsavedChangesContext)
  if (!context) {
    throw new Error('useUnsavedChanges must be used within UnsavedChangesProvider')
  }
  return context
}

