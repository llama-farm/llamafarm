/**
 * Floating checklist navigator that appears when users navigate away from dashboard
 * via a checklist link. Shows current step progress and allows quick return to guide.
 */

import { useEffect, useState, useCallback, useMemo } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { X, ArrowLeft, ArrowRight, Sparkles, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useOnboardingContext } from '../../contexts/OnboardingContext'
import { useUpgradeAvailability } from '../../hooks/useUpgradeAvailability'
import { generateNavigatorTip } from '../../utils/navigatorTips'

interface ChecklistNavigatorProps {
  className?: string
}

export function ChecklistNavigator({ className }: ChecklistNavigatorProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { checklist, state, isStepCompleted, completeChecklistStep } = useOnboardingContext()
  const { upgradeAvailable, isDismissedFor } = useUpgradeAvailability()

  // Check if upgrade banner is visible (affects our positioning)
  const upgradeBannerVisible = upgradeAvailable && !isDismissedFor('project')

  // Track whether we came from checklist (via URL param)
  const [showNavigator, setShowNavigator] = useState(false)

  // Check URL param on mount and route changes
  useEffect(() => {
    const fromChecklist = searchParams.get('from') === 'checklist'

    // Only show on project-specific pages (not home, not dashboard)
    const isProjectPage = location.pathname.startsWith('/chat/')
    const isDashboard = location.pathname === '/chat/dashboard'

    if (fromChecklist && isProjectPage && !isDashboard) {
      setShowNavigator(true)
      // Clean up the URL param without triggering navigation
      const newParams = new URLSearchParams(searchParams)
      newParams.delete('from')
      setSearchParams(newParams, { replace: true })
    }
  }, [location.pathname, searchParams, setSearchParams])

  // Hide when navigating back to dashboard or leaving project context
  useEffect(() => {
    const isProjectPage = location.pathname.startsWith('/chat/')
    const isDashboard = location.pathname === '/chat/dashboard'

    if (isDashboard || !isProjectPage) {
      setShowNavigator(false)
    }
  }, [location.pathname])

  // Find current step based on path
  // When multiple steps have the same path (e.g., train then test on same page),
  // we use the first uncompleted step, or the last completed one if all matching steps are done
  const currentStepIndex = (() => {
    const pathname = location.pathname

    // Find all steps that match this path
    const matchingIndices: number[] = []

    checklist.forEach((step, index) => {
      if (!step.linkPath) return
      const stepPath = step.linkPath.split('?')[0]
      // Check for exact match or partial match
      if (stepPath === pathname || pathname.includes(stepPath)) {
        matchingIndices.push(index)
      }
    })

    if (matchingIndices.length === 0) return -1

    // If only one match, use it
    if (matchingIndices.length === 1) return matchingIndices[0]

    // Multiple steps on same page - find first uncompleted one
    for (const idx of matchingIndices) {
      if (!isStepCompleted(checklist[idx].id)) {
        return idx
      }
    }

    // All matching steps completed - return the last one
    return matchingIndices[matchingIndices.length - 1]
  })()
  const currentStep = currentStepIndex >= 0 ? checklist[currentStepIndex] : null

  // Generate contextual tip based on current step and user's answers
  const tip = useMemo(
    () => generateNavigatorTip(currentStep, state.answers, location.pathname),
    [currentStep, state.answers, location.pathname]
  )

  // Count completed steps
  const completedCount = checklist.filter(step => isStepCompleted(step.id)).length

  // Determine next step (if any)
  const nextStep = currentStepIndex >= 0 && currentStepIndex < checklist.length - 1
    ? checklist[currentStepIndex + 1]
    : null

  // Check if current step is already completed
  const isCurrentStepCompleted = currentStep ? isStepCompleted(currentStep.id) : false

  // Check if this is the last step
  const isLastStep = currentStepIndex === checklist.length - 1

  const handleDismiss = useCallback(() => {
    setShowNavigator(false)
  }, [])

  const handleBackToGuide = useCallback(() => {
    setShowNavigator(false)
    navigate('/chat/dashboard')
  }, [navigate])

  const handleDismissClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation() // Prevent triggering the parent click
    handleDismiss()
  }, [handleDismiss])

  // Handle "Next step" button - completes current step and navigates to next (if different page)
  const handleNextStep = useCallback((e: React.MouseEvent) => {
    e.stopPropagation() // Prevent triggering the parent click (back to guide)

    // Complete current step if not already completed
    if (currentStep && !isCurrentStepCompleted) {
      completeChecklistStep(currentStep.id)
    }

    // Navigate to next step if available AND if it's a different page
    if (nextStep?.linkPath) {
      const currentPath = currentStep?.linkPath?.split('?')[0]
      const nextPath = nextStep.linkPath.split('?')[0]

      // If next step is on the same page, just complete current step (don't navigate)
      // The navigator will auto-update to show the next step
      if (currentPath === nextPath) {
        // Already completed the step above - component will re-render with new step
        return
      }

      // Different page - navigate there
      const separator = nextStep.linkPath.includes('?') ? '&' : '?'
      const pathWithParam = `${nextStep.linkPath}${separator}from=checklist`
      navigate(pathWithParam)
    } else {
      // No next step - go back to dashboard
      setShowNavigator(false)
      navigate('/chat/dashboard')
    }
  }, [currentStep, isCurrentStepCompleted, nextStep, completeChecklistStep, navigate])

  // Handle "Done" button for last step - completes step and goes back to dashboard
  const handleDone = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()

    // Complete current step if not already completed
    if (currentStep && !isCurrentStepCompleted) {
      completeChecklistStep(currentStep.id)
    }

    // Go back to dashboard
    setShowNavigator(false)
    navigate('/chat/dashboard')
  }, [currentStep, isCurrentStepCompleted, completeChecklistStep, navigate])

  // Don't render if not showing or no checklist
  if (!showNavigator || checklist.length === 0) {
    return null
  }

  // Don't render if onboarding not completed or checklist dismissed
  if (!state.onboardingCompleted || state.checklistDismissed) {
    return null
  }

  // Don't render for advanced users ("get out of my way")
  if (state.answers.experienceLevel === 'advanced') {
    return null
  }

  const stepNum = currentStep ? currentStep.stepNumber : completedCount + 1

  return (
    <div
      className={cn(
        // Position: fixed bottom-right
        // If upgrade banner is visible, position higher to avoid overlap
        // z-[9999] ensures we sit above browser dev tools
        'fixed right-4 z-[9999] transition-all duration-300',
        upgradeBannerVisible ? 'bottom-20' : 'bottom-4',
        // Sizing: larger card
        'w-[320px]',
        // Styling: colorful gradient border with glow (teal to sky blue)
        'rounded-xl shadow-xl',
        'bg-gradient-to-r from-teal-500/30 via-cyan-500/30 to-sky-500/30',
        'p-[2px]', // Gradient border effect
        'animate-in fade-in slide-in-from-bottom-3 duration-300',
        className
      )}
    >
      {/* Inner card */}
      <div className="rounded-[10px] bg-card/98 backdrop-blur-md">
        <div className="flex items-center gap-3 px-4 py-3">
          {/* Icon with gradient background */}
          <div className="flex-shrink-0 h-9 w-9 rounded-lg bg-gradient-to-br from-teal-400 to-sky-500 flex items-center justify-center shadow-md">
            <Sparkles className="h-5 w-5 text-slate-900" />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-foreground">
                Step {stepNum} of {checklist.length}
              </span>
              {/* Progress dots */}
              <div className="flex gap-1">
                {checklist.map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'h-1.5 w-1.5 rounded-full transition-colors',
                      i < stepNum
                        ? 'bg-teal-500'
                        : 'bg-muted-foreground/30'
                    )}
                  />
                ))}
              </div>
            </div>
            {currentStep && (
              <p className="text-xs text-muted-foreground mt-0.5 truncate">
                {currentStep.title}
              </p>
            )}
          </div>

          {/* Dismiss button */}
          <button
            onClick={handleDismissClick}
            className="flex-shrink-0 p-1.5 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Dismiss navigator"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Contextual tip */}
        {tip && (
          <div className="px-4 pb-2 pt-0">
            <p className="text-xs text-muted-foreground dark:text-gray-400 leading-relaxed">
              {tip.text}
            </p>
          </div>
        )}

        {/* Action buttons */}
        <div className="px-4 pb-3 pt-1 flex items-center gap-2">
          <button
            onClick={handleBackToGuide}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-muted/50 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to guide
          </button>

          {isLastStep ? (
            <button
              onClick={handleDone}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-gradient-to-r from-teal-500 to-sky-500 text-white hover:from-teal-600 hover:to-sky-600 transition-colors shadow-sm"
            >
              <Check className="h-3.5 w-3.5" />
              {isCurrentStepCompleted ? 'Done' : 'Complete & finish'}
            </button>
          ) : (
            <button
              onClick={handleNextStep}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-gradient-to-r from-teal-500 to-sky-500 text-white hover:from-teal-600 hover:to-sky-600 transition-colors shadow-sm"
            >
              {isCurrentStepCompleted ? 'Next step' : 'Done, next'}
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
