/**
 * Floating checklist navigator that appears when users navigate away from dashboard
 * via a checklist link. Shows current step progress and allows quick return to guide.
 */

import { useEffect, useState, useCallback } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { X, ArrowLeft, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useOnboardingContext } from '../../contexts/OnboardingContext'
import { useUpgradeAvailability } from '../../hooks/useUpgradeAvailability'

interface ChecklistNavigatorProps {
  className?: string
}

export function ChecklistNavigator({ className }: ChecklistNavigatorProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { checklist, state, isStepCompleted } = useOnboardingContext()
  const { upgradeAvailable, isDismissedFor } = useUpgradeAvailability()

  // Check if upgrade banner is visible (affects our positioning)
  const upgradeBannerVisible = upgradeAvailable && !isDismissedFor('project')

  // Track whether we came from checklist (via URL param)
  const [showNavigator, setShowNavigator] = useState(false)

  // Check URL param on mount and route changes
  useEffect(() => {
    const fromChecklist = searchParams.get('from') === 'checklist'

    // Don't show on dashboard
    const isDashboard = location.pathname === '/chat/dashboard'

    if (fromChecklist && !isDashboard) {
      setShowNavigator(true)
      // Clean up the URL param without triggering navigation
      const newParams = new URLSearchParams(searchParams)
      newParams.delete('from')
      setSearchParams(newParams, { replace: true })
    }
  }, [location.pathname, searchParams, setSearchParams])

  // Hide when navigating back to dashboard
  useEffect(() => {
    if (location.pathname === '/chat/dashboard') {
      setShowNavigator(false)
    }
  }, [location.pathname])

  // Find current step based on path
  const currentStepIndex = checklist.findIndex(step =>
    step.linkPath && location.pathname.includes(step.linkPath.split('?')[0])
  )
  const currentStep = currentStepIndex >= 0 ? checklist[currentStepIndex] : null

  // Count completed steps
  const completedCount = checklist.filter(step => isStepCompleted(step.id)).length

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

  // Don't render if not showing or no checklist
  if (!showNavigator || checklist.length === 0) {
    return null
  }

  // Don't render if onboarding not completed or checklist dismissed
  if (!state.onboardingCompleted || state.checklistDismissed) {
    return null
  }

  const stepNum = currentStep ? currentStep.stepNumber : completedCount + 1

  return (
    <button
      onClick={handleBackToGuide}
      className={cn(
        // Position: fixed bottom-right
        // If upgrade banner is visible, position higher to avoid overlap
        'fixed right-4 z-40 transition-all duration-300',
        upgradeBannerVisible ? 'bottom-20' : 'bottom-4',
        // Sizing: larger card
        'w-[300px]',
        // Styling: colorful gradient border with glow (teal to sky blue)
        'rounded-xl shadow-xl',
        'bg-gradient-to-r from-teal-500/30 via-cyan-500/30 to-sky-500/30',
        'p-[2px]', // Gradient border effect
        'animate-in fade-in slide-in-from-bottom-3 duration-300',
        'cursor-pointer hover:shadow-2xl hover:scale-[1.02] active:scale-[0.98]',
        'text-left', // Reset button text alignment
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
            <div className="flex items-center gap-1 text-xs text-primary mt-0.5">
              <ArrowLeft className="h-3 w-3" />
              <span>Back to guide</span>
            </div>
          </div>

          {/* Dismiss button */}
          <div
            role="button"
            tabIndex={0}
            onClick={handleDismissClick}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleDismissClick(e as any) }}
            className="flex-shrink-0 p-1.5 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Dismiss navigator"
          >
            <X className="h-4 w-4" />
          </div>
        </div>
      </div>
    </button>
  )
}
