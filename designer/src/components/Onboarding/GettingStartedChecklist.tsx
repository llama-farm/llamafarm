/**
 * Getting Started Checklist component
 * Full-width horizontal layout at the top of the dashboard
 */

import { useMemo, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { ChevronUp, ArrowRight, RotateCcw, Sparkles } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'

// Fire confetti from a specific element position
const fireConfettiAt = (element: HTMLElement) => {
  // Check for reduced motion preference
  try {
    if (
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      return
    }
  } catch {}

  const rect = element.getBoundingClientRect()
  const x = (rect.left + rect.width / 2) / window.innerWidth
  const y = (rect.top + rect.height / 2) / window.innerHeight

  const fire = () => {
    const confetti = (window as any).confetti
    if (!confetti) return

    const isDark = document.documentElement.classList.contains('dark')
    const colors = isDark
      ? ['#14b8a6', '#f472b6', '#38bdf8', '#ffffff']
      : ['#0d9488', '#ec4899', '#38bdf8', '#0f172a']

    confetti({
      particleCount: 30,
      spread: 50,
      origin: { x, y },
      colors,
      scalar: 0.8,
      gravity: 1.2,
    })
  }

  // Load confetti script if not already loaded
  const existing = (window as any).confetti
  if (existing) {
    fire()
    return
  }

  try {
    const script = document.createElement('script')
    script.src =
      'https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js'
    script.async = true
    script.onload = () => fire()
    document.body.appendChild(script)
  } catch {}
}

interface GettingStartedChecklistProps {
  className?: string
}

export function GettingStartedChecklist({
  className,
}: GettingStartedChecklistProps) {
  const navigate = useNavigate()
  const checkboxRefs = useRef<Map<string, HTMLButtonElement>>(new Map())
  const {
    checklist,
    getDescription,
    getProjectTypeLabel,
    getDeployTargetLabel,
    isStepCompleted,
    completeChecklistStep,
    uncompleteChecklistStep,
    dismissChecklist,
    resetOnboarding,
    state,
    isDemo,
    demoConfig,
  } = useOnboardingContext()

  // Check if sample model training is in progress
  const isTrainingSampleModel = state.answers.isTrainingSampleModel

  // Count completed steps
  const completedCount = useMemo(() => {
    return checklist.filter(step => isStepCompleted(step.id)).length
  }, [checklist, isStepCompleted])

  const handleToggleComplete = useCallback((stepId: string, completed: boolean, element?: HTMLElement) => {
    if (completed) {
      completeChecklistStep(stepId)
      // Fire confetti from checkbox position
      if (element) {
        fireConfettiAt(element)
      }
    } else {
      uncompleteChecklistStep(stepId)
    }
  }, [completeChecklistStep, uncompleteChecklistStep])

  const handleAction = (step: typeof checklist[0]) => {
    if (step.linkLabel === 'Start over') {
      resetOnboarding()
    } else if (step.linkPath) {
      // Navigate to the step - completion happens via the floating navigator's "Next step" button
      const separator = step.linkPath.includes('?') ? '&' : '?'
      const pathWithParam = `${step.linkPath}${separator}from=checklist`
      console.log('[GettingStartedChecklist] Navigating to:', pathWithParam)
      navigate(pathWithParam)
    }
  }

  const projectTypeLabel = getProjectTypeLabel()
  const deployTargetLabel = getDeployTargetLabel()

  return (
    <div
      className={cn(
        'rounded-xl border border-border',
        'bg-gradient-to-br from-card via-card to-teal-500/5',
        'animate-in fade-in slide-in-from-top-2 duration-500',
        'ring-2 ring-teal-500/10 ring-offset-2 ring-offset-background',
        'shadow-lg shadow-teal-500/5',
        className
      )}
    >
      {/* Header */}
      <div className="px-5 py-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isDemo && demoConfig ? (
              <>
                <span className="text-2xl">{demoConfig.icon}</span>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-foreground">
                      {demoConfig.displayName}
                    </h3>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gradient-to-r from-teal-500/20 to-sky-500/20 text-xs font-medium text-teal-700 dark:text-teal-300">
                      <Sparkles className="h-3 w-3" />
                      Demo
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {completedCount}/{checklist.length} done
                  </span>
                </div>
              </>
            ) : (
              <>
                <span className="text-2xl">🦙</span>
                <div>
                  <h3 className="font-semibold text-foreground">
                    Getting Started
                  </h3>
                  <div className="flex items-center gap-2 mt-0.5">
                    {(projectTypeLabel || deployTargetLabel) && (
                      <span className="text-xs text-muted-foreground">
                        {projectTypeLabel}
                        {deployTargetLabel && ` • ${deployTargetLabel}`}
                      </span>
                    )}
                    <span className="text-xs text-muted-foreground">
                      • {completedCount}/{checklist.length} done
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={resetOnboarding}
              className="text-muted-foreground text-xs gap-1.5"
            >
              <RotateCcw className="h-3 w-3" />
              {isDemo ? 'Build your own' : 'Start over'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={dismissChecklist}
              className="h-8 w-8 p-0 text-muted-foreground"
              title="Collapse checklist"
            >
              <ChevronUp className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Checklist items - horizontal grid */}
      <div className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {checklist.map(step => {
            const completed = isStepCompleted(step.id)
            const description = getDescription(step)

            return (
              <div
                key={step.id}
                className={cn(
                  'relative p-4 rounded-lg border transition-all',
                  completed
                    ? 'bg-muted/30 border-border'
                    : 'bg-background border-border hover:border-primary/30'
                )}
              >
                {/* Checkbox in top right */}
                <div className="absolute top-3 right-3">
                  <Checkbox
                    ref={(el) => {
                      if (el) checkboxRefs.current.set(step.id, el)
                      else checkboxRefs.current.delete(step.id)
                    }}
                    checked={completed}
                    onCheckedChange={(checked) => {
                      const element = checkboxRefs.current.get(step.id)
                      handleToggleComplete(step.id, !!checked, element)
                    }}
                    aria-label={`Mark "${step.title}" as ${completed ? 'incomplete' : 'complete'}`}
                    className="h-5 w-5"
                  />
                </div>

                {/* Step title */}
                <span
                  className={cn(
                    'text-sm font-medium leading-tight block pr-8',
                    completed ? 'text-muted-foreground line-through' : 'text-foreground'
                  )}
                >
                  {step.stepNumber}. {step.title}
                </span>

                {/* Description */}
                <p
                  className={cn(
                    'text-xs leading-relaxed mt-2 mb-3',
                    completed ? 'text-muted-foreground/60' : 'text-muted-foreground'
                  )}
                >
                  {description}
                </p>

                {/* Action button - real button style */}
                {!completed && (step.linkPath || step.linkLabel === 'Start over') && (() => {
                  // Check if this is a training step that should be disabled while training
                  const isTrainingStep = step.stepNumber === 1 && (
                    step.id === 'classifier-data' || step.id === 'anomaly-data'
                  )
                  const shouldDisable = isTrainingStep && isTrainingSampleModel

                  return (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs gap-1.5"
                      onClick={() => handleAction(step)}
                      disabled={shouldDisable}
                    >
                      {shouldDisable ? 'Training...' : step.linkLabel}
                      {!shouldDisable && <ArrowRight className="h-3 w-3" />}
                    </Button>
                  )
                })()}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
