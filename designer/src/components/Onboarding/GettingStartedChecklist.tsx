/**
 * Getting Started Checklist component
 * Full-width horizontal layout at the top of the dashboard
 */

import { useMemo, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import confetti from 'canvas-confetti'
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

  // Find the current step (first incomplete step)
  const currentStepId = useMemo(() => {
    const firstIncomplete = checklist.find(step => !isStepCompleted(step.id))
    return firstIncomplete?.id || null
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
      // Check if it's an external URL (starts with http)
      if (step.linkPath.startsWith('http')) {
        window.open(step.linkPath, '_blank', 'noopener,noreferrer')
      } else {
        // Navigate to the step - completion happens via the floating navigator's "Next step" button
        const separator = step.linkPath.includes('?') ? '&' : '?'
        const pathWithParam = `${step.linkPath}${separator}from=checklist`
        navigate(pathWithParam)
      }
    }
  }

  const projectTypeLabel = getProjectTypeLabel()
  const deployTargetLabel = getDeployTargetLabel()

  return (
    <div
      className={cn(
        'rounded-xl bg-gradient-to-r from-teal-500/50 via-cyan-500/50 to-sky-500/50 p-[2px]',
        'animate-in fade-in slide-in-from-top-2 duration-500',
        'shadow-lg shadow-teal-500/10',
        className
      )}
    >
      <div
        className={cn(
          'rounded-[10px]',
          'bg-card'
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
            const isCurrent = step.id === currentStepId

            // All cards use consistent wrapper structure for stable sizing
            return (
              <div
                key={step.id}
                className={cn(
                  'rounded-lg p-[2px]',
                  isCurrent
                    ? 'bg-gradient-to-r from-teal-500/50 via-cyan-500/50 to-sky-500/50'
                    : 'bg-border'
                )}
              >
                <div
                  className={cn(
                    'relative p-4 rounded-[6px] transition-all h-full',
                    completed
                      ? 'bg-muted/30'
                      : isCurrent
                        ? 'bg-background'
                        : 'bg-background hover:bg-muted/10'
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

                {/* Description - renders markdown links */}
                <p
                  className={cn(
                    'text-xs leading-relaxed mt-2 mb-3',
                    completed ? 'text-muted-foreground/60' : 'text-muted-foreground'
                  )}
                >
                  {description.split(/(\[.*?\]\(.*?\))/).map((part, i) => {
                    const linkMatch = part.match(/\[(.*?)\]\((.*?)\)/)
                    if (linkMatch) {
                      return (
                        <a
                          key={i}
                          href={linkMatch[2]}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline hover:text-primary/80"
                        >
                          {linkMatch[1]}
                        </a>
                      )
                    }
                    return part
                  })}
                </p>

                {/* Action button */}
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
              </div>
            )
          })}
        </div>
      </div>
      </div>
    </div>
  )
}
