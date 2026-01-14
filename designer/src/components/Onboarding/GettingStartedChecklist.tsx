/**
 * Getting Started Checklist component
 * Full-width horizontal layout at the top of the dashboard
 */

import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { X, ArrowRight, RotateCcw } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'

interface GettingStartedChecklistProps {
  className?: string
}

export function GettingStartedChecklist({
  className,
}: GettingStartedChecklistProps) {
  const navigate = useNavigate()
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
  } = useOnboardingContext()

  // Count completed steps
  const completedCount = useMemo(() => {
    return checklist.filter(step => isStepCompleted(step.id)).length
  }, [checklist, isStepCompleted])

  const handleToggleComplete = (stepId: string, completed: boolean) => {
    if (completed) {
      completeChecklistStep(stepId)
    } else {
      uncompleteChecklistStep(stepId)
    }
  }

  const handleAction = (step: typeof checklist[0]) => {
    if (step.linkLabel === 'Start over') {
      resetOnboarding()
    } else if (step.linkPath) {
      navigate(step.linkPath)
    }
  }

  const projectTypeLabel = getProjectTypeLabel()
  const deployTargetLabel = getDeployTargetLabel()

  return (
    <div
      className={cn(
        'rounded-xl border border-border bg-card',
        className
      )}
    >
      {/* Header */}
      <div className="px-5 py-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
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
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={resetOnboarding}
              className="text-muted-foreground text-xs gap-1.5"
            >
              <RotateCcw className="h-3 w-3" />
              Start over
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={dismissChecklist}
              className="h-8 w-8 p-0 text-muted-foreground"
            >
              <X className="h-4 w-4" />
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
                    checked={completed}
                    onCheckedChange={(checked) => handleToggleComplete(step.id, !!checked)}
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
                {!completed && (step.linkPath || step.linkLabel === 'Start over') && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs gap-1.5"
                    onClick={() => handleAction(step)}
                  >
                    {step.linkLabel}
                    <ArrowRight className="h-3 w-3" />
                  </Button>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
