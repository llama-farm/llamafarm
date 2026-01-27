/**
 * Collapsed checklist banner - shown when the full checklist is collapsed
 * Provides a compact view with progress and a way to expand
 */

import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ChevronDown, RotateCcw } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'

interface CollapsedChecklistProps {
  className?: string
}

export function CollapsedChecklist({ className }: CollapsedChecklistProps) {
  const {
    checklist,
    isStepCompleted,
    showChecklist,
    resetOnboarding,
    getProjectTypeLabel,
    getDeployTargetLabel,
  } = useOnboardingContext()

  // Count completed steps
  const completedCount = useMemo(() => {
    return checklist.filter(step => isStepCompleted(step.id)).length
  }, [checklist, isStepCompleted])

  const projectTypeLabel = getProjectTypeLabel()
  const deployTargetLabel = getDeployTargetLabel()
  const allDone = completedCount === checklist.length

  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-card px-4 py-3',
        'flex items-center justify-between gap-4',
        className
      )}
    >
      <div className="flex items-center gap-3">
        <span className="text-2xl">🦙</span>
        <div className="flex items-center gap-2">
          <span className="font-medium text-foreground">Getting Started</span>
          {(projectTypeLabel || deployTargetLabel) && (
            <span className="text-xs text-muted-foreground">
              {projectTypeLabel}
              {deployTargetLabel && ` • ${deployTargetLabel}`}
            </span>
          )}
          <span className="text-xs text-muted-foreground">
            • {completedCount}/{checklist.length} done
          </span>
          {allDone && (
            <span className="text-xs text-primary font-medium">All complete!</span>
          )}
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
          variant="outline"
          size="sm"
          onClick={showChecklist}
          className="gap-1.5"
        >
          <ChevronDown className="h-4 w-4" />
          Show checklist
        </Button>
      </div>
    </div>
  )
}
