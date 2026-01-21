/**
 * Progress indicator for the onboarding wizard
 * Shows current step out of total steps
 */

import { cn } from '@/lib/utils'

interface WizardProgressProps {
  currentStep: number
  totalSteps: number
  className?: string
}

export function WizardProgress({
  currentStep,
  totalSteps,
  className,
}: WizardProgressProps) {
  return (
    <div className={cn('flex items-center gap-3', className)}>
      <span className="text-sm text-muted-foreground">
        Step {currentStep} of {totalSteps}
      </span>
      <div className="flex items-center gap-1.5">
        {Array.from({ length: totalSteps }, (_, i) => (
          <div
            key={i}
            className={cn(
              'h-2 w-2 rounded-full transition-colors',
              i + 1 === currentStep
                ? 'bg-primary'
                : i + 1 < currentStep
                  ? 'bg-primary/50'
                  : 'bg-muted-foreground/30'
            )}
          />
        ))}
      </div>
    </div>
  )
}
