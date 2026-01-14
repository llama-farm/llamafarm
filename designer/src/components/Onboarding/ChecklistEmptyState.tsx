/**
 * Empty state component shown when wizard is skipped or checklist is dismissed
 * Provides a way back into the onboarding wizard
 */

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ArrowRight } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'

interface ChecklistEmptyStateProps {
  className?: string
}

export function ChecklistEmptyState({ className }: ChecklistEmptyStateProps) {
  const { openWizard } = useOnboardingContext()

  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-card p-8 text-center',
        className
      )}
    >
      <div className="text-6xl mb-4">🦙</div>
      <h3 className="text-lg font-medium text-foreground mb-2">
        Not sure where to start?
      </h3>
      <p className="text-sm text-muted-foreground mb-4">
        Let us help you get set up with a personalized guide.
      </p>
      <Button onClick={openWizard} className="gap-2">
        Show me what to do
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
