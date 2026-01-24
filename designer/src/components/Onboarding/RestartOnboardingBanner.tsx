/**
 * Narrow full-width banner shown when onboarding was skipped/dismissed
 * Provides a way to restart the onboarding wizard
 */

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Sparkles } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'

interface RestartOnboardingBannerProps {
  className?: string
}

export function RestartOnboardingBanner({
  className,
}: RestartOnboardingBannerProps) {
  const { openWizard } = useOnboardingContext()

  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-gradient-to-r from-primary/5 to-accent/10 px-4 py-3',
        'flex items-center justify-between gap-4',
        className
      )}
    >
      <div className="flex items-center gap-3">
        <span className="text-2xl">🦙</span>
        <div>
          <span className="font-medium text-foreground">
            Need help getting started?
          </span>
          <span className="text-muted-foreground ml-2 text-sm">
            We can build you a personalized setup guide.
          </span>
        </div>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={openWizard}
        className="gap-2 shrink-0"
      >
        <Sparkles className="h-4 w-4" />
        Show me
      </Button>
    </div>
  )
}
