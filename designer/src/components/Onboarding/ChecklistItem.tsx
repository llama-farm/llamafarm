/**
 * Individual checklist item component
 * Compact design for side panel layout
 */

import { useNavigate } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { ArrowRight } from 'lucide-react'
import type { ChecklistStep } from '../../types/onboarding'

interface ChecklistItemProps {
  step: ChecklistStep
  description: string
  isCompleted: boolean
  isCurrent: boolean
  onToggleComplete: (stepId: string, completed: boolean) => void
  onStartOver?: () => void
  className?: string
}

export function ChecklistItem({
  step,
  description,
  isCompleted,
  isCurrent,
  onToggleComplete,
  onStartOver,
  className,
}: ChecklistItemProps) {
  const navigate = useNavigate()

  const handleAction = () => {
    if (step.linkLabel === 'Start over' && onStartOver) {
      onStartOver()
    } else if (step.linkPath) {
      navigate(step.linkPath)
    }
  }

  const handleCheckboxChange = (checked: boolean) => {
    onToggleComplete(step.id, checked)
  }

  return (
    <div
      className={cn(
        'p-3 rounded-lg border transition-all',
        isCompleted
          ? 'bg-muted/50 border-border opacity-60'
          : isCurrent
            ? 'bg-accent/20 border-primary/30'
            : 'bg-card border-border',
        className
      )}
    >
      <div className="flex items-start gap-3">
        {/* Checkbox */}
        <div className="flex-shrink-0 pt-0.5">
          <Checkbox
            checked={isCompleted}
            onCheckedChange={handleCheckboxChange}
            aria-label={`Mark "${step.title}" as ${isCompleted ? 'incomplete' : 'complete'}`}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <span
            className={cn(
              'text-sm font-medium leading-tight',
              isCompleted ? 'text-muted-foreground line-through' : 'text-foreground'
            )}
          >
            {step.stepNumber}. {step.title}
          </span>

          <p
            className={cn(
              'mt-1 text-xs leading-snug',
              isCompleted ? 'text-muted-foreground/70' : 'text-muted-foreground'
            )}
          >
            {description}
          </p>

          {/* Action button */}
          {!isCompleted && (step.linkPath || step.linkLabel === 'Start over') && (
            <Button
              variant="link"
              size="sm"
              className="mt-1.5 h-auto p-0 text-xs text-primary"
              onClick={handleAction}
            >
              {step.linkLabel}
              <ArrowRight className="ml-1 h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
