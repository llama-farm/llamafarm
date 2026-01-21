/**
 * Reusable radio-style selector for wizard screens 2-4
 * Card-based radio buttons with title and description
 */

import { cn } from '@/lib/utils'
import { Check } from 'lucide-react'

interface RadioOption<T extends string> {
  id: T
  title: string
  description: string
  icon?: React.ReactNode
  iconBg?: string
}

interface RadioSelectorProps<T extends string> {
  title: string
  subtitle?: string
  options: RadioOption<T>[]
  selected: T | null
  onSelect: (value: T) => void
  className?: string
}

export function RadioSelector<T extends string>({
  title,
  subtitle,
  options,
  selected,
  onSelect,
  className,
}: RadioSelectorProps<T>) {
  return (
    <div className={cn('space-y-6', className)}>
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground">{title}</h2>
        {subtitle && (
          <p className="mt-2 text-muted-foreground">{subtitle}</p>
        )}
      </div>

      <div className="space-y-3">
        {options.map((option, index) => (
          <button
            type="button"
            key={option.id}
            onClick={() => onSelect(option.id)}
            className={cn(
              'group w-full flex items-center gap-4 p-4 rounded-xl border-2 text-left transition-all duration-200',
              'hover:scale-[1.01] hover:shadow-md',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
              'bg-card',
              selected === option.id
                ? 'border-primary shadow-md'
                : 'border-border hover:border-primary/40'
            )}
            role="radio"
            aria-checked={selected === option.id}
          >
            {/* Icon or number indicator */}
            <div className={cn(
              'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
              option.iconBg || (selected === option.id
                ? 'bg-primary/20'
                : 'bg-muted group-hover:bg-primary/10'),
              selected === option.id ? 'scale-105' : 'group-hover:scale-105'
            )}>
              {option.icon || (
                <span className={cn(
                  'text-lg font-semibold',
                  selected === option.id ? 'text-primary' : 'text-muted-foreground'
                )}>
                  {index + 1}
                </span>
              )}
            </div>

            <div className="flex-1 min-w-0">
              <div className="font-medium text-foreground">{option.title}</div>
              <div className="text-sm text-muted-foreground mt-0.5">
                {option.description}
              </div>
            </div>

            {/* Selection indicator */}
            <div className={cn(
              'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
              selected === option.id
                ? 'border-primary bg-primary'
                : 'border-muted-foreground/30 group-hover:border-primary/50'
            )}>
              {selected === option.id && (
                <Check className="w-4 h-4 text-primary-foreground" />
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
