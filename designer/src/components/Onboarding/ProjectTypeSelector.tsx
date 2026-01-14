/**
 * Project type selection screen (Step 1 of wizard)
 * Grid of selectable cards for different project types
 */

import { cn } from '@/lib/utils'
import type { ProjectType } from '../../types/onboarding'
import {
  MessageSquare,
  Tag,
  AlertTriangle,
  FileText,
  FlaskConical,
} from 'lucide-react'

interface ProjectTypeOption {
  id: ProjectType
  icon: React.ReactNode
  title: string
  subtitle: string
  gradient: string
  iconBg: string
}

const projectTypes: ProjectTypeOption[] = [
  {
    id: 'doc-qa',
    icon: <MessageSquare className="w-7 h-7" />,
    title: 'Chat with my documents',
    subtitle: 'Ask questions, get answers from your files',
    gradient: 'from-blue-500/20 to-cyan-500/20',
    iconBg: 'bg-blue-500/15 text-blue-600 dark:text-blue-400',
  },
  {
    id: 'classifier',
    icon: <Tag className="w-7 h-7" />,
    title: 'Sort & label content',
    subtitle: 'Teach AI to categorize like you do',
    gradient: 'from-purple-500/20 to-pink-500/20',
    iconBg: 'bg-purple-500/15 text-purple-600 dark:text-purple-400',
  },
  {
    id: 'anomaly',
    icon: <AlertTriangle className="w-7 h-7" />,
    title: 'Spot the odd ones out',
    subtitle: 'Find what doesn\'t belong',
    gradient: 'from-amber-500/20 to-orange-500/20',
    iconBg: 'bg-amber-500/15 text-amber-600 dark:text-amber-400',
  },
  {
    id: 'doc-scan',
    icon: <FileText className="w-7 h-7" />,
    title: 'Extract info from docs',
    subtitle: 'Pull structured data from messy files',
    gradient: 'from-emerald-500/20 to-teal-500/20',
    iconBg: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400',
  },
  {
    id: 'exploring',
    icon: <FlaskConical className="w-7 h-7" />,
    title: 'Just poking around',
    subtitle: 'Show me what this thing can do!',
    gradient: 'from-rose-500/20 to-violet-500/20',
    iconBg: 'bg-rose-500/15 text-rose-600 dark:text-rose-400',
  },
]

interface ProjectTypeSelectorProps {
  selected: ProjectType | null
  onSelect: (type: ProjectType) => void
  className?: string
}

export function ProjectTypeSelector({
  selected,
  onSelect,
  className,
}: ProjectTypeSelectorProps) {
  return (
    <div className={cn('space-y-6', className)}>
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground">
          What are you building?
        </h2>
        <p className="mt-2 text-muted-foreground">
          Pick the one that fits best. You can always change your mind later.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {projectTypes.map(option => (
          <button
            key={option.id}
            onClick={() => onSelect(option.id)}
            className={cn(
              'group relative flex flex-col items-center text-center p-6 rounded-xl border-2 transition-all duration-200',
              'hover:scale-[1.02] hover:shadow-lg',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
              selected === option.id
                ? 'border-primary bg-gradient-to-br ' + option.gradient + ' shadow-md'
                : 'border-border bg-card hover:border-primary/40'
            )}
            aria-pressed={selected === option.id}
          >
            {/* Centered icon */}
            <div className={cn(
              'w-14 h-14 rounded-xl flex items-center justify-center mb-4 transition-all duration-200',
              option.iconBg,
              selected === option.id ? 'scale-110' : 'group-hover:scale-110'
            )}>
              {option.icon}
            </div>

            <div className="font-medium text-foreground">{option.title}</div>
            <div className="text-sm text-muted-foreground mt-1">
              {option.subtitle}
            </div>

            {/* Selection indicator */}
            {selected === option.id && (
              <div className="absolute -top-1 -right-1 w-5 h-5 bg-primary rounded-full flex items-center justify-center">
                <svg className="w-3 h-3 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
