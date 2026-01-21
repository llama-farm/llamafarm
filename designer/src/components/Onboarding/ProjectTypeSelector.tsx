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
    title: 'Chat with documents',
    subtitle: 'Good for: customer support bots, knowledge bases, FAQ assistants',
    gradient: 'from-blue-500/20 to-cyan-500/20',
    iconBg: 'bg-blue-500/20 text-blue-400 dark:bg-blue-500/25 dark:text-blue-300',
  },
  {
    id: 'classifier',
    icon: <Tag className="w-7 h-7" />,
    title: 'Sort & label content',
    subtitle: 'Good for: ticket routing, content tagging, sentiment analysis',
    gradient: 'from-purple-500/20 to-pink-500/20',
    iconBg: 'bg-purple-500/20 text-purple-400 dark:bg-purple-500/25 dark:text-purple-300',
  },
  {
    id: 'anomaly',
    icon: <AlertTriangle className="w-7 h-7" />,
    title: 'Spot the odd ones out',
    subtitle: 'Good for: fraud detection, quality control, outlier analysis',
    gradient: 'from-amber-500/20 to-orange-500/20',
    iconBg: 'bg-amber-500/20 text-amber-400 dark:bg-amber-500/25 dark:text-amber-300',
  },
  {
    id: 'doc-scan',
    icon: <FileText className="w-7 h-7" />,
    title: 'Extract info from docs',
    subtitle: 'Good for: invoice processing, resume parsing, form extraction',
    gradient: 'from-emerald-500/20 to-teal-500/20',
    iconBg: 'bg-emerald-500/20 text-emerald-400 dark:bg-emerald-500/25 dark:text-emerald-300',
  },
  {
    id: 'exploring',
    icon: <FlaskConical className="w-7 h-7" />,
    title: 'Just poking around',
    subtitle: 'Good for: learning the platform and experimenting with AI',
    gradient: 'from-rose-500/20 to-violet-500/20',
    iconBg: 'bg-rose-500/20 text-rose-400 dark:bg-rose-500/25 dark:text-rose-300',
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

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {projectTypes.filter(option => option.id !== 'doc-scan').map(option => (
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

            {/* Selection indicator - inside card */}
            {selected === option.id && (
              <div className="absolute top-3 right-3 w-5 h-5 bg-primary rounded-full flex items-center justify-center">
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
