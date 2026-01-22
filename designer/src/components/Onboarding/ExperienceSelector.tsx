/**
 * Experience level selection screen (Step 4 of wizard)
 */

import { RadioSelector } from './RadioSelector'
import type { ExperienceLevel } from '../../types/onboarding'
import { HandHeart, Navigation, Rocket } from 'lucide-react'

const experienceOptions = [
  {
    id: 'beginner' as ExperienceLevel,
    icon: <HandHeart className="w-5 h-5" />,
    iconBg: 'bg-amber-500/20 text-amber-600 dark:bg-amber-500/25 dark:text-amber-300',
    title: 'Hold my hand',
    description: 'Walk me through everything step by step',
  },
  {
    id: 'intermediate' as ExperienceLevel,
    icon: <Navigation className="w-5 h-5" />,
    iconBg: 'bg-teal-500/20 text-teal-600 dark:bg-teal-500/25 dark:text-teal-300',
    title: 'Just nudge me along',
    description: 'I know the basics, point me in the right direction',
  },
  {
    id: 'advanced' as ExperienceLevel,
    icon: <Rocket className="w-5 h-5" />,
    iconBg: 'bg-rose-500/20 text-rose-600 dark:bg-rose-500/25 dark:text-rose-300',
    title: 'Get out of my way',
    description: "Just give me the checklist, I've got this",
  },
]

interface ExperienceSelectorProps {
  selected: ExperienceLevel | null
  onSelect: (level: ExperienceLevel) => void
  className?: string
}

export function ExperienceSelector({
  selected,
  onSelect,
  className,
}: ExperienceSelectorProps) {
  return (
    <RadioSelector
      title="How much guidance do you want?"
      options={experienceOptions}
      selected={selected}
      onSelect={onSelect}
      className={className}
    />
  )
}
