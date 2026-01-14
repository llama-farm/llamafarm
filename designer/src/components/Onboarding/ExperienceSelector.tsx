/**
 * Experience level selection screen (Step 4 of wizard)
 */

import { RadioSelector } from './RadioSelector'
import type { ExperienceLevel } from '../../types/onboarding'

const experienceOptions = [
  {
    id: 'beginner' as ExperienceLevel,
    title: 'Hold my hand',
    description: 'Walk me through everything step by step',
    emoji: '🤝',
  },
  {
    id: 'intermediate' as ExperienceLevel,
    title: 'Just nudge me along',
    description: 'I know the basics, point me in the right direction',
    emoji: '👉',
  },
  {
    id: 'advanced' as ExperienceLevel,
    title: 'Get out of my way',
    description: 'Just give me the checklist, I\'ve got this',
    emoji: '🚀',
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
