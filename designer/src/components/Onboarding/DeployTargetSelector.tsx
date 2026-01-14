/**
 * Deploy target selection screen (Step 3 of wizard)
 */

import { RadioSelector } from './RadioSelector'
import type { DeployTarget } from '../../types/onboarding'

const deployTargetOptions = [
  {
    id: 'local' as DeployTarget,
    title: 'On my own turf',
    description: 'Local machine, on-prem servers, or air-gapped',
    emoji: '🏠',
  },
  {
    id: 'cloud' as DeployTarget,
    title: 'Up in the cloud',
    description: 'AWS, GCP, Azure, or similar',
    emoji: '☁️',
  },
  {
    id: 'tbd' as DeployTarget,
    title: 'Haven\'t decided yet',
    description: 'No worries, we\'ll figure it out together',
    emoji: '🤔',
  },
]

interface DeployTargetSelectorProps {
  selected: DeployTarget | null
  onSelect: (target: DeployTarget) => void
  className?: string
}

export function DeployTargetSelector({
  selected,
  onSelect,
  className,
}: DeployTargetSelectorProps) {
  return (
    <RadioSelector
      title="Where will this run?"
      options={deployTargetOptions}
      selected={selected}
      onSelect={onSelect}
      className={className}
    />
  )
}
