/**
 * Deploy target selection screen (Step 3 of wizard)
 */

import { RadioSelector } from './RadioSelector'
import type { DeployTarget } from '../../types/onboarding'
import { Home, Cloud, HelpCircle } from 'lucide-react'

const deployTargetOptions = [
  {
    id: 'local' as DeployTarget,
    icon: <Home className="w-5 h-5" />,
    iconBg: 'bg-blue-500/20 text-blue-600 dark:bg-blue-500/25 dark:text-blue-300',
    title: 'On my own turf',
    description: 'Local machine, on-prem servers, or air-gapped',
  },
  {
    id: 'cloud' as DeployTarget,
    icon: <Cloud className="w-5 h-5" />,
    iconBg: 'bg-cyan-500/20 text-cyan-600 dark:bg-cyan-500/25 dark:text-cyan-300',
    title: 'Up in the cloud',
    description: 'AWS, GCP, Azure, or similar',
  },
  {
    id: 'tbd' as DeployTarget,
    icon: <HelpCircle className="w-5 h-5" />,
    iconBg: 'bg-slate-500/20 text-slate-600 dark:bg-slate-500/25 dark:text-slate-300',
    title: "Haven't decided yet",
    description: "No worries, we'll figure it out together",
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
