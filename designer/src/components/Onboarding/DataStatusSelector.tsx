/**
 * Data status selection screen (Step 2 of wizard)
 */

import { RadioSelector } from './RadioSelector'
import type { DataStatus } from '../../types/onboarding'

const dataStatusOptions = [
  {
    id: 'has-data' as DataStatus,
    title: 'Yep, ready to go!',
    description: 'I have PDFs, docs, or text files ready to upload',
    emoji: '📁',
  },
  {
    id: 'sample-data' as DataStatus,
    title: 'Let me kick the tires first',
    description: 'Use sample data so I can see how it works',
    emoji: '🎮',
  },
  {
    id: 'need-data' as DataStatus,
    title: 'Still gathering my data',
    description: 'Point me to some resources to get started',
    emoji: '🔎',
  },
]

interface DataStatusSelectorProps {
  selected: DataStatus | null
  onSelect: (status: DataStatus) => void
  className?: string
}

export function DataStatusSelector({
  selected,
  onSelect,
  className,
}: DataStatusSelectorProps) {
  return (
    <RadioSelector
      title="Do you have data to work with?"
      subtitle="Our local models are small—they need your data to shine."
      options={dataStatusOptions}
      selected={selected}
      onSelect={onSelect}
      className={className}
    />
  )
}
