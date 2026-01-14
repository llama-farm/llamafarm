/**
 * Data status selection screen (Step 2 of wizard)
 * Shows sample dataset picker when "sample-data" is selected
 */

import { cn } from '@/lib/utils'
import { Check, Gamepad2, FolderOpen, Search } from 'lucide-react'
import type { DataStatus } from '../../types/onboarding'
import { AVAILABLE_DEMOS } from '../../config/demos'

interface DataStatusSelectorProps {
  selected: DataStatus | null
  onSelect: (status: DataStatus) => void
  selectedSampleDataset: string | null
  onSelectSampleDataset: (demoId: string | null) => void
  className?: string
}

const dataStatusOptions = [
  {
    id: 'has-data' as DataStatus,
    icon: <FolderOpen className="w-5 h-5" />,
    title: 'Yep, ready to go!',
    description: 'I have PDFs, docs, or text files ready to upload',
    iconBg: 'bg-emerald-500/20 text-emerald-600 dark:bg-emerald-500/25 dark:text-emerald-300',
  },
  {
    id: 'sample-data' as DataStatus,
    icon: <Gamepad2 className="w-5 h-5" />,
    title: 'Let me kick the tires first',
    description: 'Use sample data so I can see how it works',
    iconBg: 'bg-violet-500/20 text-violet-600 dark:bg-violet-500/25 dark:text-violet-300',
  },
  {
    id: 'need-data' as DataStatus,
    icon: <Search className="w-5 h-5" />,
    title: 'Still gathering my data',
    description: 'Point me to some resources to get started',
    iconBg: 'bg-amber-500/20 text-amber-600 dark:bg-amber-500/25 dark:text-amber-300',
  },
]

export function DataStatusSelector({
  selected,
  onSelect,
  selectedSampleDataset,
  onSelectSampleDataset,
  className,
}: DataStatusSelectorProps) {
  return (
    <div className={cn('space-y-6', className)}>
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground">
          Do you have data to work with?
        </h2>
        <p className="mt-2 text-muted-foreground">
          Our local models are small—they need your data to shine.
        </p>
      </div>

      <div className="space-y-3">
        {dataStatusOptions.map(option => {
          const isSelected = selected === option.id
          const showSamplePicker = option.id === 'sample-data' && isSelected

          // For sample-data option, wrap everything in an expandable card
          if (option.id === 'sample-data') {
            return (
              <div
                key={option.id}
                className={cn(
                  'rounded-xl border-2 transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-white dark:bg-primary/10 shadow-md ring-1 ring-primary/20'
                    : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
                )}
              >
                <button
                  onClick={() => {
                    onSelect(option.id)
                  }}
                  className={cn(
                    'group w-full flex items-center gap-4 p-4 text-left transition-all duration-200',
                    'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 rounded-xl',
                    !isSelected && 'hover:scale-[1.01]'
                  )}
                  role="radio"
                  aria-checked={isSelected}
                >
                  {/* Icon */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                      option.iconBg,
                      isSelected ? 'scale-105' : 'group-hover:scale-105'
                    )}
                  >
                    {option.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{option.title}</div>
                    <div className="text-sm text-muted-foreground mt-0.5">
                      {option.description}
                    </div>
                  </div>

                  {/* Selection indicator */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                      isSelected
                        ? 'border-primary bg-primary'
                        : 'border-muted-foreground/30 group-hover:border-primary/50'
                    )}
                  >
                    {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
                  </div>
                </button>

                {/* Sample dataset picker - expands inside the card */}
                {showSamplePicker && (
                  <div className="px-4 pb-4 animate-in fade-in slide-in-from-top-2 duration-300">
                    <div className="pt-3 border-t border-primary/20">
                      <p className="text-sm text-muted-foreground mb-3">
                        Pick a sample dataset to start with:
                      </p>
                      <div className="space-y-2">
                        {AVAILABLE_DEMOS.map(demo => {
                          const isDemoSelected = selectedSampleDataset === demo.id
                          return (
                            <button
                              key={demo.id}
                              onClick={e => {
                                e.stopPropagation()
                                onSelectSampleDataset(demo.id)
                              }}
                              className={cn(
                                'w-full flex items-center gap-4 p-3 rounded-lg border text-left transition-all duration-200',
                                'hover:shadow-sm',
                                isDemoSelected
                                  ? 'border-primary bg-primary/5 dark:bg-primary/10'
                                  : 'border-border bg-card/50 hover:bg-white dark:hover:bg-card hover:border-primary/40'
                              )}
                            >
                              <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-violet-500/20 dark:bg-violet-500/25 flex items-center justify-center text-xl">
                                {demo.icon}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-sm text-foreground">
                                  {demo.displayName}
                                </div>
                                <div className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                                  {demo.description}
                                </div>
                              </div>
                              {/* Selection indicator */}
                              <div
                                className={cn(
                                  'flex-shrink-0 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                                  isDemoSelected
                                    ? 'border-primary bg-primary'
                                    : 'border-muted-foreground/30'
                                )}
                              >
                                {isDemoSelected && (
                                  <Check className="w-3 h-3 text-primary-foreground" />
                                )}
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          }

          // Standard option card for other options
          return (
            <button
              key={option.id}
              onClick={() => {
                onSelect(option.id)
                onSelectSampleDataset(null)
              }}
              className={cn(
                'group w-full flex items-center gap-4 p-4 rounded-xl border-2 text-left transition-all duration-200',
                'hover:scale-[1.01] hover:shadow-md',
                'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                isSelected
                  ? 'border-primary bg-white dark:bg-primary/10 shadow-md ring-1 ring-primary/20'
                  : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
              )}
              role="radio"
              aria-checked={isSelected}
            >
              {/* Icon */}
              <div
                className={cn(
                  'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                  option.iconBg,
                  isSelected ? 'scale-105' : 'group-hover:scale-105'
                )}
              >
                {option.icon}
              </div>

              <div className="flex-1 min-w-0">
                <div className="font-medium text-foreground">{option.title}</div>
                <div className="text-sm text-muted-foreground mt-0.5">
                  {option.description}
                </div>
              </div>

              {/* Selection indicator */}
              <div
                className={cn(
                  'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-primary'
                    : 'border-muted-foreground/30 group-hover:border-primary/50'
                )}
              >
                {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
