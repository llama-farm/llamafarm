import { useState } from 'react'
import { cn } from '@/lib/utils'
import { DetectPanel } from './DetectPanel'
import { ClassifyPanel } from './ClassifyPanel'
import { DetectClassifyPanel } from './DetectClassifyPanel'
import { TrainingPanel } from './TrainingPanel'
import { ModelsPanel } from './ModelsPanel'
import { StreamingPanel } from './StreamingPanel'
import { ReviewPanel } from './ReviewPanel'

const VISION_TABS = [
  { id: 'detect', label: 'Detect' },
  { id: 'classify', label: 'Classify' },
  { id: 'detect-classify', label: 'Detect+Classify' },
  { id: 'train', label: 'Train' },
  { id: 'models', label: 'Models' },
  { id: 'stream', label: 'Stream' },
  { id: 'review', label: 'Review' },
] as const

type VisionTab = (typeof VISION_TABS)[number]['id']

export function VisionPanel() {
  const [activeTab, setActiveTab] = useState<VisionTab>('detect')

  return (
    <div className="flex flex-col gap-4 pt-4">
      {/* Inner tab bar */}
      <div className="flex items-center gap-1 border-b border-border">
        {VISION_TABS.map(tab => (
          <button
            key={tab.id}
            className={cn(
              'px-3 py-2 -mb-[1px] border-b-2 transition-colors text-sm rounded-t-md',
              activeTab === tab.id
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="min-h-0">
        {activeTab === 'detect' && <DetectPanel />}
        {activeTab === 'classify' && <ClassifyPanel />}
        {activeTab === 'detect-classify' && <DetectClassifyPanel />}
        {activeTab === 'train' && <TrainingPanel />}
        {activeTab === 'models' && <ModelsPanel />}
        {activeTab === 'stream' && <StreamingPanel />}
        {activeTab === 'review' && <ReviewPanel />}
      </div>
    </div>
  )
}
