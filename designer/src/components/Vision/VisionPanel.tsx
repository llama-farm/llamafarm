import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Search, GraduationCap, Database, Video, CheckCircle } from 'lucide-react'
import { AnalyzePanel } from './AnalyzePanel'
import { TrainingPanel } from './TrainingPanel'
import { ModelsPanel } from './ModelsPanel'
import { StreamingPanel } from './StreamingPanel'
import { ReviewPanel } from './ReviewPanel'

const VISION_TABS = [
  { id: 'analyze', label: 'Analyze', icon: Search, description: 'Upload and analyze images' },
  { id: 'train', label: 'Train', icon: GraduationCap, description: 'Train custom models' },
  { id: 'models', label: 'Models', icon: Database, description: 'Manage saved models' },
  { id: 'stream', label: 'Stream', icon: Video, description: 'Real-time detection' },
  { id: 'review', label: 'Review', icon: CheckCircle, description: 'Review detections' },
] as const

type VisionTab = (typeof VISION_TABS)[number]['id']

export function VisionPanel() {
  const [activeTab, setActiveTab] = useState<VisionTab>('analyze')

  return (
    <div className="flex h-full min-h-[400px]">
      {/* Left sidebar navigation */}
      <div className="w-48 flex-shrink-0 border-r border-border bg-muted/20 py-2">
        <div className="px-3 py-2 mb-1">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Vision</span>
        </div>
        {VISION_TABS.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              className={cn(
                'w-full flex items-center gap-2.5 px-3 py-2 text-sm transition-colors text-left',
                activeTab === tab.id
                  ? 'bg-primary/10 text-foreground font-medium border-r-2 border-primary'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              )}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>

      {/* Content area */}
      <div className="flex-1 min-w-0 overflow-y-auto p-4">
        {activeTab === 'analyze' && <AnalyzePanel />}
        {activeTab === 'train' && <TrainingPanel />}
        {activeTab === 'models' && <ModelsPanel onNavigateToTrain={() => setActiveTab('train')} />}
        {activeTab === 'stream' && <StreamingPanel />}
        {activeTab === 'review' && <ReviewPanel />}
      </div>
    </div>
  )
}
