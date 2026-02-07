import { useState } from 'react'
import { StreamingPanel } from './StreamingPanel'
import { TrainingPanel } from './TrainingPanel'
import { ModelsPanel } from './ModelsPanel'
import { ReviewPanel } from './ReviewPanel'
import FontIcon from '../../common/FontIcon'

type VisionTab = 'analyze' | 'training' | 'models' | 'review'

const TABS: { id: VisionTab; label: string; icon: string; description: string }[] = [
  { id: 'analyze', label: 'Analyze', icon: 'eye', description: 'Detect, classify, segment images' },
  { id: 'training', label: 'Training', icon: 'tools-alt', description: 'Train and fine-tune models' },
  { id: 'models', label: 'Models', icon: 'model', description: 'Manage saved models' },
  { id: 'review', label: 'Review', icon: 'user-avatar', description: 'Review uncertain detections' },
]

export default function Vision() {
  const [activeTab, setActiveTab] = useState<VisionTab>('analyze')
  
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border">
        <h1 className="text-xl font-semibold">Vision</h1>
        <p className="text-sm text-muted-foreground">
          Computer vision with auto-learning cascade
        </p>
      </div>
      
      {/* Tab Bar */}
      <div className="px-6 py-2 border-b border-border">
        <div className="flex gap-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === tab.id 
                  ? 'bg-primary text-primary-foreground' 
                  : 'hover:bg-muted'
              }`}
            >
              <FontIcon type={tab.icon as any} className="w-4 h-4" />
              <span className="font-medium">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {activeTab === 'analyze' && <StreamingPanel />}
        {activeTab === 'training' && <TrainingPanel />}
        {activeTab === 'models' && <ModelsPanel />}
        {activeTab === 'review' && <ReviewPanel />}
      </div>
    </div>
  )
}
