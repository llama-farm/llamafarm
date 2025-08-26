import { useState } from 'react'
import Tabs from '../../Tabs'
import Evaluate from './Evaluate'
import Prompts from './Prompts'
// Model tab removed

const GeneratedOutputs = () => {
  const [activeTab, setActiveTab] = useState('evaluate')

  return (
    <div className="w-full h-full flex flex-col gap-2 pb-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl ">Prompt</h2>
      </div>
      <div className="flex flex-row items-center mb-4">
        <Tabs
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          tabs={[
            { id: 'evaluate', label: 'Evaluate' },
            { id: 'prompts', label: 'Prompts' },
          ]}
        />
        <div className="flex flex-col border border-input bg-card rounded-xl p-2 ml-6 min-w-20 items-center">
          <div className="text-2xl text-foreground">23%</div>
          <div className="text-xs text-muted-foreground">accuracy</div>
        </div>
      </div>
      {activeTab === 'evaluate' && <Evaluate />}
      {activeTab === 'prompts' && <Prompts />}
    </div>
  )
}

export default GeneratedOutputs
