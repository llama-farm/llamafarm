import { useState } from 'react'
import Tabs from '../../Tabs'
import Evaluate from './Evaluate'
import Prompts from './Prompts'
import Model from './Model'

const GeneratedOutputs = () => {
  const [activeTab, setActiveTab] = useState('evaluate')

  return (
    <div className="w-full h-full flex flex-col gap-2 pb-4">
      <div>Prompt</div>
      <div className="flex flex-row mb-4">
        <Tabs
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          tabs={[
            { id: 'evaluate', label: 'Evaluate' },
            { id: 'prompts', label: 'Prompts' },
            { id: 'model', label: 'Model' },
          ]}
        />
        <div className="flex flex-col border-[1px] border-solid border-blue-50 dark:border-blue-600 rounded-xl p-2 ml-10">
          <div className="text-2xl text-blue-100 dark:text-green-100">23%</div>
          <div className="text-sm text-blue-100 dark:text-green-100">
            accuracy
          </div>
        </div>
      </div>
      {activeTab === 'evaluate' && <Evaluate />}
      {activeTab === 'prompts' && <Prompts />}
      {activeTab === 'model' && <Model />}
    </div>
  )
}

export default GeneratedOutputs
