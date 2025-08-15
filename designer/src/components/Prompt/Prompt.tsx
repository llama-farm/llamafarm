import { useState } from 'react'
import ModeToggle, { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor'
import GeneratedOutputs from './GeneratedOutput/GeneratedOutputs'

const Prompt = () => {
  const [hasGeneratedOutputs, setHasGeneratedOutputs] = useState(false)
  const [mode, setMode] = useState<Mode>('designer')

  if (hasGeneratedOutputs) {
    return <GeneratedOutputs />
  }

  return (
    <div className="h-full w-full flex flex-col gap-2">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Prompt' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <button className="opacity-50 cursor-not-allowed text-sm px-3 py-2 rounded-lg border border-blue-50 text-blue-50 dark:text-blue-100 dark:border-blue-400">
            Deploy
          </button>
        </div>
      </div>
      {mode === 'designer' && (
        <div className="text-blue-100">
          Test, tune, and train. Add inputs, review outputs, and give feedback
          to shape better prompts and guide the model toward your ideal results.
        </div>
      )}
      {mode === 'designer' ? (
        <div className="bg-white dark:bg-blue-500 rounded-lg p-4 flex flex-col">
          <div className="mb-2">Expected outputs</div>
          <div className="text-sm mb-6">
            Help us generate better results, faster
          </div>
          <label className="text-sm mb-2">
            What kind of output are you hoping for?
          </label>
          <textarea
            className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 text-lg mb-6 code-like"
            placeholder="e.g. “Summarize each entry,” “Generate a visual timeline,” “Classify logs,” “Extract sensor anomalies,” “Create a chart,” “Turn this into a checklist,” etc."
          />
          <label className="text-xs text-gray-100 mb-2">Example input</label>
          <input
            className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 mb-1 code-like"
            placeholder="Enter here"
          />
          <label className="text-xs text-blue-100 mb-4">
            Connect your projects to OpenAI
          </label>
          <div className="mb-3">
            What are some expected outputs you'd hope to see?
          </div>
          <label className="text-xs text-gray-100 mb-2">
            Example ideal outputs
          </label>
          <div className="flex flex-col gap-2 mb-6">
            <input
              className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 code-like"
              placeholder="Enter here"
            />
            <input
              className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 code-like"
              placeholder="Enter here"
            />
            <input
              className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 code-like"
              placeholder="Enter here"
            />
          </div>
          <button
            className="bg-blue-200 text-white rounded-lg px-4 py-2 w-fit self-end"
            onClick={() => setHasGeneratedOutputs(true)}
          >
            Generate outputs
          </button>
        </div>
      ) : (
        <ConfigEditor />
      )}
    </div>
  )
}

export default Prompt
