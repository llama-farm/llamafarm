import { useState } from 'react'
import GeneratedOutputs from './GeneratedOutput/GeneratedOutputs'

const Prompt = () => {
  const [hasGeneratedOutputs, setHasGeneratedOutputs] = useState(false)

  if (hasGeneratedOutputs) {
    return <GeneratedOutputs />
  }

  return (
    <div className="h-full w-full flex flex-col gap-2">
      <div>Prompt</div>
      <div className="text-blue-100">
        Test, tune, and train. Add inputs, review outputs, and give feedback to
        shape better prompts and guide the model toward your ideal results.
      </div>
      <div className="bg-white dark:bg-blue-500 rounded-lg p-4 flex flex-col">
        <div className="mb-2">Expected outputs</div>
        <div className="text-sm mb-6">
          Help us generate better results, faster
        </div>
        <label className="text-sm mb-2">
          What kind of output are you hoping for?
        </label>
        <textarea
          className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 text-lg mb-6"
          placeholder="e.g. “Summarize each entry,” “Generate a visual timeline,” “Classify logs,” “Extract sensor anomalies,” “Create a chart,” “Turn this into a checklist,” etc."
        />
        <label className="text-xs text-gray-100 mb-2">Example input</label>
        <input
          className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 mb-1"
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
            className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 "
            placeholder="Enter here"
          />
          <input
            className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 "
            placeholder="Enter here"
          />
          <input
            className="w-full h-18 bg-gray-200 dark:bg-blue-600 border-none rounded-lg px-4 py-2 "
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
    </div>
  )
}

export default Prompt
