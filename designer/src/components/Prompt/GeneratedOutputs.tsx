import { useState } from 'react'
import Tabs from '../Tabs'
import RateOutput from './RateOutput'
import FontIcon from '../../common/FontIcon'

const GeneratedOutputs = () => {
  const [activeTab, setActiveTab] = useState('evaluate')

  const tempOutputs = [
    {
      output:
        'A pressure drop in the hydraulic pump during taxi on an F-16 could be caused by fluid leakage, air in the system, or a failing pressure sensor. Recommended next steps include inspecting hydraulic lines for leaks, checking fluid levels, and running a diagnostic on the pressure sensor.',
      tag: 'tag here',
    },
    {
      output:
        'Check for leaks or sensor issues. You may need to replace the pump.',
      tag: 'tag here',
    },
    {
      output:
        'This issue is likely due to avionics interference. Reset the flight computer and continue the mission.',
      tag: 'tag here',
    },
  ]

  const tempPrompts = [
    {
      version: '1.0',
      status: 'Active',
      preview:
        'You are an experienced aircraft maintenance technician with 15+ years of experience working on military and commercial aircraft. You specialize in...',
      settings: '[ ]',
    },
    {
      version: '1.1',
      status: 'Active',
      preview:
        'You are a senior aircraft maintenance specialist focused on rapid diagnosis and...',
      settings: '[ ]',
    },
    {
      version: '1.2',
      status: 'Active',
      preview:
        'You are an aircraft maintenance worker focused on diagnosis and error handling...',
      settings: '[ ]',
    },
  ]

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
      {activeTab === 'evaluate' && (
        <div className="flex flex-col justify-between h-full">
          <div>
            <div className="bg-white dark:bg-blue-500 rounded-lg p-4 flex flex-col mb-4">
              Input: The hydraulic pump on the F-16 showed a pressure drop
              during taxi. What are the most likely causes and the next steps
              for inspection?
            </div>
            <div className="mb-2">Rate outputs</div>
            <div className="flex flex-col gap-2 max-h-[320px] overflow-y-auto scrollbar-thin">
              {tempOutputs.map((output, index) => (
                <RateOutput
                  key={index}
                  output={output.output}
                  tag={output.tag}
                />
              ))}
            </div>
          </div>
          <div className="mt-4 flex flex-col gap-2">
            <div className="flex flex-wrap">
              <div className="text-sm dark:text-white mb-2 bg-blue-50 dark:bg-blue-500 rounded-2xl py-2 px-4 w-fit">
                Whats the most likely fix for Installation of a Fuel Filter
                happen?
              </div>
              <div className="text-sm dark:text-white mb-2 bg-blue-50 dark:bg-blue-500 rounded-2xl py-2 px-4 w-fit">
                whats the most costly software related aircraft error
              </div>
              <div className="text-sm dark:text-white mb-2 bg-blue-50 dark:bg-blue-500 rounded-2xl py-2 px-4 w-fit">
                whats the most common error in aircraft maintenance
              </div>
            </div>
            <div className="flex flex-row border-[1px] border-solid border-blue-100 rounded-lg p-2">
              <textarea
                className="w-full h-18 bg-transparent border-none rounded-lg px-4 py-2 text-lg resize-none focus:outline-none"
                placeholder="Try another input"
              />
              <button className="bg-blue-100 text-white rounded-lg px-4 py-2 w-fit self-start">
                Submit
              </button>
            </div>
            <label className="text-sm text-blue-100">
              Not sure where to start? Think about the questions your users will
              actually ask to test model reliability.
            </label>
          </div>
        </div>
      )}
      {activeTab === 'prompts' && (
        <div className="w-full h-full">
          <table className="w-full">
            <thead className="bg-white dark:bg-blue-600 font-normal">
              <tr>
                <th className="text-left w-[10%] py-2 px-3">Version</th>
                <th className="text-left w-[10%] py-2 px-3">Status</th>
                <th className="text-left w-[50%] py-2 px-3">Preview</th>
                <th className="text-left w-[10%] py-2 px-3">Settings</th>
                <th className="text-left w-[10%] py-2 px-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {tempPrompts.map((prompt, index) => (
                <tr
                  key={index}
                  className={`border-b border-solid border-white dark:border-blue-600 text-sm font-mono leading-4 tracking-[0.32px]${
                    index === tempPrompts.length - 1 ? 'border-b-0' : 'border-b'
                  }`}
                >
                  <td className="align-top p-3">{prompt.version}</td>
                  <td className="align-top p-3">
                    <FontIcon
                      type="checkmark-outline"
                      className="w-6 h-6 text-blue-100 dark:text-green-100"
                    />
                  </td>
                  <td className="align-top p-3">{prompt.preview}</td>
                  <td className="align-top p-3">{prompt.settings}</td>
                  <td className="flex flex-row gap-4 align-top p-3">
                    <FontIcon
                      type="edit"
                      className="w-6 h-6 text-blue-100"
                      isButton
                    />
                    <FontIcon
                      type="trashcan"
                      className="w-6 h-6 text-blue-100"
                      isButton
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {activeTab === 'model' && <div>Model</div>}
    </div>
  )
}

export default GeneratedOutputs
