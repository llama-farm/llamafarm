import RateOutput from '../RateOutput'

const Evaluate = () => {
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
  return (
    <div className="flex flex-col justify-between h-full">
      <div>
        <div className="bg-white dark:bg-blue-500 rounded-lg p-4 flex flex-col mb-4">
          Input: The hydraulic pump on the F-16 showed a pressure drop during
          taxi. What are the most likely causes and the next steps for
          inspection?
        </div>
        <div className="mb-2">Rate outputs</div>
        <div className="flex flex-col gap-2 max-h-[320px] overflow-y-auto scrollbar-thin">
          {tempOutputs.map((output, index) => (
            <RateOutput key={index} output={output.output} tag={output.tag} />
          ))}
        </div>
      </div>
      <div className="mt-4 flex flex-col gap-2">
        <div className="flex flex-wrap">
          <div className="text-sm dark:text-white mb-2 bg-blue-50 dark:bg-blue-500 rounded-2xl py-2 px-4 w-fit">
            Whats the most likely fix for Installation of a Fuel Filter happen?
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
  )
}

export default Evaluate
