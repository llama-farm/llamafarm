import RateOutput from '../RateOutput'
import { Textarea } from '../../ui/textarea'
import { Button } from '../../ui/button'

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
    <div className="flex flex-col h-full">
      <div className="bg-card border border-input rounded-lg p-4 flex flex-col mb-4 text-foreground">
        Input: The hydraulic pump on the F-16 showed a pressure drop during
        taxi. What are the most likely causes and the next steps for inspection?
      </div>
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="mb-2 text-foreground">Rate outputs</div>
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin pr-1">
          <div className="flex flex-col gap-2 pb-4">
            {tempOutputs.map((output, index) => (
              <RateOutput key={index} output={output.output} tag={output.tag} />
            ))}
          </div>
        </div>
        <div className="mt-4 flex flex-col gap-2">
          <div className="flex flex-wrap">
            <div className="text-sm mb-2 bg-accent text-accent-foreground rounded-2xl py-2 px-4 w-fit">
              Whats the most likely fix for Installation of a Fuel Filter
              happen?
            </div>
            <div className="text-sm mb-2 bg-accent text-accent-foreground rounded-2xl py-2 px-4 w-fit">
              whats the most costly software related aircraft error
            </div>
            <div className="text-sm mb-2 bg-accent text-accent-foreground rounded-2xl py-2 px-4 w-fit">
              whats the most common error in aircraft maintenance
            </div>
          </div>
        </div>
      </div>
      <div className="sticky bottom-0 mt-4 bg-card/90 backdrop-blur supports-[backdrop-filter]:bg-card/60 border-t border-border p-2">
        <div className="flex flex-row items-start gap-2">
          <Textarea
            className="w-full h-18 text-lg"
            placeholder="Try another input"
            aria-label="Prompt input"
          />
          <Button className="self-start" aria-label="Submit prompt">
            Submit
          </Button>
        </div>
        <label className="block mt-2 text-sm text-muted-foreground">
          Not sure where to start? Think about the questions your users will
          actually ask to test model reliability.
        </label>
      </div>
    </div>
  )
}

export default Evaluate
