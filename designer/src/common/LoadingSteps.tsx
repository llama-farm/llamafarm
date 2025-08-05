import FontIcon from './FontIcon'

export default function ScrollingLoadingLog() {
  const steps = [
    'Chunking and parsing PDFS found',
    'What else is happening',
    'Analyzing and tagging',
  ]

  const loopedSteps = [...steps, '', ...steps]

  return (
    <div className="relative w-full max-w-md h-16 overflow-hidden rounded-lg px-4 py-2 text-sm text-blue-100">
      <div className="animate-[scroll-up_10s_linear_infinite] flex flex-col gap-3">
        {loopedSteps.map((step, i) => (
          <div key={i} className="flex items-center gap-2 h-5">
            {step ? (
              <>
                <FontIcon type="checkmark-filled" className="w-4 h-4" />
                <span>{step}</span>
              </>
            ) : (
              <div className="h-5" /> // Spacer
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
