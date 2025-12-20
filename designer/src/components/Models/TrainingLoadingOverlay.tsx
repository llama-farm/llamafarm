type Props = {
  message?: string
}

function TrainingLoadingOverlay({ message = 'Training your model...' }: Props) {
  return (
    <div className="absolute inset-0 bg-background/90 backdrop-blur-sm z-50 flex items-center justify-center rounded-lg">
      <div className="flex flex-col items-center gap-4 p-6">
        {/* Animated spinner */}
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 border-4 border-primary/20 rounded-full"></div>
          <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        </div>

        {/* Message */}
        <div className="text-center space-y-1">
          <h3 className="text-base font-medium text-foreground">{message}</h3>
          <p className="text-xs text-muted-foreground">
            This may take a few moments
          </p>
        </div>

        {/* Progress dots animation */}
        <div className="flex gap-2">
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0ms]"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:150ms]"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:300ms]"></div>
        </div>
      </div>
    </div>
  )
}

export default TrainingLoadingOverlay
