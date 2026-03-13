interface StoppedOverlayProps {
  droneName: string
  onReturnToLaunch: () => void
  onDismiss: () => void
}

export function StoppedOverlay({ droneName, onReturnToLaunch, onDismiss }: StoppedOverlayProps) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[2500]">
      <div className="bg-surface-raised rounded-2xl p-6 w-80 max-w-[90vw] border border-red-500/50 shadow-2xl">
        {/* Status */}
        <div className="text-center mb-6">
          <div className="w-20 h-20 mx-auto bg-red-500/20 rounded-full flex items-center justify-center mb-4 animate-pulse">
            <svg className="w-10 h-10 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-red-500 mb-2">{droneName} Stopped</h2>
          <p className="text-gray-400 text-sm">
            Drone is hovering in place
          </p>
        </div>

        {/* Status details */}
        <div className="bg-surface-overlay rounded-lg p-3 mb-4 text-sm">
          <div className="flex justify-between text-gray-400 mb-1">
            <span>Status</span>
            <span className="text-red-400">EMERGENCY STOP</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>Action</span>
            <span className="text-amber-400">Hovering</span>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-2">
          <button
            onClick={onReturnToLaunch}
            className="w-full py-3 bg-amber-600 hover:bg-amber-500 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
            </svg>
            Return to Launch
          </button>
          <button
            onClick={onDismiss}
            className="w-full py-3 bg-surface-overlay hover:bg-white/10 text-gray-400 rounded-lg transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  )
}
