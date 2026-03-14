interface StoppedOverlayProps {
  droneName: string
  onReturnToLaunch: () => void
  onDismiss: () => void
}

export function StoppedOverlay({ droneName, onReturnToLaunch, onDismiss }: StoppedOverlayProps) {
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-[2500]" onClick={onDismiss}>
      <div className="bg-surface-raised rounded-2xl p-6 w-96 max-w-[90vw] border border-surface-border shadow-2xl" onClick={e => e.stopPropagation()}>
        {/* Status */}
        <div className="text-center mb-6">
          <div className="w-16 h-16 mx-auto bg-kill/20 rounded-full flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-kill-hover" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-text-primary mb-2">{droneName} Stopped</h2>
          <p className="text-text-dim text-sm">
            Drone is hovering in place
          </p>
        </div>

        {/* Status details */}
        <div className="bg-surface-overlay rounded-lg p-3 mb-4 text-sm">
          <div className="flex justify-between text-text-dim mb-1">
            <span>Status</span>
            <span className="text-status-critical">EMERGENCY STOP</span>
          </div>
          <div className="flex justify-between text-text-dim">
            <span>Action</span>
            <span className="text-status-warning">Hovering</span>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-2">
          <button
            onClick={onReturnToLaunch}
            className="w-full py-3 bg-status-warning hover:bg-status-warning/80 text-text-primary font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
            </svg>
            Return to Launch
          </button>
          <button
            onClick={onDismiss}
            className="w-full py-3 bg-surface-overlay hover:bg-surface-border text-text-dim rounded-lg transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  )
}
