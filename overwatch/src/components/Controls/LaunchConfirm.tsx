interface LaunchConfirmProps {
  droneName: string
  onConfirm: () => void
  onCancel: () => void
}

export function LaunchConfirm({ droneName, onConfirm, onCancel }: LaunchConfirmProps) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[2000]" onClick={onCancel}>
      <div className="bg-surface-raised rounded-2xl p-6 w-96 max-w-[90vw] border border-surface-border shadow-2xl" onClick={e => e.stopPropagation()}>
        {/* Icon */}
        <div className="text-center mb-4">
          <div className="w-16 h-16 mx-auto bg-accent-muted/20 rounded-full flex items-center justify-center mb-3">
            <svg className="w-8 h-8 text-accent-muted" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
            </svg>
          </div>
          <h2 className="text-xl font-bold text-text-primary mb-2">Deploy {droneName}?</h2>
          <p className="text-text-dim text-sm">
            Launch to search the selected area
          </p>
        </div>

        {/* Area info */}
        <div className="bg-surface-overlay rounded-lg p-3 mb-4 text-sm">
          <div className="flex justify-between text-text-dim mb-1">
            <span>Mission</span>
            <span className="text-text-primary">Search Pattern</span>
          </div>
          <div className="flex justify-between text-text-dim">
            <span>Est. Coverage</span>
            <span className="text-text-primary">~15 min</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-3 bg-surface-overlay hover:bg-surface-border text-text-secondary rounded-lg transition-colors whitespace-nowrap"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-3 bg-status-good hover:bg-status-good/80 text-text-primary font-medium rounded-lg transition-colors whitespace-nowrap"
          >
            Confirm Launch
          </button>
        </div>
      </div>
    </div>
  )
}
