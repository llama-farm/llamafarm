interface LaunchConfirmProps {
  droneName: string
  onConfirm: () => void
  onCancel: () => void
}

export function LaunchConfirm({ droneName, onConfirm, onCancel }: LaunchConfirmProps) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[2000]">
      <div className="bg-surface-raised rounded-2xl p-6 w-80 max-w-[90vw] border border-surface-border shadow-2xl">
        {/* Icon */}
        <div className="text-center mb-4">
          <div className="w-16 h-16 mx-auto bg-green-500/20 rounded-full flex items-center justify-center mb-3">
            <svg className="w-8 h-8 text-green-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
            </svg>
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Deploy {droneName}?</h2>
          <p className="text-gray-400 text-sm">
            Launch to search the selected area
          </p>
        </div>

        {/* Area info */}
        <div className="bg-surface-overlay rounded-lg p-3 mb-4 text-sm">
          <div className="flex justify-between text-gray-400 mb-1">
            <span>Mission</span>
            <span className="text-white">Search Pattern</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>Est. Coverage</span>
            <span className="text-white">~15 min</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-3 bg-surface-overlay hover:bg-white/10 text-white rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-3 bg-green-600 hover:bg-green-500 text-white font-medium rounded-lg transition-colors"
          >
            Confirm Launch
          </button>
        </div>
      </div>
    </div>
  )
}
