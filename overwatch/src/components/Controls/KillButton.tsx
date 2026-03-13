interface KillButtonProps {
  onClick: () => void
  disabled?: boolean
  droneName?: string
}

export function KillButton({ onClick, disabled = false, droneName }: KillButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        flex items-center gap-1.5 px-3 py-1.5 rounded-md
        font-bold text-xs tracking-wide
        transition-all shrink-0
        ${disabled
          ? 'bg-surface-overlay/80 text-gray-600 cursor-not-allowed'
          : 'bg-red-600/90 hover:bg-red-500 active:scale-95 text-white'
        }
      `}
      style={{ boxShadow: disabled ? 'none' : '0 0 12px rgba(239, 68, 68, 0.4)' }}
      aria-label="Emergency Stop"
    >
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
      </svg>
      {droneName ? `KILL ${droneName}` : 'KILL'}
    </button>
  )
}
