interface DrawButtonProps {
  isDrawing: boolean
  disabled: boolean
  onClick: () => void
}

export function DrawButton({ isDrawing, disabled, onClick }: DrawButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        absolute z-[1500]
        px-3 py-2 rounded-lg
        flex items-center gap-2
        font-medium text-sm tracking-wide
        transition-all
        ${disabled
          ? 'bg-surface-overlay/80 text-text-dim cursor-not-allowed'
          : isDrawing
            ? 'bg-accent-muted/80 text-text-primary border border-accent/30'
            : 'bg-surface-overlay/90 border border-surface-border text-text-secondary active:bg-surface-raised'
        }
      `}
      style={{ top: '12px', left: '60px', minHeight: '40px', height: '40px' }}
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
      </svg>
      {isDrawing ? 'Drawing...' : 'Draw Area'}
    </button>
  )
}
