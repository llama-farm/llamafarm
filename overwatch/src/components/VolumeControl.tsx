interface VolumeControlProps {
  volume: number
  onVolumeChange: (volume: number) => void
  muted: boolean
  onToggleMute: () => void
  compact?: boolean
}

export function VolumeControl({ volume, onVolumeChange, muted, onToggleMute, compact = false }: VolumeControlProps) {
  return (
    <div className={`flex items-center gap-2 ${compact ? '' : ''}`}>
      <button
        onClick={onToggleMute}
        className="w-6 h-6 flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors shrink-0"
        title={muted ? 'Unmute' : 'Mute'}
      >
        {muted || volume === 0 ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
          </svg>
        ) : volume < 0.5 ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
          </svg>
        )}
      </button>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={muted ? 0 : volume}
        onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
        className={`accent-accent h-1 bg-surface-border rounded-full appearance-none cursor-pointer ${compact ? 'w-20' : 'flex-1'}`}
        style={{ colorScheme: 'dark' }}
      />
      <span className="text-[10px] text-text-dim font-mono w-7 text-right shrink-0">
        {muted ? 'OFF' : `${Math.round(volume * 100)}%`}
      </span>
    </div>
  )
}
