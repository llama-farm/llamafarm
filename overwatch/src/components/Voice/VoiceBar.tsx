interface VoiceBarProps {
  isListening: boolean
  lastMessage: string
  onToggleMic: () => void
  onOpenLog: () => void
}

export function VoiceBar({ isListening, lastMessage, onToggleMic, onOpenLog }: VoiceBarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5 bg-surface-bar border-t border-surface-border">
      {/* Tap text area to open voice log */}
      <button
        onClick={onOpenLog}
        className="flex-1 text-left truncate min-h-[36px] flex items-center"
      >
        <span className="text-sm text-text-secondary truncate">
          {lastMessage || 'Tap mic to speak a command...'}
        </span>
      </button>

      {/* Mic button */}
      <button
        onClick={onToggleMic}
        className={`
          w-11 h-11 rounded-full flex items-center justify-center flex-shrink-0 transition-colors
          ${isListening
            ? 'bg-accent/20 ring-2 ring-accent'
            : 'bg-surface-raised hover:bg-surface-overlay'
          }
        `}
      >
        <svg viewBox="0 0 24 24" className={`w-5 h-5 ${isListening ? 'text-accent' : 'text-text-secondary'}`} fill="currentColor">
          <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"/>
        </svg>
      </button>
    </div>
  )
}
