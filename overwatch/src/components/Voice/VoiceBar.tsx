import { useState, useCallback } from 'react'

interface VoiceBarProps {
  isListening: boolean
  lastMessage: string
  onToggleMic: () => void
  onTextCommand?: (text: string) => void
}

export function VoiceBar({ isListening, lastMessage, onToggleMic, onTextCommand }: VoiceBarProps) {
  const [inputText, setInputText] = useState('')

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = inputText.trim()
    if (trimmed && onTextCommand) {
      onTextCommand(trimmed)
      setInputText('')
    }
  }, [inputText, onTextCommand])

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 px-3 py-2 bg-surface-bar border-t border-surface-border relative z-[1100]">
      <input
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder={lastMessage || 'Type or tap mic...'}
        className="flex-1 min-h-[36px] bg-surface-raised rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder:text-text-dim border border-surface-border focus:border-accent/40 focus:outline-none transition-colors"
      />
      <button
        type="button"
        onClick={onToggleMic}
        className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
          isListening ? 'bg-accent/20 ring-2 ring-accent' : 'bg-surface-raised hover:bg-surface-overlay'
        }`}
        title={isListening ? 'Stop listening' : 'Start listening'}
      >
        {isListening ? (
          <svg viewBox="0 0 24 24" className="w-4 h-4 text-accent" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-text-secondary" fill="currentColor">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"/>
          </svg>
        )}
      </button>
    </form>
  )
}
