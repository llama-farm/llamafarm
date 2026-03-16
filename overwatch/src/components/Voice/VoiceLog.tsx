import { useEffect, useRef, useState, useCallback } from 'react'
import type { VoiceEntry } from '../../types'
import { getDetectionColor } from '../../types'

interface VoiceLogProps {
  entries: VoiceEntry[]
  isListening: boolean
  feedCollapsed?: boolean
  onToggleMic: () => void
  onTextCommand?: (text: string) => void
  layout?: 'mobile' | 'desktop'
}

export function VoiceLog({ entries, isListening, feedCollapsed = false, onToggleMic, onTextCommand, layout = 'mobile' }: VoiceLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [inputText, setInputText] = useState('')

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries.length])

  const formatTime = (date: Date) =>
    date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = inputText.trim()
    if (trimmed && onTextCommand) {
      onTextCommand(trimmed)
      setInputText('')
    }
  }, [inputText, onTextCommand])

  // Desktop layout — flows naturally in the right panel
  if (layout === 'desktop') {
    return (
      <div className="flex flex-col h-full min-h-0">
        {/* Header */}
        <div className="px-3 py-2 border-b border-surface-border flex items-center justify-between shrink-0">
          <span className="text-xs font-medium text-text-secondary tracking-wide">Ace / Chat</span>
          <div className="flex items-center gap-1.5">
            {isListening && (
              <span className="text-[10px] text-accent animate-pulse">● LISTENING</span>
            )}
            <span className="text-[10px] text-text-dim">{entries.length} msgs</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2 min-h-0">
          {entries.length === 0 && (
            <p className="text-text-dim text-sm text-center mt-8">No voice activity yet. Type a command or tap the mic.</p>
          )}
          {entries.map(entry => (
            <div
              key={entry.id}
              className={`flex flex-col animate-fade-in ${entry.speaker === 'operator' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                  entry.speaker === 'operator'
                    ? 'bg-accent/15 text-text-primary'
                    : entry.speaker === 'system'
                      ? 'bg-surface-overlay text-text-secondary italic'
                      : 'bg-surface-raised text-text-primary border border-surface-border'
                }`}
              >
                {entry.type && (
                  <span
                    className="inline-block w-2 h-2 rounded-full mr-2"
                    style={{ backgroundColor: getDetectionColor(entry.type) }}
                  />
                )}
                {entry.text}
              </div>
              <span className="text-[10px] text-text-dim mt-0.5 px-1">
                {formatTime(entry.timestamp)}
              </span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 px-3 py-2 border-t border-surface-border shrink-0">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={isListening ? 'Listening...' : 'Type command...'}
            className="flex-1 min-h-[36px] bg-surface-raised rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder:text-text-dim border border-surface-border focus:border-accent/40 focus:outline-none transition-colors"
          />
          <button
            type="button"
            onClick={onToggleMic}
            className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
              isListening ? 'bg-accent/20 ring-2 ring-accent' : 'bg-surface-raised hover:bg-surface-overlay'
            }`}
          >
            {isListening ? (
              <svg viewBox="0 0 24 24" className="w-4 h-4 text-accent" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4 text-text-secondary" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"/>
              </svg>
            )}
          </button>
        </form>
      </div>
    )
  }

  // Mobile layout (original)
  const paddingClass = feedCollapsed ? 'pr-9' : 'pr-56'

  return (
    <div className={`absolute inset-0 z-[900] flex flex-col bg-surface-base transition-all duration-300 ${paddingClass}`}>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
        {entries.length === 0 && (
          <p className="text-text-dim text-sm text-center mt-8">No voice activity yet. Tap the mic to speak.</p>
        )}
        {entries.map(entry => (
          <div
            key={entry.id}
            className={`flex flex-col ${entry.speaker === 'operator' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                entry.speaker === 'operator'
                  ? 'bg-accent/15 text-text-primary'
                  : entry.speaker === 'system'
                    ? 'bg-surface-overlay text-text-secondary italic'
                    : 'bg-surface-raised text-text-primary'
              }`}
            >
              {entry.type && (
                <span
                  className="inline-block w-2 h-2 rounded-full mr-2"
                  style={{ backgroundColor: getDetectionColor(entry.type) }}
                />
              )}
              {entry.text}
            </div>
            <span className="text-[10px] text-text-dim mt-0.5 px-1">
              {formatTime(entry.timestamp)}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex items-center gap-2 px-3 py-2 bg-surface-bar border-t border-surface-border relative z-[1100]">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder={isListening ? 'Listening...' : 'Type or tap mic...'}
          className="flex-1 min-h-[36px] bg-surface-raised rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder:text-text-dim border border-surface-border focus:border-accent/40 focus:outline-none transition-colors"
        />
        <button
          type="button"
          onClick={onToggleMic}
          className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
            isListening ? 'bg-accent/20 ring-2 ring-accent' : 'bg-surface-raised hover:bg-surface-overlay'
          }`}
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
    </div>
  )
}
