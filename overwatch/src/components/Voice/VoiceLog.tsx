import { useEffect, useRef } from 'react'
import type { VoiceEntry } from '../../types'
import { getDetectionColor } from '../../types'

interface VoiceLogProps {
  entries: VoiceEntry[]
  isListening: boolean
  onToggleMic: () => void
  onBackToMap: () => void
}

export function VoiceLog({ entries, isListening, onToggleMic, onBackToMap }: VoiceLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries.length])

  const formatTime = (date: Date) =>
    date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className="absolute inset-0 z-[2000] flex flex-col bg-surface-base">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-surface-bar border-b border-surface-border">
        <h2 className="text-sm font-medium text-text-primary">Mission Log</h2>
        <button
          onClick={onBackToMap}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-surface-raised text-text-secondary text-xs hover:bg-surface-overlay transition-colors"
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
            <path d="M20.5 3l-.16.03L15 5.1 9 3 3.36 4.9c-.21.07-.36.25-.36.48V20.5c0 .28.22.5.5.5l.16-.03L9 18.9l6 2.1 5.64-1.9c.21-.07.36-.25.36-.48V3.5c0-.28-.22-.5-.5-.5z"/>
          </svg>
          Map
        </button>
      </div>

      {/* Entries */}
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
              className={`
                max-w-[85%] rounded-lg px-3 py-2 text-sm
                ${entry.speaker === 'operator'
                  ? 'bg-accent/15 text-text-primary'
                  : entry.speaker === 'system'
                    ? 'bg-surface-overlay text-text-secondary italic'
                    : 'bg-surface-raised text-text-primary'
                }
              `}
            >
              {/* Detection type indicator */}
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

      {/* Voice bar at bottom */}
      <div className="flex items-center gap-3 px-4 py-2.5 bg-surface-bar border-t border-surface-border">
        <span className="flex-1 text-sm text-text-dim">
          {isListening ? 'Listening...' : 'Tap mic to speak'}
        </span>
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
    </div>
  )
}
