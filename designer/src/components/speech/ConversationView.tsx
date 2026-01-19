import { useRef, useEffect } from 'react'
import { Volume2, User } from 'lucide-react'
import { Button } from '../ui/button'
import type { SpeechMessage } from '../../types/ml'

interface ConversationViewProps {
  messages: SpeechMessage[]
  onPlayAudio?: (messageId: string) => void
  playingMessageId?: string | null
  className?: string
}

export function ConversationView({
  messages,
  onPlayAudio,
  playingMessageId,
  className = '',
}: ConversationViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center px-6 py-10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-indigo-500/20 border border-indigo-500/30">
            <Volume2 className="w-5 h-5 text-indigo-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            Start a voice conversation
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Speak or type to begin testing speech models
          </div>
          <div className="mt-3 text-xs text-muted-foreground">
            Tip: Click the microphone to record
          </div>
        </div>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className={`flex-1 overflow-y-auto p-4 space-y-4 ${className}`}>
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          isPlaying={playingMessageId === message.id}
          onPlayAudio={() => onPlayAudio?.(message.id)}
        />
      ))}
    </div>
  )
}

interface MessageBubbleProps {
  message: SpeechMessage
  isPlaying: boolean
  onPlayAudio: () => void
}

function MessageBubble({ message, isPlaying, onPlayAudio }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-primary/20 border border-primary/30'
            : 'bg-indigo-500/20 border border-indigo-500/30'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary" />
        ) : (
          <Volume2 className="w-4 h-4 text-indigo-400" />
        )}
      </div>

      {/* Message content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block rounded-2xl px-4 py-2 ${
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-sm'
              : 'bg-muted rounded-tl-sm'
          }`}
        >
          <p className="text-sm">{message.text}</p>
        </div>

        {/* Footer with time and audio button */}
        <div
          className={`flex items-center gap-2 mt-1 ${
            isUser ? 'justify-end' : 'justify-start'
          }`}
        >
          <span className="text-xs text-muted-foreground">
            {formatTime(message.timestamp)}
          </span>

          {/* Audio playback button for assistant messages */}
          {!isUser && message.audioUrl && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs"
              onClick={onPlayAudio}
            >
              <Volume2 className={`w-3 h-3 mr-1 ${isPlaying ? 'text-primary animate-pulse' : ''}`} />
              {isPlaying ? 'Playing...' : 'Play'}
            </Button>
          )}

          {/* Transcription confidence for user messages */}
          {isUser && message.transcription?.confidence !== undefined && (
            <span className="text-xs text-muted-foreground">
              {(message.transcription.confidence * 100).toFixed(0)}% confidence
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
