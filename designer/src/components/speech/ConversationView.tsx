import { useRef, useEffect } from 'react'
import { Volume2, StopCircle } from 'lucide-react'
import type { SpeechMessage } from '../../types/ml'

interface ConversationViewProps {
  messages: SpeechMessage[]
  onPlayAudio?: (messageId: string) => void
  playingMessageId?: string | null
  /** Currently streaming user transcription text */
  streamingUserText?: string
  /** Currently streaming assistant response text */
  streamingAssistantText?: string
  /** Whether the assistant is currently speaking (for stop button) */
  isSpeaking?: boolean
  /** Callback to stop speaking */
  onStopSpeaking?: () => void
  className?: string
}

export function ConversationView({
  messages,
  onPlayAudio,
  playingMessageId,
  streamingUserText,
  streamingAssistantText,
  isSpeaking,
  onStopSpeaking,
  className = '',
}: ConversationViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive or streaming content changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, streamingUserText, streamingAssistantText])

  if (messages.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center px-6 py-10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
            <Volume2 className="w-5 h-5 text-primary" />
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
    <div ref={scrollRef} className={`flex-1 overflow-y-auto p-4 space-y-5 ${className}`}>
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          isPlaying={playingMessageId === message.id}
          onPlayAudio={() => onPlayAudio?.(message.id)}
        />
      ))}

      {/* Streaming user transcription */}
      {streamingUserText && (
        <div className="w-full flex justify-end">
          <div className="flex flex-col items-end max-w-[80%] md:max-w-[70%]">
            <div className="bg-secondary text-foreground rounded-lg px-4 py-3">
              <p className="text-base leading-relaxed italic text-foreground/70">
                {streamingUserText}
                <span className="animate-pulse ml-0.5">▊</span>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Streaming assistant response OR stop button while audio plays */}
      {(streamingAssistantText || isSpeaking) && (
        <div className="w-full flex justify-start">
          <div className="flex flex-col items-start max-w-[80%] md:max-w-[70%]">
            <div className="flex items-start gap-2">
              {/* Stop button when speaking */}
              {isSpeaking && onStopSpeaking && (
                <button
                  onClick={onStopSpeaking}
                  className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors bg-red-500/20 text-red-600 hover:bg-red-500/30"
                  aria-label="Stop speaking"
                >
                  <StopCircle className="w-4 h-4" />
                </button>
              )}
              {streamingAssistantText && (
                <div className="text-[15px] md:text-base leading-relaxed text-foreground/90">
                  <p className="text-base leading-relaxed">
                    {streamingAssistantText}
                    <span className="animate-pulse ml-0.5">▊</span>
                  </p>
                </div>
              )}
              {/* Show "Playing audio..." when audio is playing but text is done */}
              {!streamingAssistantText && isSpeaking && (
                <div className="text-sm text-muted-foreground italic">
                  Playing audio...
                </div>
              )}
            </div>
          </div>
        </div>
      )}
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
    <div className={`w-full flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[80%] md:max-w-[70%]`}>
        {/* Message row with optional play button */}
        <div className={`flex items-start gap-2 ${isUser ? 'flex-row-reverse' : ''}`}>
          {/* Audio play button for assistant messages */}
          {!isUser && message.audioUrl && (
            <button
              onClick={onPlayAudio}
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
                isPlaying
                  ? 'bg-primary/20 text-primary'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground'
              }`}
              aria-label={isPlaying ? 'Playing audio' : 'Play audio'}
            >
              <Volume2 className={`w-4 h-4 ${isPlaying ? 'animate-pulse' : ''}`} />
            </button>
          )}

          {/* Message bubble */}
          <div
            className={
              isUser
                ? 'bg-secondary text-foreground rounded-lg px-4 py-3'
                : 'text-[15px] md:text-base leading-relaxed text-foreground/90'
            }
          >
            <p className="text-base leading-relaxed">
              {message.text}
            </p>
          </div>
        </div>

        {/* Footer with time */}
        <div
          className={`flex items-center gap-2 mt-1.5 ${
            isUser ? 'pr-1' : !isUser && message.audioUrl ? 'ml-10' : ''
          }`}
        >
          <span className="text-xs text-muted-foreground">
            {formatTime(message.timestamp)}
          </span>

          {/* Transcription confidence for user messages */}
          {isUser && message.transcription?.confidence !== undefined && (
            <span className="text-xs text-muted-foreground">
              {(message.transcription.confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
