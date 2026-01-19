import { useState, useCallback, useRef, useEffect } from 'react'
import { Mic, Send, MicOff, StopCircle, Volume2 } from 'lucide-react'
import { Button } from '../ui/button'
import { SpeechToTextConfig } from './SpeechToTextConfig'
import { TextToSpeechConfig } from './TextToSpeechConfig'
import { VoiceCloning } from './VoiceCloning'
import { ConversationView } from './ConversationView'
import { TranscriptionOutput } from './TranscriptionOutput'
import { AudioPlayer } from './AudioPlayer'
import { MicPermissionPrompt } from './MicPermissionPrompt'
import {
  STT_MODELS,
  TTS_MODELS,
  type VoiceClone,
  type SpeechMessage,
  type TranscriptionResult,
  type MicPermissionState,
  type RecordingState,
} from '../../types/ml'

interface SpeechTestPanelProps {
  className?: string
}

export function SpeechTestPanel({ className = '' }: SpeechTestPanelProps) {
  // STT Config State
  const [sttEnabled, setSttEnabled] = useState(true)
  const [sttModel, setSttModel] = useState('base')
  const [sttLanguage, setSttLanguage] = useState('auto')
  const [wordTimestamps, setWordTimestamps] = useState(false)

  // TTS Config State
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [ttsModel, setTtsModel] = useState('xtts-v2')
  const [ttsVoice, setTtsVoice] = useState('allison')
  const [ttsSpeed, setTtsSpeed] = useState(1.0)

  // Voice Cloning State
  const [customVoices, setCustomVoices] = useState<VoiceClone[]>([
    { id: 'custom-1', name: 'My Voice', duration: 24, createdAt: '2025-01-15' },
  ])
  const [previewingVoiceId, setPreviewingVoiceId] = useState<string | null>(null)

  // Conversation State
  const [messages, setMessages] = useState<SpeechMessage[]>([])
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null)

  // STT-only State
  const [transcriptionResult, setTranscriptionResult] = useState<TranscriptionResult | null>(null)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null)

  // TTS-only State
  const [ttsInputText, setTtsInputText] = useState('')
  const [ttsOutputBlob, setTtsOutputBlob] = useState<Blob | null>(null)
  const [isSynthesizing, setIsSynthesizing] = useState(false)

  // Input State
  const [textInput, setTextInput] = useState('')
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [micPermission, setMicPermission] = useState<MicPermissionState>('prompt')
  const [micError, setMicError] = useState<string | undefined>()

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  // Determine which mode we're in
  const mode = sttEnabled && ttsEnabled ? 'conversation' : sttEnabled ? 'stt' : 'tts'

  // Check microphone permission on mount
  useEffect(() => {
    const checkMicPermission = async () => {
      try {
        const result = await navigator.permissions.query({ name: 'microphone' as PermissionName })
        setMicPermission(result.state as MicPermissionState)
        result.onchange = () => {
          setMicPermission(result.state as MicPermissionState)
        }
      } catch {
        // Permissions API not supported, will check on first use
      }
    }
    checkMicPermission()
  }, [])

  // Request microphone permission
  const requestMicPermission = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(track => track.stop())
      setMicPermission('granted')
      setMicError(undefined)
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          setMicPermission('denied')
        } else {
          setMicPermission('error')
          setMicError(err.message)
        }
      }
    }
  }, [])

  // Start recording
  const startRecording = useCallback(async () => {
    // Request permission if not granted - note that micPermission state may not update synchronously
    if (micPermission !== 'granted') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        stream.getTracks().forEach(track => track.stop())
        setMicPermission('granted')
      } catch (err) {
        if (err instanceof Error && err.name === 'NotAllowedError') {
          setMicPermission('denied')
        } else {
          setMicPermission('error')
        }
        return
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(track => track.stop())

        // Process the recording based on mode
        if (mode === 'stt') {
          await processTranscription(blob)
        } else {
          await processConversationInput(blob)
        }
      }

      mediaRecorder.start()
      setRecordingState('recording')
    } catch (err) {
      console.error('Failed to start recording:', err)
      setRecordingState('error')
    }
  }, [micPermission, mode, requestMicPermission])

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setRecordingState('processing')
    }
  }, [])

  // Mock transcription processing
  const processTranscription = useCallback(async (_audioBlob: Blob) => {
    setIsTranscribing(true)
    setTranscriptionError(null)

    // Mock API delay
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Mock transcription result
    const mockResult: TranscriptionResult = {
      text: "Hello, this is a test transcription. The quick brown fox jumps over the lazy dog.",
      language: sttLanguage === 'auto' ? 'en' : sttLanguage,
      confidence: 0.92,
      duration: 4.5,
      segments: wordTimestamps ? [
        { id: 1, start: 0, end: 0.5, text: "Hello,", confidence: 0.95 },
        { id: 2, start: 0.5, end: 0.8, text: "this", confidence: 0.94 },
        { id: 3, start: 0.8, end: 1.0, text: "is", confidence: 0.98 },
        { id: 4, start: 1.0, end: 1.2, text: "a", confidence: 0.97 },
        { id: 5, start: 1.2, end: 1.5, text: "test", confidence: 0.93 },
        { id: 6, start: 1.5, end: 2.2, text: "transcription.", confidence: 0.91 },
      ] : undefined,
    }

    setTranscriptionResult(mockResult)
    setIsTranscribing(false)
    setRecordingState('idle')
  }, [sttLanguage, wordTimestamps])

  // Process conversation input (voice)
  const processConversationInput = useCallback(async (_audioBlob: Blob) => {
    // Mock transcription
    await new Promise(resolve => setTimeout(resolve, 800))

    const transcription: TranscriptionResult = {
      text: "Hello, how are you doing today?",
      language: 'en',
      confidence: 0.94,
    }

    // Add user message
    const userMessage: SpeechMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      text: transcription.text,
      timestamp: new Date(),
      transcription,
    }
    setMessages(prev => [...prev, userMessage])

    // Mock TTS response
    await new Promise(resolve => setTimeout(resolve, 1000))

    // Add assistant message
    const assistantMessage: SpeechMessage = {
      id: `msg-${Date.now() + 1}`,
      role: 'assistant',
      text: "I'm doing great, thank you for asking! How can I help you today?",
      timestamp: new Date(),
      audioUrl: '#mock-audio', // Would be a real URL in production
    }
    setMessages(prev => [...prev, assistantMessage])

    setRecordingState('idle')
  }, [])

  // Send text message
  const sendTextMessage = useCallback(async () => {
    if (!textInput.trim()) return

    if (mode === 'tts') {
      // TTS-only mode: synthesize the text
      setIsSynthesizing(true)
      setTtsInputText(textInput)

      // Mock synthesis delay
      await new Promise(resolve => setTimeout(resolve, 1200))

      // Mock audio blob (in production, this would come from the API)
      setTtsOutputBlob(new Blob([], { type: 'audio/mp3' }))
      setIsSynthesizing(false)
      setTextInput('')
    } else {
      // Conversation mode
      const userMessage: SpeechMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        text: textInput,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, userMessage])
      setTextInput('')

      // Mock response
      await new Promise(resolve => setTimeout(resolve, 1000))

      const assistantMessage: SpeechMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        text: "That's an interesting question! Let me think about that...",
        timestamp: new Date(),
        audioUrl: '#mock-audio',
      }
      setMessages(prev => [...prev, assistantMessage])
    }
  }, [textInput, mode])

  // Handle key press in textarea
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendTextMessage()
    }
  }, [sendTextMessage])

  // Voice cloning handlers
  const handleAddVoice = useCallback((voice: VoiceClone) => {
    setCustomVoices(prev => [...prev, voice])
  }, [])

  const handleDeleteVoice = useCallback((voiceId: string) => {
    setCustomVoices(prev => prev.filter(v => v.id !== voiceId))
    // If this voice was selected for TTS, switch to a preset
    if (ttsVoice === voiceId) {
      setTtsVoice('allison')
    }
  }, [ttsVoice])

  const handlePreviewVoice = useCallback((voiceId: string) => {
    if (previewingVoiceId === voiceId) {
      setPreviewingVoiceId(null)
    } else {
      setPreviewingVoiceId(voiceId)
      // Mock: stop preview after 2 seconds
      setTimeout(() => setPreviewingVoiceId(null), 2000)
    }
  }, [previewingVoiceId])

  // Play message audio
  const handlePlayMessageAudio = useCallback((messageId: string) => {
    if (playingMessageId === messageId) {
      setPlayingMessageId(null)
    } else {
      setPlayingMessageId(messageId)
      // Mock: stop playing after 2 seconds
      setTimeout(() => setPlayingMessageId(null), 2000)
    }
  }, [playingMessageId])

  // Show mic permission prompt if needed and trying to record
  const needsMicPermission = micPermission !== 'granted' && sttEnabled

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Configuration Section */}
      <div className="flex-shrink-0 p-4 border-b border-border space-y-3 overflow-y-auto max-h-[40%]">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <SpeechToTextConfig
            enabled={sttEnabled}
            onEnabledChange={setSttEnabled}
            selectedModel={sttModel}
            onModelChange={setSttModel}
            selectedLanguage={sttLanguage}
            onLanguageChange={setSttLanguage}
            wordTimestamps={wordTimestamps}
            onWordTimestampsChange={setWordTimestamps}
            models={STT_MODELS}
          />

          <TextToSpeechConfig
            enabled={ttsEnabled}
            onEnabledChange={setTtsEnabled}
            selectedModel={ttsModel}
            onModelChange={setTtsModel}
            selectedVoice={ttsVoice}
            onVoiceChange={setTtsVoice}
            speed={ttsSpeed}
            onSpeedChange={setTtsSpeed}
            models={TTS_MODELS}
            customVoices={customVoices}
          />
        </div>

        <VoiceCloning
          voices={customVoices}
          onAddVoice={handleAddVoice}
          onDeleteVoice={handleDeleteVoice}
          onPreviewVoice={handlePreviewVoice}
          previewingVoiceId={previewingVoiceId}
        />
      </div>

      {/* Test Area */}
      <div className="flex-1 min-h-0 flex flex-col">
        {/* Mic permission prompt */}
        {needsMicPermission && (
          <div className="p-4">
            <MicPermissionPrompt
              state={micPermission}
              onRequestPermission={requestMicPermission}
              onContinueWithoutVoice={() => setSttEnabled(false)}
              errorMessage={micError}
            />
          </div>
        )}

        {/* Main content area based on mode - hidden when mic permission prompt is showing */}
        {!needsMicPermission && (
        <div className="flex-1 min-h-0 overflow-hidden">
          {mode === 'conversation' && (
            <ConversationView
              messages={messages}
              onPlayAudio={handlePlayMessageAudio}
              playingMessageId={playingMessageId}
              className="h-full"
            />
          )}

          {mode === 'stt' && (
            <div className="h-full p-4 overflow-y-auto">
              <TranscriptionOutput
                result={transcriptionResult}
                isLoading={isTranscribing}
                error={transcriptionError}
                showTimestamps={wordTimestamps}
              />
            </div>
          )}

          {mode === 'tts' && (
            <div className="h-full p-4 overflow-y-auto">
              <div className="max-w-2xl mx-auto space-y-4">
                {/* TTS Input */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Text to Speak</label>
                  <textarea
                    value={ttsInputText || textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Enter text to convert to speech..."
                    rows={4}
                    className="w-full px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                {/* Synthesize button */}
                <Button
                  onClick={sendTextMessage}
                  disabled={!textInput.trim() || isSynthesizing}
                  className="w-full"
                >
                  {isSynthesizing ? (
                    <>
                      <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Synthesizing...
                    </>
                  ) : (
                    <>
                      <Volume2 className="w-4 h-4 mr-2" />
                      Speak
                    </>
                  )}
                </Button>

                {/* Audio output */}
                {ttsOutputBlob && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Output</label>
                    <AudioPlayer blob={ttsOutputBlob} />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        )}

        {/* Input Area (for conversation and STT modes) - hidden when mic permission prompt is showing */}
        {!needsMicPermission && (mode === 'conversation' || mode === 'stt') && (
          <div className="flex-shrink-0 p-3 border-t border-border bg-background/60">
            <div className="flex items-end gap-2">
              {/* Text input (only for conversation mode) */}
              {mode === 'conversation' && (
                <textarea
                  ref={inputRef}
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type or speak..."
                  rows={1}
                  className="flex-1 px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring text-sm min-h-[40px] max-h-[120px]"
                  style={{ height: 'auto' }}
                />
              )}

              {/* STT-only prompt */}
              {mode === 'stt' && (
                <div className="flex-1 text-sm text-muted-foreground py-2">
                  {recordingState === 'idle'
                    ? 'Click the microphone to start recording'
                    : recordingState === 'recording'
                      ? 'Recording... Click to stop'
                      : 'Processing audio...'}
                </div>
              )}

              {/* Mic button */}
              {sttEnabled && micPermission === 'granted' && (
                <Button
                  variant={recordingState === 'recording' ? 'destructive' : 'outline'}
                  size="icon"
                  className="h-10 w-10 rounded-full"
                  onClick={recordingState === 'recording' ? stopRecording : startRecording}
                  disabled={recordingState === 'processing'}
                  aria-label={recordingState === 'recording' ? 'Stop recording' : 'Start recording'}
                >
                  {recordingState === 'recording' ? (
                    <StopCircle className="h-5 w-5" />
                  ) : recordingState === 'processing' ? (
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : (
                    <Mic className="h-5 w-5" />
                  )}
                </Button>
              )}

              {/* Mic permission denied indicator */}
              {sttEnabled && micPermission === 'denied' && (
                <Button
                  variant="outline"
                  size="icon"
                  className="h-10 w-10 rounded-full"
                  onClick={requestMicPermission}
                  aria-label="Microphone access denied"
                >
                  <MicOff className="h-5 w-5 text-muted-foreground" />
                </Button>
              )}

              {/* Send button (only for conversation mode) */}
              {mode === 'conversation' && (
                <Button
                  size="icon"
                  className="h-10 w-10 rounded-full"
                  onClick={sendTextMessage}
                  disabled={!textInput.trim()}
                  aria-label="Send message"
                >
                  <Send className="h-5 w-5" />
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
