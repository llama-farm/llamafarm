import { useState, useCallback, useRef, useEffect } from 'react'
import { Mic, Send, MicOff, StopCircle, Volume2, Wifi, WifiOff, AlertCircle, ChevronDown, ChevronRight, Settings2 } from 'lucide-react'
import { Button } from '../ui/button'
import { SpeechToTextConfig } from './SpeechToTextConfig'
import { TextToSpeechConfig } from './TextToSpeechConfig'
import { VoiceCloning } from './VoiceCloning'
import { ConversationView } from './ConversationView'
import { TranscriptionOutput } from './TranscriptionOutput'
import { AudioPlayer } from './AudioPlayer'
import { MicPermissionPrompt } from './MicPermissionPrompt'
import { Waveform } from './Waveform'
import {
  STT_MODELS,
  TTS_MODELS,
  getVoicesForModel,
  type VoiceClone,
  type SpeechMessage,
  type TranscriptionResult,
  type MicPermissionState,
  type RecordingState,
} from '../../types/ml'
import {
  transcribeAudio,
  synthesizeSpeech,
  listVoices,
  type VoiceInfo,
} from '../../api/voiceService'

// Universal Runtime URL for health checks
const UNIVERSAL_RUNTIME_URL =
  import.meta.env.VITE_UNIVERSAL_RUNTIME_URL || 'http://localhost:11540'

interface SpeechTestPanelProps {
  className?: string
}

export function SpeechTestPanel({ className = '' }: SpeechTestPanelProps) {
  // STT Config State
  const [sttEnabled, setSttEnabled] = useState(true)
  const [sttModel, setSttModel] = useState('large-v3-turbo')
  const [sttLanguage, setSttLanguage] = useState('en')
  const [wordTimestamps, setWordTimestamps] = useState(false)

  // TTS Config State
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [ttsModel, setTtsModel] = useState('kokoro')
  const [ttsVoice, setTtsVoice] = useState('af_heart')
  const [ttsSpeed, setTtsSpeed] = useState(1.0)

  // Available voices from backend (fetched but used for validation)
  const [, setAvailableVoices] = useState<VoiceInfo[]>([])
  const [, setVoicesLoading] = useState(false)

  // Voice Cloning State
  const [customVoices, setCustomVoices] = useState<VoiceClone[]>([])
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
  const [ttsError, setTtsError] = useState<string | null>(null)

  // Input State
  const [textInput, setTextInput] = useState('')
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [micPermission, setMicPermission] = useState<MicPermissionState>('prompt')
  const [micError, setMicError] = useState<string | undefined>()
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null)

  // Backend connectivity and model loading
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null)
  const [sttModelLoading, setSttModelLoading] = useState(false)
  const [, setSttModelReady] = useState(false)
  const [, setSttModelError] = useState<string | null>(null)

  // UI State
  const [configExpanded, setConfigExpanded] = useState(true)

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  // Determine which mode we're in
  const mode = sttEnabled && ttsEnabled ? 'conversation' : sttEnabled ? 'stt' : 'tts'

  // Fetch available voices from backend
  useEffect(() => {
    const fetchVoices = async () => {
      setVoicesLoading(true)
      try {
        const voices = await listVoices(ttsModel)
        setAvailableVoices(voices)
        setBackendConnected(true)

        // If current voice isn't in the list, switch to first available
        if (voices.length > 0 && !voices.find(v => v.id === ttsVoice)) {
          setTtsVoice(voices[0].id)
        }
      } catch (err) {
        console.warn('Failed to fetch voices from backend:', err)
        setBackendConnected(false)
        // Fall back to static voices
        setAvailableVoices([])
      } finally {
        setVoicesLoading(false)
      }
    }

    fetchVoices()
  }, [ttsModel])

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

  // Pre-warm STT model when enabled and model changes
  useEffect(() => {
    if (!sttEnabled || !backendConnected) return

    const warmupModel = async () => {
      setSttModelLoading(true)
      setSttModelReady(false)
      setSttModelError(null)

      try {
        // Send a tiny audio blob to trigger model loading
        // The backend will load the model on first request
        const response = await fetch(`${UNIVERSAL_RUNTIME_URL}/health`)
        if (response.ok) {
          // Model endpoint is reachable, but model may still need to load
          // We'll mark as ready and handle loading state during actual transcription
          setSttModelReady(true)
        }
      } catch (err) {
        setSttModelError('Backend not reachable')
      } finally {
        setSttModelLoading(false)
      }
    }

    warmupModel()
  }, [sttEnabled, sttModel, backendConnected])

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
    // Request permission if not granted
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
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream
      setActiveStream(stream)

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })
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
        setActiveStream(null)

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
  }, [micPermission, mode])

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setRecordingState('processing')
    }
  }, [])

  // Process transcription using real backend
  const processTranscription = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true)
    setTranscriptionError(null)

    try {
      const result = await transcribeAudio(audioBlob, {
        model: sttModel,
        language: sttLanguage === 'auto' ? undefined : sttLanguage,
        responseFormat: 'verbose_json',
      })

      // Convert backend response to our TranscriptionResult format
      const transcriptionResult: TranscriptionResult = {
        text: result.text,
        language: result.language,
        confidence: result.language_probability,
        duration: result.duration,
        segments: wordTimestamps && result.segments
          ? result.segments.map(seg => ({
              id: seg.id,
              start: seg.start,
              end: seg.end,
              text: seg.text,
              confidence: seg.avg_logprob ? Math.exp(seg.avg_logprob) : undefined,
            }))
          : undefined,
      }

      setTranscriptionResult(transcriptionResult)
      setBackendConnected(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Transcription failed'
      setTranscriptionError(message)
      setBackendConnected(false)
    } finally {
      setIsTranscribing(false)
      setRecordingState('idle')
    }
  }, [sttModel, sttLanguage, wordTimestamps])

  // Process conversation input (voice) using real backend
  const processConversationInput = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true)
    setTranscriptionError(null)

    try {
      // Transcribe the audio
      const result = await transcribeAudio(audioBlob, {
        model: sttModel,
        language: sttLanguage === 'auto' ? undefined : sttLanguage,
      })

      const transcription: TranscriptionResult = {
        text: result.text,
        language: result.language,
        confidence: result.language_probability,
      }

      // Check if we got any text
      if (!transcription.text || !transcription.text.trim()) {
        setTranscriptionError('No speech detected. Try speaking louder or closer to the microphone.')
        setRecordingState('idle')
        setIsTranscribing(false)
        return
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

      // For now, we'll use a simple echo response since we don't have
      // the full voice chat WebSocket connected yet.
      // In production, this would use useVoiceChat hook with LLM + TTS
      if (ttsEnabled) {
        // Generate TTS response (simple echo for demo)
        const responseText = `I heard you say: "${transcription.text}"`

        try {
          const audioBlob = await synthesizeSpeech({
            model: ttsModel,
            input: responseText,
            voice: ttsVoice,
            speed: ttsSpeed,
            response_format: 'mp3',
          })

          const audioUrl = URL.createObjectURL(audioBlob)

          const assistantMessage: SpeechMessage = {
            id: `msg-${Date.now() + 1}`,
            role: 'assistant',
            text: responseText,
            timestamp: new Date(),
            audioUrl,
          }
          setMessages(prev => [...prev, assistantMessage])
        } catch (ttsErr) {
          // TTS failed, just add text response
          const assistantMessage: SpeechMessage = {
            id: `msg-${Date.now() + 1}`,
            role: 'assistant',
            text: responseText,
            timestamp: new Date(),
          }
          setMessages(prev => [...prev, assistantMessage])
        }
      }

      setBackendConnected(true)
    } catch (err) {
      console.error('Conversation processing failed:', err)
      const message = err instanceof Error ? err.message : 'Transcription failed'
      setTranscriptionError(message)
      setBackendConnected(false)
    } finally {
      setRecordingState('idle')
      setIsTranscribing(false)
    }
  }, [sttModel, sttLanguage, ttsEnabled, ttsModel, ttsVoice, ttsSpeed])

  // Send text message / synthesize TTS
  const sendTextMessage = useCallback(async () => {
    if (!textInput.trim()) return

    if (mode === 'tts') {
      // TTS-only mode: synthesize the text
      setIsSynthesizing(true)
      setTtsInputText(textInput)
      setTtsError(null)

      try {
        const audioBlob = await synthesizeSpeech({
          model: ttsModel,
          input: textInput,
          voice: ttsVoice,
          speed: ttsSpeed,
          response_format: 'mp3',
        })

        setTtsOutputBlob(audioBlob)
        setBackendConnected(true)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Speech synthesis failed'
        setTtsError(message)
        setBackendConnected(false)
      } finally {
        setIsSynthesizing(false)
        setTextInput('')
      }
    } else {
      // Conversation mode - text input
      const userMessage: SpeechMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        text: textInput,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, userMessage])
      setTextInput('')

      // Generate TTS response if enabled
      if (ttsEnabled) {
        const responseText = `I heard you say: "${textInput}"`

        try {
          const audioBlob = await synthesizeSpeech({
            model: ttsModel,
            input: responseText,
            voice: ttsVoice,
            speed: ttsSpeed,
            response_format: 'mp3',
          })

          const audioUrl = URL.createObjectURL(audioBlob)

          const assistantMessage: SpeechMessage = {
            id: `msg-${Date.now() + 1}`,
            role: 'assistant',
            text: responseText,
            timestamp: new Date(),
            audioUrl,
          }
          setMessages(prev => [...prev, assistantMessage])
          setBackendConnected(true)
        } catch (err) {
          // TTS failed, add text-only response
          const assistantMessage: SpeechMessage = {
            id: `msg-${Date.now() + 1}`,
            role: 'assistant',
            text: responseText,
            timestamp: new Date(),
          }
          setMessages(prev => [...prev, assistantMessage])
        }
      }
    }
  }, [textInput, mode, ttsModel, ttsVoice, ttsSpeed, ttsEnabled])

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
    if (ttsVoice === voiceId) {
      const defaultVoices = getVoicesForModel(ttsModel)
      setTtsVoice(defaultVoices[0]?.id || 'af_heart')
    }
  }, [ttsVoice, ttsModel])

  const handlePreviewVoice = useCallback(async (voiceId: string) => {
    if (previewingVoiceId === voiceId) {
      setPreviewingVoiceId(null)
      return
    }

    setPreviewingVoiceId(voiceId)

    try {
      // Synthesize a preview phrase
      const previewText = 'Hello! This is a preview of my voice.'
      await synthesizeSpeech({
        model: ttsModel,
        input: previewText,
        voice: voiceId,
        speed: ttsSpeed,
        response_format: 'mp3',
      })
      // Note: In a full implementation, we'd play this audio
    } catch (err) {
      console.error('Voice preview failed:', err)
    } finally {
      setTimeout(() => setPreviewingVoiceId(null), 2000)
    }
  }, [previewingVoiceId, ttsModel, ttsSpeed])

  // Play message audio
  const handlePlayMessageAudio = useCallback((messageId: string) => {
    if (playingMessageId === messageId) {
      setPlayingMessageId(null)
    } else {
      setPlayingMessageId(messageId)
    }
  }, [playingMessageId])

  // Show mic permission prompt if needed and trying to record
  const needsMicPermission = micPermission !== 'granted' && sttEnabled

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Backend status indicator */}
      {backendConnected !== null && (
        <div className={`flex-shrink-0 px-4 py-1.5 text-xs flex items-center gap-1.5 ${
          backendConnected ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-600'
        }`}>
          {backendConnected ? (
            <>
              <Wifi className="h-3 w-3" />
              <span>Connected to Universal Runtime</span>
              {sttEnabled && sttModelLoading && (
                <span className="ml-2 text-yellow-600">• Loading STT model...</span>
              )}
              {sttEnabled && isTranscribing && (
                <span className="ml-2 text-blue-600">• Transcribing (model may be loading on first use)...</span>
              )}
            </>
          ) : (
            <>
              <WifiOff className="h-3 w-3" />
              <span>Backend unavailable - check that Universal Runtime is running</span>
            </>
          )}
        </div>
      )}

      {/* Configuration Section - Collapsible */}
      <div className="flex-shrink-0 border-b border-border">
        {/* Header - always visible */}
        <button
          onClick={() => setConfigExpanded(!configExpanded)}
          className="w-full px-4 py-2 flex items-center justify-between hover:bg-muted/50 transition-colors"
        >
          <div className="flex items-center gap-2 text-sm font-medium">
            <Settings2 className="h-4 w-4 text-muted-foreground" />
            <span>Model Configuration</span>
            {/* Compact summary when collapsed */}
            {!configExpanded && (
              <span className="text-xs text-muted-foreground font-normal ml-2">
                {sttEnabled && `STT: ${STT_MODELS.find(m => m.id === sttModel)?.name || sttModel}`}
                {sttEnabled && ttsEnabled && ' • '}
                {ttsEnabled && `TTS: ${TTS_MODELS.find(m => m.id === ttsModel)?.name || ttsModel}`}
              </span>
            )}
          </div>
          {configExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </button>

        {/* Expandable content */}
        {configExpanded && (
          <div className="px-4 pb-3 space-y-3">
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
                onModelChange={(model) => {
                  setTtsModel(model)
                  // Reset voice to first available for new model
                  const modelVoices = getVoicesForModel(model)
                  setTtsVoice(modelVoices[0]?.id || 'af_heart')
                }}
                selectedVoice={ttsVoice}
                onVoiceChange={setTtsVoice}
                speed={ttsSpeed}
                onSpeedChange={setTtsSpeed}
                models={TTS_MODELS}
                customVoices={customVoices}
              />
            </div>

            {/* Only show voice cloning if the selected TTS model supports it */}
            {TTS_MODELS.find(m => m.id === ttsModel)?.supportsVoiceCloning && (
              <VoiceCloning
                voices={customVoices}
                onAddVoice={handleAddVoice}
                onDeleteVoice={handleDeleteVoice}
                onPreviewVoice={handlePreviewVoice}
                previewingVoiceId={previewingVoiceId}
              />
            )}
          </div>
        )}
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

        {/* Main content area based on mode */}
        {!needsMicPermission && (
        <div className="flex-1 min-h-0 overflow-hidden">
          {mode === 'conversation' && (
            <div className="h-full flex flex-col">
              {/* Error display for conversation mode */}
              {transcriptionError && (
                <div className="flex-shrink-0 mx-4 mt-4 flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-600 text-sm">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <span>{transcriptionError}</span>
                  <button
                    onClick={() => setTranscriptionError(null)}
                    className="ml-auto text-xs hover:text-red-800"
                  >
                    Dismiss
                  </button>
                </div>
              )}
              <ConversationView
                messages={messages}
                onPlayAudio={handlePlayMessageAudio}
                playingMessageId={playingMessageId}
                className="flex-1"
              />
            </div>
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

                {/* Error message */}
                {ttsError && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-600 text-sm">
                    <AlertCircle className="h-4 w-4 flex-shrink-0" />
                    <span>{ttsError}</span>
                  </div>
                )}

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

        {/* Input Area (for conversation and STT modes) */}
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

              {/* STT-only prompt or waveform */}
              {mode === 'stt' && (
                <div className="flex-1 py-2">
                  {recordingState === 'recording' && activeStream ? (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2 text-red-500">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-sm font-medium">Recording</span>
                      </div>
                      <Waveform
                        stream={activeStream}
                        isActive={true}
                        height={32}
                        barCount={24}
                        color="rgb(239, 68, 68)"
                        className="flex-1"
                      />
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      {recordingState === 'idle'
                        ? 'Click the microphone to start recording'
                        : 'Processing audio...'}
                    </div>
                  )}
                </div>
              )}

              {/* Waveform for conversation mode when recording */}
              {mode === 'conversation' && recordingState === 'recording' && activeStream && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <Waveform
                    stream={activeStream}
                    isActive={true}
                    height={32}
                    barCount={16}
                    color="rgb(239, 68, 68)"
                    className="w-24"
                  />
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
