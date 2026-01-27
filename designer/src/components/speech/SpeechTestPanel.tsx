import { useState, useCallback, useRef, useEffect } from 'react'
import { Mic, Send, MicOff, StopCircle, Volume2, AlertCircle, ChevronDown, ChevronRight, Settings2, MessageSquare, HelpCircle } from 'lucide-react'
import { Button } from '../ui/button'
import { SpeechToTextConfig } from './SpeechToTextConfig'
import { TextToSpeechConfig } from './TextToSpeechConfig'
import { TurnDetectionSettings } from './TurnDetectionSettings'
import { VoiceCloning } from './VoiceCloning'
import { ConversationView } from './ConversationView'
import { TranscriptionOutput } from './TranscriptionOutput'
import { MicPermissionPrompt } from './MicPermissionPrompt'
import { Waveform } from './Waveform'
import { Selector } from '../ui/selector'
import { Switch } from '../ui/switch'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip'
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
import { sendChatCompletion } from '../../api/chatCompletionsService'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProjectModels } from '../../hooks/useProjectModels'
import { useVoiceChat, type VoiceMessage } from '../../hooks/useVoiceChat'

// Universal Runtime URL for health checks
// Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
const UNIVERSAL_RUNTIME_URL =
  import.meta.env.VITE_UNIVERSAL_RUNTIME_URL || 'http://127.0.0.1:11540'

interface SpeechTestPanelProps {
  className?: string
  /** Ref to expose clear function to parent */
  clearRef?: React.MutableRefObject<(() => void) | null>
  /** Callback when messages change (for parent to track if clear should be enabled) */
  onMessagesChange?: (hasMessages: boolean) => void
}

/** History item for TTS-only mode */
interface TTSHistoryItem {
  id: string
  text: string
  audioUrl: string
  timestamp: Date
}

export function SpeechTestPanel({ className = '', clearRef, onMessagesChange }: SpeechTestPanelProps) {
  // STT Config State
  const [sttEnabled, setSttEnabled] = useState(true)
  const [sttModel, setSttModel] = useState('base')
  const [sttLanguage, setSttLanguage] = useState('en')
  const [wordTimestamps, setWordTimestamps] = useState(false)

  // TTS Config State
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [ttsModel, setTtsModel] = useState('pocket-tts')
  const [ttsVoice, setTtsVoice] = useState('alba')
  const [ttsSpeed, setTtsSpeed] = useState(1.0)

  // Turn Detection Config State (defaults aligned with backend)
  const [turnDetectionEnabled, setTurnDetectionEnabled] = useState(true)
  const [baseSilenceDuration, setBaseSilenceDuration] = useState(0.1)
  const [thinkingSilenceDuration, setThinkingSilenceDuration] = useState(0.7)
  const [maxSilenceDuration, setMaxSilenceDuration] = useState(1.0)

  // Available voices from backend (fetched but used for validation)
  const [, setAvailableVoices] = useState<VoiceInfo[]>([])
  const [, setVoicesLoading] = useState(false)

  // Voice Cloning State
  const [customVoices, setCustomVoices] = useState<VoiceClone[]>([])
  const [previewingVoiceId, setPreviewingVoiceId] = useState<string | null>(null)

  // Conversation State
  const [messages, setMessages] = useState<SpeechMessage[]>([])
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null)
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null)

  // STT-only State
  const [transcriptionResult, setTranscriptionResult] = useState<TranscriptionResult | null>(null)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null)

  // TTS-only State
  const [isSynthesizing, setIsSynthesizing] = useState(false)
  const [ttsError, setTtsError] = useState<string | null>(null)

  // TTS History State (for TTS-only mode conversation-style UI)
  const [ttsHistory, setTtsHistory] = useState<TTSHistoryItem[]>([])
  const [playingTtsId, setPlayingTtsId] = useState<string | null>(null)
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null)

  // Track typed input audio playback (for stop button in conversation mode)
  const [isTypedAudioPlaying, setIsTypedAudioPlaying] = useState(false)
  const typedAudioRef = useRef<HTMLAudioElement | null>(null)

  // Track waiting states for conversation mode (typed input)
  const [isWaitingForLLM, setIsWaitingForLLM] = useState(false)
  const [isSynthesizingTypedTTS, setIsSynthesizingTypedTTS] = useState(false)

  // Input State
  const [textInput, setTextInput] = useState('')
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [micPermission, setMicPermission] = useState<MicPermissionState>('prompt')
  const [micError, setMicError] = useState<string | undefined>()
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null)

  // Backend connectivity and model loading
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null)
  const [, setSttModelLoading] = useState(false)
  const [, setSttModelReady] = useState(false)
  const [, setSttModelError] = useState<string | null>(null)

  // UI State
  const [configExpanded, setConfigExpanded] = useState(true)
  const [pendingVoiceChatRecord, setPendingVoiceChatRecord] = useState(false) // Waiting for connection to start recording
  // Text messages now use REST API (not WebSocket), no pending state needed

  // LLM Integration State
  const activeProject = useActiveProject()
  const { data: projectModelsData } = useProjectModels(
    activeProject?.namespace,
    activeProject?.project
  )
  const availableLLMModels = projectModelsData?.models || []
  const [selectedLLMModel, setSelectedLLMModel] = useState<string>('')
  const [llmEnabled, setLlmEnabled] = useState(true)

  // Set default LLM model when models are loaded
  useEffect(() => {
    if (availableLLMModels.length > 0 && !selectedLLMModel) {
      // Prefer the default model, otherwise use the first one
      const defaultModel = availableLLMModels.find(m => m.default)
      setSelectedLLMModel(defaultModel?.name || availableLLMModels[0].name)
    }
  }, [availableLLMModels, selectedLLMModel])

  // Determine which mode we're in (calculated early for hook config)
  // If LLM is enabled, always use conversation mode (supports typed input with LLM responses)
  // Otherwise, use STT-only or TTS-only mode based on what's enabled
  const mode = llmEnabled
    ? 'conversation'  // LLM enabled - show conversation UI regardless of STT/TTS
    : sttEnabled
      ? 'stt'         // STT only (no LLM) - show transcription UI
      : 'tts'         // TTS only (no LLM) - show TTS-only UI

  // Default voice system prompt for concise responses
  const voiceSystemPrompt = 'You are a helpful voice assistant. Keep responses brief and conversational - aim for 1-3 sentences unless more detail is explicitly requested. Speak naturally as if having a conversation.'

  // Single voice chat hook - llmEnabled controls frontend display of LLM responses
  const voiceChat = useVoiceChat({
    namespace: activeProject?.namespace || '',
    project: activeProject?.project || '',
    llmModel: selectedLLMModel,
    sttModel,
    ttsModel: mode === 'stt' || !llmEnabled ? undefined : ttsModel,
    ttsVoice: mode === 'stt' || !llmEnabled ? undefined : ttsVoice,
    language: sttLanguage,
    speed: mode === 'stt' || !llmEnabled ? undefined : ttsSpeed,
    systemPrompt: voiceSystemPrompt,
    llmEnabled: mode !== 'stt' && llmEnabled,
    turnDetectionEnabled,
    baseSilenceDuration,
    thinkingSilenceDuration,
    maxSilenceDuration,
    autoConnect: false,
    onError: (error) => setTranscriptionError(error),
  })

  // LLM is available when we have a project with models and LLM is enabled
  const llmAvailable = availableLLMModels.length > 0 && !!activeProject && llmEnabled

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  // Handlers for STT/TTS toggles with mutual exclusivity (at least one must be enabled)
  const handleSttEnabledChange = useCallback((enabled: boolean) => {
    if (!enabled && !ttsEnabled) {
      // If trying to disable STT and TTS is already off, enable TTS first
      setTtsEnabled(true)
    }
    setSttEnabled(enabled)
  }, [ttsEnabled])

  const handleTtsEnabledChange = useCallback((enabled: boolean) => {
    if (!enabled && !sttEnabled) {
      // If trying to disable TTS and STT is already off, enable STT first
      setSttEnabled(true)
    }
    setTtsEnabled(enabled)
  }, [sttEnabled])

  // Helper to convert raw PCM to WAV blob
  const pcmToWavBlob = useCallback((pcmData: ArrayBuffer, sampleRate: number = 24000): Blob => {
    const numChannels = 1
    const bitsPerSample = 16
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8)
    const blockAlign = numChannels * (bitsPerSample / 8)
    const dataSize = pcmData.byteLength
    const headerSize = 44
    const totalSize = headerSize + dataSize

    const buffer = new ArrayBuffer(totalSize)
    const view = new DataView(buffer)

    // RIFF header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, totalSize - 8, true)
    writeString(8, 'WAVE')

    // fmt chunk
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true) // fmt chunk size
    view.setUint16(20, 1, true) // audio format (PCM)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, byteRate, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitsPerSample, true)

    // data chunk
    writeString(36, 'data')
    view.setUint32(40, dataSize, true)

    // Copy PCM data
    const pcmBytes = new Uint8Array(pcmData)
    const wavBytes = new Uint8Array(buffer)
    wavBytes.set(pcmBytes, headerSize)

    return new Blob([buffer], { type: 'audio/wav' })
  }, [])

  // Track blob URLs for cleanup to prevent memory leaks
  const blobUrlsRef = useRef<Map<string, string>>(new Map())

  // Sync voiceChat messages with local messages state
  // Only sync voice messages - typed input messages are added directly to messages state
  useEffect(() => {
    if (mode === 'conversation' && llmAvailable) {
      // Convert VoiceMessage to SpeechMessage format
      const voiceMessagesConverted: SpeechMessage[] = voiceChat.messages.map((vm: VoiceMessage) => {
        // Reuse existing blob URL if we already created one for this message
        const existingUrl = blobUrlsRef.current.get(vm.id)
        if (existingUrl) {
          return {
            id: vm.id,
            role: vm.role,
            text: vm.text,
            timestamp: vm.timestamp,
            audioUrl: existingUrl,
          }
        }

        // Create new blob URL only for new messages with audio
        let audioUrl: string | undefined
        if (vm.audioData) {
          audioUrl = URL.createObjectURL(pcmToWavBlob(vm.audioData, 24000))
          blobUrlsRef.current.set(vm.id, audioUrl)
        }

        return {
          id: vm.id,
          role: vm.role,
          text: vm.text,
          timestamp: vm.timestamp,
          audioUrl,
        }
      })

      // Merge voice messages with existing typed messages (those with 'msg-' prefix)
      // Voice messages have 'user-' prefix from voiceChat
      setMessages(prev => {
        // Keep typed messages (ids starting with 'msg-')
        const typedMessages = prev.filter(m => m.id.startsWith('msg-'))
        // Combine: voice messages + typed messages, sorted by timestamp
        const combined = [...voiceMessagesConverted, ...typedMessages]
        combined.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
        return combined
      })

      // Clean up blob URLs for voice messages that no longer exist
      // Note: Don't clean up typed message URLs (msg- prefix) here - they're managed separately
      const currentIds = new Set(voiceChat.messages.map(m => m.id))
      for (const [id, url] of blobUrlsRef.current.entries()) {
        // Skip typed message URLs - they're not in voiceChat.messages
        if (id.startsWith('msg-')) continue
        if (!currentIds.has(id)) {
          URL.revokeObjectURL(url)
          blobUrlsRef.current.delete(id)
        }
      }
    }
  }, [mode, llmAvailable, voiceChat.messages, pcmToWavBlob])

  // Clear transient errors when we receive a successful assistant response
  // This prevents errors from intermediate failures (like streaming STT) from persisting
  // when the overall conversation succeeds (via fallback paths)
  const lastMessageCountRef = useRef(0)
  useEffect(() => {
    const assistantMessages = voiceChat.messages.filter(m => m.role === 'assistant')
    if (assistantMessages.length > lastMessageCountRef.current) {
      // New assistant message arrived - conversation succeeded, clear transient errors
      setTranscriptionError(null)
    }
    lastMessageCountRef.current = assistantMessages.length
  }, [voiceChat.messages])

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      blobUrlsRef.current.forEach(url => URL.revokeObjectURL(url))
      blobUrlsRef.current.clear()
    }
  }, [])

  // Handle pending voice chat recording - consolidated effect for connection success/error
  // Note: Text messages now use REST API, so no pending message handling needed
  useEffect(() => {
    if (!pendingVoiceChatRecord) return

    // Handle error case first (takes priority over connection success)
    if (voiceChat.error) {
      setPendingVoiceChatRecord(false)
      setRecordingState('idle')
      setTranscriptionError(voiceChat.error)
      return
    }

    // Handle successful connection - start recording
    if (voiceChat.isConnected) {
      setPendingVoiceChatRecord(false)
      voiceChat.startRecording()
        .then(() => {
          setRecordingState('recording')
          setActiveStream(voiceChat.activeStream)
        })
        .catch((err) => {
          setTranscriptionError(err instanceof Error ? err.message : 'Failed to start recording')
          setRecordingState('idle')
        })
    }
  }, [pendingVoiceChatRecord, voiceChat.isConnected, voiceChat.error, voiceChat])

  // Connection timeout - prevents infinite spinner if connection hangs
  useEffect(() => {
    if (!pendingVoiceChatRecord) return

    const timeout = setTimeout(() => {
      if (pendingVoiceChatRecord) {
        setPendingVoiceChatRecord(false)
        setRecordingState('idle')
        setTranscriptionError('Connection timed out. Please try again.')
        voiceChat.disconnect()
      }
    }, 10000) // 10 second timeout

    return () => clearTimeout(timeout)
  }, [pendingVoiceChatRecord, voiceChat])

  // Sync voiceChat state with local state
  // IMPORTANT: voiceChat.isRecording reflects whether the MediaRecorder is active.
  // During a voice chat turn, the flow is: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE
  // The MediaRecorder stays running through all these states until user explicitly stops.
  // We must keep showing the waveform (recordingState='recording') as long as isRecording is true.
  useEffect(() => {
    if (mode === 'conversation' && llmAvailable) {
      // Update recording state based on voiceChat
      // Priority: isRecording > other states
      // Use functional updates and guards to prevent infinite loops
      if (voiceChat.isRecording) {
        // MediaRecorder is active - always show recording UI regardless of voiceState
        // This ensures continuous listening works after each turn
        setRecordingState(prev => prev !== 'recording' ? 'recording' : prev)
        if (voiceChat.activeStream) {
          setActiveStream(voiceChat.activeStream)
        }
      } else if (voiceChat.voiceState === 'processing' || voiceChat.voiceState === 'speaking') {
        // Not recording but still processing/speaking (e.g., text input turn)
        setRecordingState(prev => prev !== 'processing' ? 'processing' : prev)
      } else if (voiceChat.voiceState === 'idle' && !pendingVoiceChatRecord) {
        // Voice chat is idle and we're not recording - reset to idle
        setRecordingState(prev => prev !== 'idle' ? 'idle' : prev)
      }

      // Update transcription display
      if (voiceChat.currentTranscription) {
        setTranscriptionError(null)
      }

      // Update error state (but not if we're handling it in the pending effect)
      if (voiceChat.error && !pendingVoiceChatRecord) {
        setTranscriptionError(voiceChat.error)
      }
    }
  }, [mode, llmAvailable, voiceChat.isRecording, voiceChat.activeStream, voiceChat.voiceState, voiceChat.currentTranscription, voiceChat.error, pendingVoiceChatRecord])

  // Sync STT-only voice chat state with local state (for turn detection-based transcription)
  // Now uses the same voiceChat hook with sttOnly mode
  useEffect(() => {
    if (mode === 'stt' && turnDetectionEnabled) {
      // Update recording state based on voiceChat (in sttOnly mode)
      // Use functional updates and guards to prevent infinite loops
      if (voiceChat.isRecording) {
        setRecordingState(prev => prev !== 'recording' ? 'recording' : prev)
        if (voiceChat.activeStream) {
          setActiveStream(voiceChat.activeStream)
        }
      } else if (voiceChat.voiceState === 'processing') {
        setRecordingState(prev => prev !== 'processing' ? 'processing' : prev)
      } else if (voiceChat.voiceState === 'idle' && !pendingVoiceChatRecord) {
        setRecordingState(prev => prev !== 'idle' ? 'idle' : prev)
      }

      // When we get a final transcription from STT-only mode, update the result
      if (voiceChat.currentTranscription) {
        // The transcription is final when voiceState returns to idle
        if (voiceChat.voiceState === 'idle' && voiceChat.currentTranscription) {
          setTranscriptionResult({
            text: voiceChat.currentTranscription,
            language: sttLanguage,
          })
          setTranscriptionError(null)
        }
      }

      // Update error state
      if (voiceChat.error && !pendingVoiceChatRecord) {
        setTranscriptionError(voiceChat.error)
      }
    }
  }, [mode, turnDetectionEnabled, voiceChat.isRecording, voiceChat.activeStream, voiceChat.voiceState, voiceChat.currentTranscription, voiceChat.error, pendingVoiceChatRecord, sttLanguage])

  // Comprehensive error reset effect - ensures recording state resets on any error
  // This catches cases where voiceState conditions don't trigger a reset
  // Only applies when using voiceChat (not local MediaRecorder)
  const shouldUseVoiceChat = (mode === 'conversation' && llmAvailable) || (mode === 'stt' && turnDetectionEnabled)

  useEffect(() => {
    // Only reset based on voiceChat state when we're supposed to be using voiceChat
    // When LLM is disabled, we use local MediaRecorder and shouldn't reset based on voiceChat
    if (!shouldUseVoiceChat) return

    // Reset on any error while not idle (unless we're handling it in pending effect)
    // Note: Don't clear activeStream here - the microphone may still be recording
    // and user should still see the waveform. The stream is cleared when recording stops.
    if (voiceChat.error && !pendingVoiceChatRecord) {
      setRecordingState(prev => prev !== 'idle' ? 'idle' : prev)
      // Don't clear activeStream - voiceChat may still be recording
      // The stream will be cleared when voiceChat.isRecording goes false
    }
    // Also reset when voiceChat goes idle but local state is stuck
    // (catches edge cases where main sync effect conditions don't match)
    if (voiceChat.voiceState === 'idle' && !voiceChat.isRecording &&
        !pendingVoiceChatRecord && !voiceChat.isPlayingAudio) {
      // Small delay to avoid race with legitimate state transitions
      const timer = setTimeout(() => {
        if (voiceChat.voiceState === 'idle' && !voiceChat.isRecording && !pendingVoiceChatRecord) {
          setRecordingState(prev => prev !== 'idle' ? 'idle' : prev)
        }
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [shouldUseVoiceChat, voiceChat.error, voiceChat.voiceState, voiceChat.isRecording, voiceChat.isPlayingAudio, pendingVoiceChatRecord])

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

    // Collapse settings on first interaction
    setConfigExpanded(false)

    // Use voice chat for LLM conversation mode
    if (mode === 'conversation' && llmAvailable) {
      if (voiceChat.isConnected) {
        // Already connected, start recording
        await voiceChat.startRecording()
        setRecordingState('recording')
        setActiveStream(voiceChat.activeStream)
      } else {
        // Need to connect first - set pending state and connect
        // The effect below will start recording once connected
        setPendingVoiceChatRecord(true)
        setRecordingState('processing') // Show loading state
        voiceChat.connect()
      }
      return
    }

    // Use voice chat for STT-only turn detection mode (sttOnly flag handles skipping LLM/TTS)
    if (mode === 'stt' && turnDetectionEnabled) {
      if (voiceChat.isConnected) {
        // Already connected, start recording
        await voiceChat.startRecording()
        setRecordingState('recording')
        setActiveStream(voiceChat.activeStream)
      } else {
        // Need to connect first - set pending state and connect
        setPendingVoiceChatRecord(true)
        setRecordingState('processing') // Show loading state
        voiceChat.connect()
      }
      return
    }

    // Fall back to local recording for STT-only (manual mode) or no LLM
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
  }, [micPermission, mode, llmAvailable, voiceChat])

  // Stop recording
  const stopRecording = useCallback(() => {
    // If using voice chat for conversation mode, use its stop method
    if (mode === 'conversation' && llmAvailable && voiceChat.isRecording) {
      voiceChat.stopRecording()
      setRecordingState('processing')
      setActiveStream(null)
      return
    }

    // If using STT-only voice chat for turn detection-based transcription
    if (mode === 'stt' && turnDetectionEnabled && voiceChat.isRecording) {
      voiceChat.stopRecording()
      setRecordingState('processing')
      setActiveStream(null)
      return
    }

    // Otherwise use local media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setRecordingState('processing')
    }
  }, [mode, llmAvailable, turnDetectionEnabled, voiceChat])

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

  // Stop any playing audio
  const stopAudioPlayback = useCallback(() => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause()
      audioPlayerRef.current.currentTime = 0
    }
    setPlayingMessageId(null)
  }, [])

  // Send text message / synthesize TTS
  const sendTextMessage = useCallback(async () => {
    if (!textInput.trim()) return

    // Stop any playing audio when user sends input
    stopAudioPlayback()

    // Collapse settings on first interaction
    setConfigExpanded(false)

    // Check if we should use LLM (regardless of mode - user might have STT off but LLM on)
    if (llmAvailable && activeProject) {
      // Use LLM for text input - conversation mode
      const inputText = textInput
      setTextInput('')

      // Add user message immediately
      const userMessage: SpeechMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        text: inputText,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, userMessage])

      // Start waiting for LLM response
      setIsWaitingForLLM(true)

      // Send via REST API with full conversation history
      try {
        // Convert existing messages to API format and append new user message
        // Include system prompt for concise voice-friendly responses
        const apiMessages = [
          { role: 'system' as const, content: voiceSystemPrompt },
          ...messages.map(m => ({ role: m.role, content: m.text })),
          { role: 'user' as const, content: inputText },
        ]
        const result = await sendChatCompletion(
          activeProject.namespace,
          activeProject.project,
          {
            messages: apiMessages,
            model: selectedLLMModel,
          }
        )

        // LLM response received
        setIsWaitingForLLM(false)

        // Add assistant response
        const assistantText = result.response.choices?.[0]?.message?.content || ''
        if (assistantText) {
          const messageId = `msg-${Date.now() + 1}`

          // Add message first (without audio)
          const assistantMessage: SpeechMessage = {
            id: messageId,
            role: 'assistant',
            text: assistantText,
            timestamp: new Date(),
          }
          setMessages(prev => [...prev, assistantMessage])

          // If TTS is enabled, synthesize and UPDATE message with audioUrl
          if (ttsEnabled) {
            setIsSynthesizingTypedTTS(true)
            try {
              const audioBlob = await synthesizeSpeech({
                model: ttsModel,
                input: assistantText,
                voice: ttsVoice,
                speed: ttsSpeed,
                response_format: 'mp3',
              })
              const audioUrl = URL.createObjectURL(audioBlob)
              // Track the blob URL for cleanup to prevent memory leaks
              blobUrlsRef.current.set(messageId, audioUrl)

              // Update the message with audioUrl so play button appears
              setMessages(prev => prev.map(msg =>
                msg.id === messageId ? { ...msg, audioUrl } : msg
              ))

              // Auto-play the audio with tracking for stop button
              const audio = new Audio(audioUrl)
              typedAudioRef.current = audio
              setIsTypedAudioPlaying(true)
              audio.onended = () => {
                setIsTypedAudioPlaying(false)
                typedAudioRef.current = null
              }
              audio.onerror = () => {
                setIsTypedAudioPlaying(false)
                typedAudioRef.current = null
              }
              audio.play().catch(() => {
                setIsTypedAudioPlaying(false)
                typedAudioRef.current = null
              })
            } catch (ttsErr) {
              console.error('TTS synthesis failed:', ttsErr)
              // Show a brief error message to the user
              const errorMessage = ttsErr instanceof Error ? ttsErr.message : 'Audio synthesis failed'
              if (errorMessage.includes('Network Error')) {
                setTtsError('TTS unavailable - Universal Runtime not running')
              } else {
                setTtsError(`Audio synthesis failed: ${errorMessage}`)
              }
              // Clear error after 5 seconds
              setTimeout(() => setTtsError(null), 5000)
            } finally {
              setIsSynthesizingTypedTTS(false)
            }
          }
        }
      } catch (err) {
        setIsWaitingForLLM(false)
        const message = err instanceof Error ? err.message : 'Chat request failed'
        setTranscriptionError(message)
      }
      return
    }

    if (mode === 'tts') {
      // TTS-only mode (no LLM): synthesize the text and add to history
      setIsSynthesizing(true)
      const inputTextCopy = textInput
      setTtsError(null)

      try {
        const audioBlob = await synthesizeSpeech({
          model: ttsModel,
          input: inputTextCopy,
          voice: ttsVoice,
          speed: ttsSpeed,
          response_format: 'mp3',
        })

        const audioUrl = URL.createObjectURL(audioBlob)
        const newItemId = `tts-${Date.now()}`

        // Add to history
        const newItem: TTSHistoryItem = {
          id: newItemId,
          text: inputTextCopy,
          audioUrl,
          timestamp: new Date(),
        }
        setTtsHistory(prev => [...prev, newItem])
        setBackendConnected(true)
        setTextInput('')

        // Auto-play the new audio
        setPlayingTtsId(newItemId)
        const audio = new Audio(audioUrl)
        ttsAudioRef.current = audio
        audio.onended = () => setPlayingTtsId(null)
        audio.onerror = () => setPlayingTtsId(null)
        audio.play().catch(() => setPlayingTtsId(null))
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Speech synthesis failed'
        setTtsError(message)
        setBackendConnected(false)
      } finally {
        setIsSynthesizing(false)
      }
    }
  }, [textInput, mode, ttsModel, ttsVoice, ttsSpeed, ttsEnabled, llmAvailable, activeProject, selectedLLMModel, messages, voiceSystemPrompt, stopAudioPlayback])

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

  // Play TTS history item
  const handlePlayTtsHistory = useCallback((itemId: string) => {
    const item = ttsHistory.find(i => i.id === itemId)
    if (!item) return

    // If already playing this item, stop it
    if (playingTtsId === itemId) {
      if (ttsAudioRef.current) {
        ttsAudioRef.current.pause()
        ttsAudioRef.current = null
      }
      setPlayingTtsId(null)
      return
    }

    // Stop any currently playing audio
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause()
    }

    // Play the new item
    const audio = new Audio(item.audioUrl)
    ttsAudioRef.current = audio
    setPlayingTtsId(itemId)
    audio.onended = () => setPlayingTtsId(null)
    audio.onerror = () => setPlayingTtsId(null)
    audio.play().catch(() => setPlayingTtsId(null))
  }, [ttsHistory, playingTtsId])

  // Play message audio
  const handlePlayMessageAudio = useCallback((messageId: string) => {
    if (playingMessageId === messageId) {
      setPlayingMessageId(null)
    } else {
      setPlayingMessageId(messageId)
    }
  }, [playingMessageId])

  // Clear all content (works in all modes: conversation, STT-only, TTS-only)
  const handleClearConversation = useCallback(() => {
    // Stop any ongoing audio playback and backend generation
    voiceChat.interrupt()

    // Clean up blob URLs for typed messages before clearing
    for (const [id, url] of blobUrlsRef.current.entries()) {
      if (id.startsWith('msg-')) {
        URL.revokeObjectURL(url)
        blobUrlsRef.current.delete(id)
      }
    }

    // Clear conversation mode state
    setMessages([])
    voiceChat.clearMessages()
    setPlayingMessageId(null)

    // Clear STT-only mode state
    setTranscriptionResult(null)
    setTranscriptionError(null)
    setIsTranscribing(false)

    // Clear TTS-only mode state
    setTtsError(null)
    setIsSynthesizing(false)
    setTtsHistory([])
    setPlayingTtsId(null)
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause()
      ttsAudioRef.current = null
    }

    // Clear text input
    setTextInput('')
  }, [voiceChat])

  // Expose clear function to parent via ref
  useEffect(() => {
    if (clearRef) {
      clearRef.current = handleClearConversation
    }
    return () => {
      if (clearRef) {
        clearRef.current = null
      }
    }
  }, [clearRef, handleClearConversation])

  // Notify parent when there's content to clear (in any mode)
  useEffect(() => {
    const hasContent =
      messages.length > 0 ||           // Conversation mode
      transcriptionResult !== null ||   // STT-only mode
      ttsHistory.length > 0             // TTS-only mode (history)
    onMessagesChange?.(hasContent)
  }, [messages.length, transcriptionResult, ttsHistory.length, onMessagesChange])

  // Play audio when playingMessageId changes
  useEffect(() => {
    if (!playingMessageId) {
      // Stop playback if no message selected
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause()
      }
      return
    }

    const message = messages.find(m => m.id === playingMessageId)
    if (!message?.audioUrl) {
      setPlayingMessageId(null)
      return
    }

    // Create or reuse audio element
    if (!audioPlayerRef.current) {
      audioPlayerRef.current = new Audio()
    }

    const audio = audioPlayerRef.current
    audio.src = message.audioUrl
    audio.onended = () => setPlayingMessageId(null)
    audio.onerror = () => setPlayingMessageId(null)
    audio.play().catch(() => setPlayingMessageId(null))

    return () => {
      audio.onended = null
      audio.onerror = null
    }
  }, [playingMessageId, messages])

  // Stop audio playback when user starts recording or speaking
  useEffect(() => {
    if (recordingState === 'recording' || voiceChat.isRecording) {
      stopAudioPlayback()
    }
  }, [recordingState, voiceChat.isRecording, stopAudioPlayback])

  // Show mic permission prompt if needed and trying to record
  const needsMicPermission = micPermission !== 'granted' && sttEnabled

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Configuration Section - Collapsible */}
      <div className="flex-shrink-0 border-b border-border">
        {/* Header - always visible */}
        <button
          onClick={() => setConfigExpanded(!configExpanded)}
          className="w-full px-4 py-2 flex items-center justify-between hover:bg-muted/50 transition-colors"
        >
          <div className="flex items-center gap-2 text-sm font-medium">
            <Settings2 className="h-4 w-4 text-muted-foreground" />
            <span>Models & Settings</span>
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
                onEnabledChange={handleSttEnabledChange}
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
                onEnabledChange={handleTtsEnabledChange}
                selectedModel={ttsModel}
                onModelChange={(model) => {
                  setTtsModel(model)
                  // Reset voice to first available for new model
                  const modelVoices = getVoicesForModel(model)
                  const newVoice = modelVoices[0]?.id || 'af_heart'
                  setTtsVoice(newVoice)
                  // Update the connected session if already connected
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ ttsModel: model, ttsVoice: newVoice })
                  }
                }}
                selectedVoice={ttsVoice}
                onVoiceChange={(voice) => {
                  setTtsVoice(voice)
                  // Update the connected session if already connected
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ ttsVoice: voice })
                  }
                }}
                speed={ttsSpeed}
                onSpeedChange={(speed) => {
                  setTtsSpeed(speed)
                  // Update the connected session if already connected
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ speed })
                  }
                }}
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

            {/* LLM Model Selection + Turn Detection Settings - always visible, grayed out when irrelevant */}
            <div className="grid grid-cols-2 gap-2">
              {/* LLM Response Card */}
              <div className={`rounded-lg border border-border bg-card/40 p-3 ${!sttEnabled && !ttsEnabled ? 'opacity-50' : ''}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">LLM Response</span>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent side="top" className="max-w-[200px]">
                          <p>{!sttEnabled && !ttsEnabled
                            ? 'Enable STT or TTS to use LLM responses.'
                            : 'Enable for two-way conversation. When off, speech is only echoed back.'}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  {/* Toggle + Connection status */}
                  {activeProject && availableLLMModels.length > 0 && (sttEnabled || ttsEnabled) && (
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={llmEnabled}
                        onCheckedChange={(enabled) => {
                          setLlmEnabled(enabled)
                          // Disconnect if turning off while connected
                          if (!enabled && voiceChat.isConnected) {
                            voiceChat.disconnect()
                          }
                        }}
                        aria-label="Enable LLM responses"
                      />
                      <span className="text-xs text-muted-foreground">
                        {llmEnabled ? (
                          voiceChat.isConnected ? (
                            <span className="text-green-600">Connected</span>
                          ) : 'Enabled'
                        ) : 'Off'}
                      </span>
                    </div>
                  )}
                </div>
                {activeProject && availableLLMModels.length > 0 ? (
                  <div className={!llmEnabled || (!sttEnabled && !ttsEnabled) ? 'opacity-50 pointer-events-none' : ''}>
                    <Selector
                      value={selectedLLMModel}
                      options={availableLLMModels.map(m => ({
                        value: m.name,
                        label: m.name,
                        description: m.model,
                      }))}
                      onChange={setSelectedLLMModel}
                      disabled={!llmEnabled || voiceChat.isConnected || (!sttEnabled && !ttsEnabled)}
                    />
                  </div>
                ) : (
                  <span className="text-xs text-muted-foreground">
                    {!activeProject
                      ? 'Select a project to enable LLM conversation'
                      : 'No LLM models configured in this project'}
                  </span>
                )}
              </div>

              {/* Turn Detection Settings Card - always visible, grayed out when STT is disabled */}
              <TurnDetectionSettings
                enabled={turnDetectionEnabled}
                onEnabledChange={setTurnDetectionEnabled}
                baseSilenceDuration={baseSilenceDuration}
                onBaseSilenceDurationChange={(duration) => {
                  setBaseSilenceDuration(duration)
                  // Update the connected session if already connected
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ baseSilenceDuration: duration })
                  }
                }}
                thinkingSilenceDuration={thinkingSilenceDuration}
                onThinkingSilenceDurationChange={(duration) => {
                  setThinkingSilenceDuration(duration)
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ thinkingSilenceDuration: duration })
                  }
                }}
                maxSilenceDuration={maxSilenceDuration}
                onMaxSilenceDurationChange={(duration) => {
                  setMaxSilenceDuration(duration)
                  if (voiceChat.isConnected) {
                    voiceChat.updateConfig({ maxSilenceDuration: duration })
                  }
                }}
                sttDisabled={!sttEnabled}
              />
            </div>
          </div>
        )}
      </div>

      {/* Test Area */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* Mic permission prompt - scrollable container for short screens */}
        {needsMicPermission && (
          <div className="flex-1 min-h-0 overflow-y-auto p-4">
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
              {/* TTS error display for conversation mode */}
              {ttsError && (
                <div className="flex-shrink-0 mx-4 mt-4 flex items-center gap-2 p-3 rounded-lg bg-amber-500/10 text-amber-600 text-sm">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <span>{ttsError}</span>
                  <button
                    onClick={() => setTtsError(null)}
                    className="ml-auto text-xs hover:text-amber-800"
                  >
                    Dismiss
                  </button>
                </div>
              )}
              <ConversationView
                messages={messages}
                onPlayAudio={handlePlayMessageAudio}
                playingMessageId={playingMessageId}
                streamingUserText={llmAvailable ? voiceChat.currentTranscription : undefined}
                streamingAssistantText={llmAvailable ? voiceChat.currentLLMText : undefined}
                isSpeaking={isTypedAudioPlaying || (llmAvailable && (voiceChat.voiceState === 'speaking' || voiceChat.isPlayingAudio))}
                onStopSpeaking={() => {
                  // Stop typed input audio if playing
                  if (typedAudioRef.current) {
                    typedAudioRef.current.pause()
                    typedAudioRef.current = null
                    setIsTypedAudioPlaying(false)
                  }
                  // Also stop voiceChat audio
                  voiceChat.interrupt()
                }}
                isThinking={isWaitingForLLM || voiceChat.voiceState === 'processing'}
                isSynthesizingTTS={isSynthesizingTypedTTS}
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
            <div className="h-full flex flex-col">
              {/* Error message */}
              {ttsError && (
                <div className="flex-shrink-0 mx-4 mt-4 flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-600 text-sm">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <span>{ttsError}</span>
                  <button
                    onClick={() => setTtsError(null)}
                    className="ml-auto text-xs hover:text-red-800"
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {/* TTS History - scrollable area */}
              <div className="flex-1 overflow-y-auto p-4">
                {ttsHistory.length === 0 ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center px-6 py-10">
                      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                        <Volume2 className="w-5 h-5 text-primary" />
                      </div>
                      <div className="text-lg font-medium text-foreground">
                        Text-to-Speech
                      </div>
                      <div className="mt-1 text-sm text-muted-foreground">
                        Type text below to hear it spoken aloud
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {ttsHistory.map((item) => (
                      <div key={item.id} className="flex items-start gap-3">
                        {/* Play button */}
                        <button
                          onClick={() => handlePlayTtsHistory(item.id)}
                          className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
                            playingTtsId === item.id
                              ? 'bg-primary/20 text-primary'
                              : 'bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground'
                          }`}
                          aria-label={playingTtsId === item.id ? 'Stop' : 'Play'}
                        >
                          {playingTtsId === item.id ? (
                            <StopCircle className="w-5 h-5" />
                          ) : (
                            <Volume2 className={`w-5 h-5 ${playingTtsId === item.id ? 'animate-pulse' : ''}`} />
                          )}
                        </button>

                        {/* Text content */}
                        <div className="flex-1 min-w-0">
                          <p className="text-base leading-relaxed text-foreground">
                            {item.text}
                          </p>
                          <span className="text-xs text-muted-foreground mt-1 block">
                            {item.timestamp.toLocaleTimeString(undefined, {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Input area at bottom */}
              <div className="flex-shrink-0 p-3 border-t border-border bg-background/60">
                <div className="flex items-center gap-2">
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        sendTextMessage()
                      }
                    }}
                    placeholder="Type text to speak..."
                    rows={1}
                    className="flex-1 px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring text-sm min-h-[40px] max-h-[120px]"
                    style={{ height: 'auto' }}
                  />
                  <Button
                    size="icon"
                    className="h-10 w-10 rounded-full flex-shrink-0"
                    onClick={sendTextMessage}
                    disabled={!textInput.trim() || isSynthesizing}
                    aria-label={isSynthesizing ? 'Synthesizing...' : 'Speak'}
                  >
                    {isSynthesizing ? (
                      <div className="w-5 h-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    ) : (
                      <Volume2 className="h-5 w-5" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
        )}

        {/* Input Area (for conversation and STT modes) */}
        {!needsMicPermission && (mode === 'conversation' || mode === 'stt') && (
          <div className="flex-shrink-0 p-3 border-t border-border bg-background/60">
            <div className="flex items-center gap-2">
              {/* Recording mode: full-width waveform with stop button on right */}
              {/* Check both local state and voiceChat state to handle race conditions */}
              {(recordingState === 'recording' || voiceChat.isRecording) && (activeStream || voiceChat.activeStream) ? (
                <>
                  {/* Waveform with optional helper text for manual mode */}
                  <div className="flex-1 flex flex-col items-center justify-center gap-1">
                    <Waveform
                      stream={activeStream || voiceChat.activeStream}
                      isActive={true}
                      height={24}
                      barCount={80}
                      gap={1}
                      color="rgb(156, 163, 175)"
                      className="w-full"
                    />
                    {/* Show helper text when turn detection is off (manual mode) */}
                    {!turnDetectionEnabled && (
                      <span className="text-xs text-muted-foreground">
                        Tap stop when done speaking
                      </span>
                    )}
                  </div>

                  {/* Stop button */}
                  <Button
                    variant="destructive"
                    size="icon"
                    className="h-10 w-10 rounded-full flex-shrink-0"
                    onClick={stopRecording}
                    aria-label="Stop recording"
                  >
                    <StopCircle className="h-5 w-5" />
                  </Button>
                </>
              ) : (
                <>
                  {/* Text input (only for conversation mode, hidden when recording) */}
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
                    <div className="flex-1 py-2">
                      <div className="text-sm text-muted-foreground">
                        {recordingState === 'idle'
                          ? 'Click the microphone to start recording'
                          : 'Processing audio...'}
                      </div>
                    </div>
                  )}

                  {/* Mic button */}
                  {sttEnabled && micPermission === 'granted' && (
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-10 w-10 rounded-full flex-shrink-0"
                      onClick={startRecording}
                      disabled={recordingState === 'processing' || (llmAvailable && !selectedLLMModel)}
                      aria-label="Start recording"
                    >
                      {recordingState === 'processing' ? (
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
                      className="h-10 w-10 rounded-full flex-shrink-0"
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
                      className="h-10 w-10 rounded-full flex-shrink-0"
                      onClick={sendTextMessage}
                      disabled={!textInput.trim()}
                      aria-label="Send message"
                    >
                      <Send className="h-5 w-5" />
                    </Button>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
