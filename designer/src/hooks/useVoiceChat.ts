/**
 * useVoiceChat - Hook for managing voice chat sessions
 *
 * Handles:
 * - WebSocket connection to voice chat endpoint
 * - Audio recording and sending
 * - Audio playback queue for TTS responses
 * - State management for the voice pipeline
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import {
  createVoiceChatConnection,
  sendInterrupt,
  sendEndSignal,
  sendConfigUpdate,
  type VoiceState,
  type VoiceChatConfig,
} from '../api/voiceService'

export interface VoiceMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  timestamp: Date
  audioData?: ArrayBuffer
}

export interface UseVoiceChatOptions {
  namespace: string
  project: string
  llmModel: string
  sttModel?: string
  ttsModel?: string
  ttsVoice?: string
  language?: string
  speed?: number
  systemPrompt?: string
  // LLM enabled flag - when false, transcription works but LLM response is filtered (frontend-only)
  llmEnabled?: boolean
  // Turn detection settings (sent via config message after connect)
  turnDetectionEnabled?: boolean
  baseSilenceDuration?: number    // For complete utterances (0.1-2.0s, default 0.4)
  thinkingSilenceDuration?: number // For incomplete utterances (0.3-5.0s, default 1.2)
  maxSilenceDuration?: number     // Hard timeout (0.5-10.0s, default 2.5)
  // Barge-in settings
  bargeInEnabled?: boolean
  // Emotion detection callbacks
  onEmotion?: (emotion: string, confidence: number, allScores: Record<string, number>) => void
  onToolCall?: (toolCallId: string, functionName: string, args: string) => void
  autoConnect?: boolean
  onError?: (error: string) => void
}

export interface UseVoiceChatReturn {
  // Connection state
  isConnected: boolean
  sessionId: string | null
  voiceState: VoiceState
  error: string | null

  // Messages
  messages: VoiceMessage[]
  currentTranscription: string
  currentLLMText: string

  // Recording state
  isRecording: boolean
  activeStream: MediaStream | null

  // Audio playback state
  isPlayingAudio: boolean

  // Actions
  connect: () => void
  disconnect: () => void
  startRecording: () => Promise<void>
  stopRecording: () => void
  interrupt: () => void
  stopAudio: () => void
  clearMessages: () => void
  updateConfig: (config: Partial<VoiceChatConfig>) => void
}

export function useVoiceChat(options: UseVoiceChatOptions): UseVoiceChatReturn {
  const {
    namespace,
    project,
    llmModel,
    sttModel,
    ttsModel,
    ttsVoice,
    language,
    speed,
    systemPrompt,
    llmEnabled = true,
    turnDetectionEnabled,
    baseSilenceDuration,
    thinkingSilenceDuration,
    maxSilenceDuration,
    bargeInEnabled,
    onEmotion,
    onToolCall,
    autoConnect = false,
    onError,
  } = options

  // WebSocket connection
  const wsRef = useRef<WebSocket | null>(null)
  const disconnectRef = useRef<(() => void) | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [error, setError] = useState<string | null>(null)

  // Messages
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  const [currentTranscription, setCurrentTranscription] = useState('')
  const [currentLLMText, setCurrentLLMText] = useState('')

  // Recording - using AudioWorklet for raw PCM capture
  const [isRecording, setIsRecording] = useState(false)
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null)
  const recordingContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)

  // Audio playback
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioQueueRef = useRef<ArrayBuffer[]>([])
  const isPlayingRef = useRef(false)
  const audioInterruptedRef = useRef(false) // Flag to stop audio queue processing
  const [isPlayingAudio, setIsPlayingAudio] = useState(false) // Exposed state for UI
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null) // Track current audio source for stopping
  const currentUserTextRef = useRef('')
  const currentAssistantTextRef = useRef('')
  const currentAssistantAudioRef = useRef<ArrayBuffer[]>([])
  const hasConnectedRef = useRef(false) // Track if we ever successfully connected
  const errorReceivedRef = useRef(false) // Track if we received an error message from server
  // Refs to track latest state values for use in callbacks (avoids stale closures)
  const isRecordingRef = useRef(false)
  const voiceStateRef = useRef<VoiceState>('idle')

  // Keep refs in sync with state for use in callbacks
  useEffect(() => {
    isRecordingRef.current = isRecording
  }, [isRecording])

  useEffect(() => {
    voiceStateRef.current = voiceState
  }, [voiceState])

  // Get or create audio context
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new AudioContext({ sampleRate: 24000 })
    }
    return audioContextRef.current
  }, [])

  // Play audio from buffer
  const playAudioBuffer = useCallback(async (audioData: ArrayBuffer) => {
    try {
      const audioContext = getAudioContext()

      // Resume if suspended (browser autoplay policy)
      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      // Convert PCM 24kHz 16-bit mono to AudioBuffer
      const int16Array = new Int16Array(audioData)
      const float32Array = new Float32Array(int16Array.length)
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0
      }

      const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000)
      audioBuffer.copyToChannel(float32Array, 0)

      const source = audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext.destination)

      // Track current source so we can stop it on interrupt
      currentSourceRef.current = source
      source.start()

      return new Promise<void>((resolve) => {
        source.onended = () => {
          if (currentSourceRef.current === source) {
            currentSourceRef.current = null
          }
          resolve()
        }
      })
    } catch (err) {
      console.error('Failed to play audio:', err)
      // Reset playback state on error to prevent queue blocking
      isPlayingRef.current = false
      setIsPlayingAudio(false)
      audioQueueRef.current = [] // Clear queue to prevent cascade errors
    }
  }, [getAudioContext])

  // Process audio queue
  const processAudioQueue = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return
    }

    isPlayingRef.current = true
    audioInterruptedRef.current = false // Reset interrupt flag
    setIsPlayingAudio(true)

    while (audioQueueRef.current.length > 0 && !audioInterruptedRef.current) {
      const audioData = audioQueueRef.current.shift()
      if (audioData && !audioInterruptedRef.current) {
        await playAudioBuffer(audioData)
      }
    }

    // Add a small delay after playback stops before allowing audio capture
    // This gives echo cancellation time to settle and prevents the mic
    // from picking up the tail end of TTS audio
    if (!audioInterruptedRef.current) {
      await new Promise(resolve => setTimeout(resolve, 300))
    }

    isPlayingRef.current = false
    setIsPlayingAudio(false)
  }, [playAudioBuffer])

  // Connect to voice chat WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setError(null)
    hasConnectedRef.current = false // Reset connection tracking
    errorReceivedRef.current = false // Reset error tracking

    const config: VoiceChatConfig = {
      llmModel,
      sttModel,
      ttsModel,
      ttsVoice,
      language,
      speed,
      systemPrompt,
      // Note: turn detection is sent via config message after connect
      turnDetectionEnabled,
      baseSilenceDuration,
      thinkingSilenceDuration,
      maxSilenceDuration,
      bargeInEnabled,
    }

    const ws = createVoiceChatConnection(namespace, project, config, {
      onSessionInfo: (id) => {
        hasConnectedRef.current = true // Mark as successfully connected
        setSessionId(id)
        setIsConnected(true)
        // Send turn detection config via config message (backend ignores query params)
        if (ws.readyState === WebSocket.OPEN) {
          sendConfigUpdate(ws, {
            turn_detection_enabled: turnDetectionEnabled,
            base_silence_duration: baseSilenceDuration,
            thinking_silence_duration: thinkingSilenceDuration,
            max_silence_duration: maxSilenceDuration,
          })
        }
      },
      onStateChange: (state) => {
        setVoiceState(state)
      },
      onTranscription: (text, isFinal) => {
        setCurrentTranscription(text)
        if (isFinal && text.trim()) {
          // Add user message immediately when transcription is final
          // This ensures user sees their message even if LLM fails/errors
          const userMessage: VoiceMessage = {
            id: `user-${Date.now()}`,
            role: 'user',
            text: text,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, userMessage])
          setCurrentTranscription('') // Clear streaming display
          currentUserTextRef.current = '' // Clear ref since message is added
        }
      },
      onLLMText: (text, isFinal) => {
        // When LLM is disabled (frontend-only), skip LLM response display
        // User message is already added in onTranscription
        if (!llmEnabled) {
          // Clear audio/text buffers when response is final to prevent memory leak
          // (onAudio still accumulates data even when LLM display is disabled)
          if (isFinal) {
            currentAssistantAudioRef.current = []
            currentAssistantTextRef.current = ''
          }
          return // Skip LLM response display
        }

        // User message is already added in onTranscription, just handle assistant response

        // Accumulate the text for the assistant response
        currentAssistantTextRef.current += text
        // Update display with the full accumulated text (not appending to state)
        setCurrentLLMText(currentAssistantTextRef.current)

        if (isFinal) {
          // LLM response complete - add the full assistant message now
          if (currentAssistantTextRef.current) {
            const combinedAudio = currentAssistantAudioRef.current.length > 0
              ? combineAudioBuffers(currentAssistantAudioRef.current)
              : undefined

            const assistantMessage: VoiceMessage = {
              id: `assistant-${Date.now()}`,
              role: 'assistant',
              text: currentAssistantTextRef.current,
              timestamp: new Date(),
              audioData: combinedAudio,
            }
            setMessages((prev) => [...prev, assistantMessage])
            currentAssistantTextRef.current = ''
            currentAssistantAudioRef.current = []
          }
          setCurrentLLMText('')
        }
      },
      onTTSDone: () => {
        // TTS done is sent per-phrase, not at end of response
        // We don't add messages here - messages are added when LLM text is final
        // This callback can be used for other purposes like tracking playback progress
      },
      onAudio: (audioData) => {
        // Store for message history
        currentAssistantAudioRef.current.push(audioData)
        // Queue for immediate playback
        audioQueueRef.current.push(audioData)
        processAudioQueue()
      },
      onError: (message) => {
        errorReceivedRef.current = true // Mark that we received a server error
        setError(message)
        setVoiceState('idle') // Reset to idle on error
        setIsRecording(false) // Ensure recording stops
        onError?.(message)
      },
      onClose: () => {
        // If we were recording/processing, this is unexpected - show error
        // Use refs to get latest state values (avoids stale closure)
        const wasActive = isRecordingRef.current || voiceStateRef.current !== 'idle'
        if (wasActive && !errorReceivedRef.current) {
          const errorMsg = hasConnectedRef.current
            ? 'Connection lost. Please try again.'
            : 'Failed to connect to voice chat server. Is the server running on port 14345?'
          setError(errorMsg)
          onError?.(errorMsg)
        } else if (!hasConnectedRef.current && !errorReceivedRef.current) {
          // Never connected - connection failure
          const errorMsg = 'Failed to connect to voice chat server. Is the server running on port 14345?'
          setError(errorMsg)
          onError?.(errorMsg)
        }
        setIsConnected(false)
        setSessionId(null)
        setVoiceState('idle')
        setIsRecording(false) // Ensure recording state is reset
      },
      onEmotion: (emotion, confidence, allScores) => {
        onEmotion?.(emotion, confidence, allScores)
      },
      onToolCall: (toolCallId, functionName, args) => {
        onToolCall?.(toolCallId, functionName, args)
      },
    })

    wsRef.current = ws
  }, [namespace, project, llmModel, sttModel, ttsModel, ttsVoice, language, speed, systemPrompt, llmEnabled, turnDetectionEnabled, baseSilenceDuration, thinkingSilenceDuration, maxSilenceDuration, bargeInEnabled, onError, onEmotion, onToolCall, processAudioQueue])

  // Disconnect from voice chat
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setSessionId(null)
    setVoiceState('idle')

    // Stop any ongoing recording (AudioWorklet cleanup)
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }
    if (recordingContextRef.current) {
      recordingContextRef.current.close()
      recordingContextRef.current = null
    }
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop())
      setActiveStream(null)
    }
    setIsRecording(false)
  }, [activeStream])

  // Keep disconnect ref updated for cleanup effect
  disconnectRef.current = disconnect

  // Start recording using AudioWorklet for raw PCM capture
  const startRecording = useCallback(async () => {
    if (!isConnected) {
      setError('Not connected to voice chat')
      return
    }

    // If assistant is speaking, interrupt first (barge-in)
    if (voiceState === 'speaking' && wsRef.current?.readyState === WebSocket.OPEN) {
      sendInterrupt(wsRef.current)
    }

    try {
      // Get microphone stream with echo cancellation
      // Echo cancellation is applied at getUserMedia level, before Web Audio API
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      setActiveStream(stream)

      // Create AudioContext at 16kHz for recording
      const audioContext = new AudioContext({ sampleRate: 16000 })
      recordingContextRef.current = audioContext

      // Load AudioWorklet processor
      await audioContext.audioWorklet.addModule('/audio-processor.js')

      // Create source from mic stream
      const source = audioContext.createMediaStreamSource(stream)
      sourceNodeRef.current = source

      // Create worklet node for PCM capture
      const workletNode = new AudioWorkletNode(audioContext, 'pcm-processor')
      workletNodeRef.current = workletNode

      // Handle PCM data from worklet - send directly to backend
      // Don't send audio while frontend is playing TTS to prevent echo feedback.
      // Even with echo cancellation, some audio can leak through.
      workletNode.port.onmessage = (event) => {
        // Skip sending if we're playing audio (prevents echo from being transcribed)
        if (isPlayingRef.current) {
          return
        }
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          // Send raw PCM directly (ArrayBuffer)
          wsRef.current.send(event.data)
        }
      }

      // Connect: mic → worklet (no destination - don't play back locally)
      source.connect(workletNode)

      setIsRecording(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start recording'
      setError(message)
      onError?.(message)
    }
  }, [isConnected, voiceState, onError])

  // Stop recording
  const stopRecording = useCallback(() => {
    // Disconnect AudioWorklet nodes
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }
    if (recordingContextRef.current) {
      recordingContextRef.current.close()
      recordingContextRef.current = null
    }

    // Stop media stream tracks
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop())
      setActiveStream(null)
    }

    setIsRecording(false)

    // Only send end signal if we're actively processing (not idle)
    // This prevents processing residual noise when user stops after conversation completes
    if (wsRef.current?.readyState === WebSocket.OPEN && voiceState !== 'idle') {
      sendEndSignal(wsRef.current)
    }
  }, [activeStream, voiceState])

  // Stop audio playback immediately
  const stopAudio = useCallback(() => {
    // Set interrupt flag to stop queue processing
    audioInterruptedRef.current = true
    // Clear the queue
    audioQueueRef.current = []
    // Stop currently playing audio
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop()
      } catch {
        // Ignore errors if already stopped
      }
      currentSourceRef.current = null
    }
    isPlayingRef.current = false
    setIsPlayingAudio(false)
  }, [])

  // Interrupt TTS (barge-in) - stops both backend generation and frontend playback
  const interrupt = useCallback(() => {
    // Stop frontend audio playback
    stopAudio()
    // Tell backend to stop generating
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sendInterrupt(wsRef.current)
    }
  }, [stopAudio])

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([])
    setCurrentTranscription('')
    setCurrentLLMText('')
    currentUserTextRef.current = ''
    currentAssistantTextRef.current = ''
    currentAssistantAudioRef.current = []
  }, [])

  // Update session config
  const updateConfig = useCallback((config: Partial<VoiceChatConfig>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sendConfigUpdate(wsRef.current, {
        stt_model: config.sttModel,
        tts_model: config.ttsModel,
        tts_voice: config.ttsVoice,
        llm_model: config.llmModel,
        language: config.language,
        speed: config.speed,
        sentence_boundary_only: config.sentenceBoundaryOnly,
        // Turn detection settings
        turn_detection_enabled: config.turnDetectionEnabled,
        base_silence_duration: config.baseSilenceDuration,
        thinking_silence_duration: config.thinkingSilenceDuration,
        max_silence_duration: config.maxSilenceDuration,
        // Barge-in settings
        barge_in_enabled: config.bargeInEnabled,
        barge_in_noise_filter: config.bargeInNoiseFilter,
        barge_in_min_chunks: config.bargeInMinChunks,
        // Emotion detection settings
        emotion_detection_enabled: config.emotionDetectionEnabled,
        emotion_model: config.emotionModel,
        emotion_confidence_threshold: config.emotionConfidenceThreshold,
      })
    }
  }, [])

  // Auto-connect on mount if enabled
  // Use a ref for connect to avoid triggering cleanup on every callback recreation
  const connectRef = useRef<(() => void) | null>(null)
  connectRef.current = connect

  useEffect(() => {
    let didAutoConnect = false
    if (autoConnect && namespace && project && llmModel) {
      didAutoConnect = true
      connectRef.current?.()
    }

    return () => {
      // Only disconnect if we auto-connected in this effect instance
      // Manual connections (via connect() call) are managed separately
      if (didAutoConnect) {
        disconnectRef.current?.()
      }
    }
    // Note: connectRef is used instead of connect to avoid cleanup running
    // on every callback recreation (which would disconnect manual connections)
  }, [autoConnect, namespace, project, llmModel])

  return {
    isConnected,
    sessionId,
    voiceState,
    error,
    messages,
    currentTranscription,
    currentLLMText,
    isRecording,
    isPlayingAudio,
    activeStream,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    interrupt,
    stopAudio,
    clearMessages,
    updateConfig,
  }
}

// Helper to combine multiple audio buffers
function combineAudioBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
  const totalLength = buffers.reduce((sum, buf) => sum + buf.byteLength, 0)
  const combined = new Uint8Array(totalLength)
  let offset = 0
  for (const buffer of buffers) {
    combined.set(new Uint8Array(buffer), offset)
    offset += buffer.byteLength
  }
  return combined.buffer
}

export default useVoiceChat
