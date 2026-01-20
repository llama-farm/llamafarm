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
  sendAudioData,
  sendInterrupt,
  sendEndSignal,
  sendConfigUpdate,
  sendTextMessage as sendTextToWs,
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
  silenceDuration?: number // VAD silence duration in seconds (0.2-2.0, default 0.4)
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

  // Actions
  connect: () => void
  disconnect: () => void
  startRecording: () => Promise<void>
  stopRecording: () => void
  sendTextMessage: (text: string) => void
  interrupt: () => void
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
    silenceDuration,
    autoConnect = false,
    onError,
  } = options

  // WebSocket connection
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [error, setError] = useState<string | null>(null)

  // Messages
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  const [currentTranscription, setCurrentTranscription] = useState('')
  const [currentLLMText, setCurrentLLMText] = useState('')

  // Recording
  const [isRecording, setIsRecording] = useState(false)
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)

  // Audio playback
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioQueueRef = useRef<ArrayBuffer[]>([])
  const isPlayingRef = useRef(false)
  const currentUserTextRef = useRef('')
  const currentAssistantTextRef = useRef('')
  const currentAssistantAudioRef = useRef<ArrayBuffer[]>([])
  const hasConnectedRef = useRef(false) // Track if we ever successfully connected

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
      source.start()

      return new Promise<void>((resolve) => {
        source.onended = () => resolve()
      })
    } catch (err) {
      console.error('Failed to play audio:', err)
    }
  }, [getAudioContext])

  // Process audio queue
  const processAudioQueue = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return
    }

    isPlayingRef.current = true

    while (audioQueueRef.current.length > 0) {
      const audioData = audioQueueRef.current.shift()
      if (audioData) {
        await playAudioBuffer(audioData)
      }
    }

    isPlayingRef.current = false
  }, [playAudioBuffer])

  // Connect to voice chat WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setError(null)
    hasConnectedRef.current = false // Reset connection tracking

    const config: VoiceChatConfig = {
      llmModel,
      sttModel,
      ttsModel,
      ttsVoice,
      language,
      speed,
      systemPrompt,
      silenceDuration,
    }

    const ws = createVoiceChatConnection(namespace, project, config, {
      onSessionInfo: (id) => {
        hasConnectedRef.current = true // Mark as successfully connected
        setSessionId(id)
        setIsConnected(true)
        // Send VAD config after connection established
        if (silenceDuration !== undefined && ws.readyState === WebSocket.OPEN) {
          sendConfigUpdate(ws, { silence_duration: silenceDuration })
        }
      },
      onStateChange: (state) => {
        setVoiceState(state)
      },
      onTranscription: (text, isFinal) => {
        setCurrentTranscription(text)
        if (isFinal) {
          currentUserTextRef.current = text
        }
      },
      onLLMText: (text, isFinal) => {
        // Add user message on FIRST LLM text received (before any assistant content)
        // This ensures proper message ordering: user message appears before assistant response
        if (currentUserTextRef.current && currentAssistantTextRef.current === '') {
          const userMessage: VoiceMessage = {
            id: `user-${Date.now()}`,
            role: 'user',
            text: currentUserTextRef.current,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, userMessage])
          currentUserTextRef.current = ''
          setCurrentTranscription('')
        }

        setCurrentLLMText((prev) => prev + text)
        currentAssistantTextRef.current += text

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
        setError(message)
        onError?.(message)
      },
      onClose: () => {
        // If we never got connected (no session_id), this is a connection failure
        if (!hasConnectedRef.current) {
          const errorMsg = 'Failed to connect to voice chat server. Is the server running on port 8000?'
          setError(errorMsg)
          onError?.(errorMsg)
        }
        setIsConnected(false)
        setSessionId(null)
        setVoiceState('idle')
      },
    })

    wsRef.current = ws
  }, [namespace, project, llmModel, sttModel, ttsModel, ttsVoice, language, speed, systemPrompt, silenceDuration, onError, processAudioQueue])

  // Disconnect from voice chat
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setSessionId(null)
    setVoiceState('idle')

    // Stop any ongoing recording
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
    }
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop())
      setActiveStream(null)
    }
    setIsRecording(false)
  }, [activeStream])

  // Start recording
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
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })

      setActiveStream(stream)

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          sendAudioData(wsRef.current, e.data)
        }
      }

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        setActiveStream(null)
        setIsRecording(false)

        // Send end signal to trigger processing
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          sendEndSignal(wsRef.current)
        }
      }

      // Start recording with small timeslices for low latency
      mediaRecorder.start(100)
      setIsRecording(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start recording'
      setError(message)
      onError?.(message)
    }
  }, [isConnected, voiceState, onError])

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  // Send text message (bypasses STT)
  const sendTextMessage = useCallback((text: string) => {
    if (!isConnected || !wsRef.current) {
      setError('Not connected to voice chat')
      return
    }
    if (!text.trim()) {
      return
    }
    // Store the user text so it can be added to messages when LLM response is final
    // This ensures the user message is captured even if server transcription echo is delayed
    currentUserTextRef.current = text.trim()
    sendTextToWs(wsRef.current, text.trim())
  }, [isConnected])

  // Interrupt TTS (barge-in)
  const interrupt = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sendInterrupt(wsRef.current)
      // Clear audio queue
      audioQueueRef.current = []
    }
  }, [])

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
        silence_duration: config.silenceDuration,
      })
    }
  }, [])

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && namespace && project && llmModel) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, namespace, project, llmModel]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    isConnected,
    sessionId,
    voiceState,
    error,
    messages,
    currentTranscription,
    currentLLMText,
    isRecording,
    activeStream,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendTextMessage,
    interrupt,
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
