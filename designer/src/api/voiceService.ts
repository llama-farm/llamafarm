/**
 * Voice Service - API client for speech-to-text and text-to-speech endpoints
 *
 * Endpoints:
 * - REST: Universal Runtime for TTS and STT
 * - WebSocket: Server voice chat for full-duplex conversation
 */

// Universal Runtime URL for direct TTS/STT calls
const UNIVERSAL_RUNTIME_URL =
  import.meta.env.VITE_UNIVERSAL_RUNTIME_URL || 'http://localhost:11540'

// Server URL for voice WebSocket (goes through API gateway)
const API_HOST = (import.meta.env as Record<string, string>).VITE_APP_API_URL || 'http://localhost:8000'

// =============================================================================
// Types
// =============================================================================

export interface VoiceInfo {
  id: string
  name: string
  language: string
  model: string
  preview_url: string | null
}

export interface VoiceListResponse {
  object: 'list'
  data: VoiceInfo[]
}

export interface SpeechRequest {
  model?: string
  input: string
  voice?: string
  response_format?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm'
  speed?: number
  stream?: boolean
}

export interface TranscriptionSegment {
  id: number
  start: number
  end: number
  text: string
  words?: Array<{
    word: string
    start: number
    end: number
    probability: number
  }>
  avg_logprob?: number
  no_speech_prob?: number
}

export interface TranscriptionResponse {
  text: string
  segments?: TranscriptionSegment[]
  language?: string
  language_probability?: number
  duration?: number
}

// Voice WebSocket message types
export type VoiceState = 'idle' | 'listening' | 'processing' | 'speaking' | 'interrupted'

export interface VoiceSessionInfo {
  type: 'session_info'
  session_id: string
}

export interface VoiceStatus {
  type: 'status'
  state: VoiceState
}

export interface VoiceTranscription {
  type: 'transcription'
  text: string
  is_final: boolean
}

export interface VoiceLLMText {
  type: 'llm_text'
  text: string
  is_final: boolean
}

export interface VoiceTTSStart {
  type: 'tts_start'
  phrase_index: number
}

export interface VoiceTTSDone {
  type: 'tts_done'
  phrase_index: number
  duration: number
}

export interface VoiceError {
  type: 'error'
  message: string
}

export interface VoiceClosed {
  type: 'closed'
}

export type VoiceMessage =
  | VoiceSessionInfo
  | VoiceStatus
  | VoiceTranscription
  | VoiceLLMText
  | VoiceTTSStart
  | VoiceTTSDone
  | VoiceError
  | VoiceClosed

export interface VoiceChatConfig {
  sessionId?: string
  sttModel?: string
  ttsModel?: string
  ttsVoice?: string
  llmModel: string
  language?: string
  speed?: number
  systemPrompt?: string
  sentenceBoundaryOnly?: boolean
}

// =============================================================================
// REST API - Text-to-Speech
// =============================================================================

/**
 * List available TTS voices
 */
export async function listVoices(model?: string): Promise<VoiceInfo[]> {
  const url = new URL(`${UNIVERSAL_RUNTIME_URL}/v1/audio/voices`)
  if (model) {
    url.searchParams.set('model', model)
  }

  const response = await fetch(url.toString())
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  const data: VoiceListResponse = await response.json()
  return data.data
}

/**
 * Synthesize speech from text (non-streaming)
 */
export async function synthesizeSpeech(request: SpeechRequest): Promise<Blob> {
  const response = await fetch(`${UNIVERSAL_RUNTIME_URL}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: request.model || 'kokoro',
      input: request.input,
      voice: request.voice || 'af_heart',
      response_format: request.response_format || 'mp3',
      speed: request.speed || 1.0,
      stream: false,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.blob()
}

// =============================================================================
// REST API - Speech-to-Text
// =============================================================================

/**
 * Transcribe audio file
 */
export async function transcribeAudio(
  audioBlob: Blob,
  options: {
    model?: string
    language?: string
    prompt?: string
    responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt'
    temperature?: number
  } = {}
): Promise<TranscriptionResponse> {
  const formData = new FormData()
  formData.append('file', audioBlob, 'audio.webm')
  formData.append('model', options.model || 'distil-large-v3-turbo')
  if (options.language) {
    formData.append('language', options.language)
  }
  if (options.prompt) {
    formData.append('prompt', options.prompt)
  }
  formData.append('response_format', options.responseFormat || 'verbose_json')
  formData.append('temperature', String(options.temperature || 0))

  const response = await fetch(`${UNIVERSAL_RUNTIME_URL}/v1/audio/transcriptions`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// =============================================================================
// WebSocket - Voice Chat Session
// =============================================================================

export interface VoiceChatCallbacks {
  onSessionInfo?: (sessionId: string) => void
  onStateChange?: (state: VoiceState) => void
  onTranscription?: (text: string, isFinal: boolean) => void
  onLLMText?: (text: string, isFinal: boolean) => void
  onTTSStart?: (phraseIndex: number) => void
  onTTSDone?: (phraseIndex: number, duration: number) => void
  onAudio?: (audioData: ArrayBuffer) => void
  onError?: (message: string) => void
  onClose?: () => void
}

/**
 * Create a WebSocket connection to the voice chat endpoint
 */
export function createVoiceChatConnection(
  namespace: string,
  project: string,
  config: VoiceChatConfig,
  callbacks: VoiceChatCallbacks
): WebSocket {
  // Build WebSocket URL with query params
  const wsProtocol = API_HOST.startsWith('https') ? 'wss' : 'ws'
  const wsHost = API_HOST.replace(/^https?:\/\//, '')
  const url = new URL(`${wsProtocol}://${wsHost}/v1/${namespace}/${project}/voice/chat`)

  // Add query parameters
  if (config.sessionId) url.searchParams.set('session_id', config.sessionId)
  if (config.sttModel) url.searchParams.set('stt_model', config.sttModel)
  if (config.ttsModel) url.searchParams.set('tts_model', config.ttsModel)
  if (config.ttsVoice) url.searchParams.set('tts_voice', config.ttsVoice)
  url.searchParams.set('llm_model', config.llmModel)
  if (config.language) url.searchParams.set('language', config.language)
  if (config.speed !== undefined) url.searchParams.set('speed', String(config.speed))
  if (config.systemPrompt) url.searchParams.set('system_prompt', config.systemPrompt)
  if (config.sentenceBoundaryOnly !== undefined) {
    url.searchParams.set('sentence_boundary_only', String(config.sentenceBoundaryOnly))
  }

  const ws = new WebSocket(url.toString())
  ws.binaryType = 'arraybuffer'

  ws.onmessage = (event) => {
    // Handle binary audio data
    if (event.data instanceof ArrayBuffer) {
      callbacks.onAudio?.(event.data)
      return
    }

    // Handle JSON messages
    try {
      const message: VoiceMessage = JSON.parse(event.data)

      switch (message.type) {
        case 'session_info':
          callbacks.onSessionInfo?.(message.session_id)
          break
        case 'status':
          callbacks.onStateChange?.(message.state)
          break
        case 'transcription':
          callbacks.onTranscription?.(message.text, message.is_final)
          break
        case 'llm_text':
          callbacks.onLLMText?.(message.text, message.is_final)
          break
        case 'tts_start':
          callbacks.onTTSStart?.(message.phrase_index)
          break
        case 'tts_done':
          callbacks.onTTSDone?.(message.phrase_index, message.duration)
          break
        case 'error':
          callbacks.onError?.(message.message)
          break
        case 'closed':
          callbacks.onClose?.()
          break
      }
    } catch (e) {
      console.error('Failed to parse voice message:', e)
    }
  }

  ws.onerror = (event) => {
    console.error('Voice WebSocket error:', event)
    callbacks.onError?.('WebSocket connection error')
  }

  ws.onclose = () => {
    callbacks.onClose?.()
  }

  return ws
}

/**
 * Send audio data to voice chat WebSocket
 */
export function sendAudioData(ws: WebSocket, audioData: ArrayBuffer | Blob): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(audioData)
  }
}

/**
 * Send interrupt signal to stop TTS (barge-in)
 */
export function sendInterrupt(ws: WebSocket): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'interrupt' }))
  }
}

/**
 * Send end signal to force processing
 */
export function sendEndSignal(ws: WebSocket): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'end' }))
  }
}

/**
 * Update session configuration
 */
export function sendConfigUpdate(
  ws: WebSocket,
  config: Partial<{
    stt_model: string
    tts_model: string
    tts_voice: string
    llm_model: string
    language: string
    speed: number
    sentence_boundary_only: boolean
  }>
): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', ...config }))
  }
}

// =============================================================================
// WebSocket - Streaming Transcription (standalone)
// =============================================================================

export interface StreamingTranscriptionCallbacks {
  onSegment?: (segment: {
    id: number
    start: number
    end: number
    text: string
    isFinal: boolean
  }) => void
  onError?: (message: string) => void
  onClose?: () => void
}

/**
 * Create a WebSocket connection for streaming transcription
 */
export function createStreamingTranscription(
  options: {
    model?: string
    language?: string
    wordTimestamps?: boolean
    chunkInterval?: number
  } = {},
  callbacks: StreamingTranscriptionCallbacks
): WebSocket {
  const url = new URL(`${UNIVERSAL_RUNTIME_URL}/v1/audio/transcriptions/stream`)

  if (options.model) url.searchParams.set('model', options.model)
  if (options.language) url.searchParams.set('language', options.language)
  if (options.wordTimestamps) url.searchParams.set('word_timestamps', 'true')
  if (options.chunkInterval) url.searchParams.set('chunk_interval', String(options.chunkInterval))

  const ws = new WebSocket(url.toString())
  ws.binaryType = 'arraybuffer'

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)

      if (data.type === 'segment') {
        callbacks.onSegment?.({
          id: data.id,
          start: data.start,
          end: data.end,
          text: data.text,
          isFinal: data.is_final,
        })
      } else if (data.type === 'error') {
        callbacks.onError?.(data.message)
      } else if (data.type === 'closed' || data.type === 'done') {
        callbacks.onClose?.()
      }
    } catch (e) {
      console.error('Failed to parse transcription message:', e)
    }
  }

  ws.onerror = () => {
    callbacks.onError?.('WebSocket connection error')
  }

  ws.onclose = () => {
    callbacks.onClose?.()
  }

  return ws
}

// =============================================================================
// Default Export
// =============================================================================

export default {
  // TTS
  listVoices,
  synthesizeSpeech,
  // STT
  transcribeAudio,
  // Voice Chat
  createVoiceChatConnection,
  sendAudioData,
  sendInterrupt,
  sendEndSignal,
  sendConfigUpdate,
  // Streaming Transcription
  createStreamingTranscription,
}
