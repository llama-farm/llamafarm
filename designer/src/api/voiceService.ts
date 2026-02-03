/**
 * Voice Service - API client for speech-to-text and text-to-speech endpoints
 *
 * Endpoints:
 * - REST: Universal Runtime for TTS and STT (via runtimeClient)
 * - WebSocket: Server voice chat for full-duplex conversation
 */

import { runtimeClient } from './client'
import { devToolsEmitter } from '../utils/devToolsEmitter'

// Server URL for voice WebSocket (goes through API gateway)
// Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
const API_HOST = (import.meta.env as Record<string, string>).VITE_APP_API_URL || 'http://127.0.0.1:14345'

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

// New message types from feat-audio-processing-2
export interface VoiceEmotion {
  type: 'emotion'
  emotion: string
  confidence: number
  all_scores: Record<string, number>
}

export interface VoiceToolCall {
  type: 'tool_call'
  tool_call_id: string
  function_name: string
  arguments: string
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
  | VoiceEmotion
  | VoiceToolCall

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
  // Turn detection settings (sent via config message after connect)
  turnDetectionEnabled?: boolean
  baseSilenceDuration?: number    // For complete utterances (0.1-2.0s, default 0.4)
  thinkingSilenceDuration?: number // For incomplete utterances (0.3-5.0s, default 1.2)
  maxSilenceDuration?: number     // Hard timeout (0.5-10.0s, default 2.5)
  // Barge-in settings
  bargeInEnabled?: boolean
  bargeInNoiseFilter?: boolean
  bargeInMinChunks?: number
  // Emotion detection settings
  emotionDetectionEnabled?: boolean
  emotionModel?: string
  emotionConfidenceThreshold?: number
}

// =============================================================================
// REST API - Text-to-Speech
// =============================================================================

/**
 * List available TTS voices
 */
export async function listVoices(model?: string): Promise<VoiceInfo[]> {
  const response = await runtimeClient.get<VoiceListResponse>('/v1/audio/voices', {
    params: model ? { model } : undefined,
  })
  return response.data.data
}

/**
 * Synthesize speech from text (non-streaming)
 */
export async function synthesizeSpeech(request: SpeechRequest): Promise<Blob> {
  const response = await runtimeClient.post<Blob>(
    '/v1/audio/speech',
    {
      model: request.model || 'kokoro',
      input: request.input,
      voice: request.voice || 'af_heart',
      response_format: request.response_format || 'mp3',
      speed: request.speed || 1.0,
      stream: false,
    },
    { responseType: 'blob' }
  )
  return response.data
}

// =============================================================================
// Health Check
// =============================================================================

/**
 * Check if Universal Runtime is available
 */
export async function checkRuntimeHealth(): Promise<boolean> {
  try {
    const response = await runtimeClient.get('/health', { timeout: 3000 })
    return response.status === 200
  } catch {
    return false
  }
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

  const response = await runtimeClient.post<TranscriptionResponse>(
    '/v1/audio/transcriptions',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  )
  return response.data
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
  // New callbacks for emotion and tool calls
  onEmotion?: (emotion: string, confidence: number, allScores: Record<string, number>) => void
  onToolCall?: (toolCallId: string, functionName: string, args: string) => void
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
  // Only set llm_model if it's not empty (backend requires this)
  if (config.llmModel) {
    url.searchParams.set('llm_model', config.llmModel)
  }
  if (config.language) url.searchParams.set('language', config.language)
  if (config.speed !== undefined) url.searchParams.set('speed', String(config.speed))
  if (config.systemPrompt) url.searchParams.set('system_prompt', config.systemPrompt)
  if (config.sentenceBoundaryOnly !== undefined) {
    url.searchParams.set('sentence_boundary_only', String(config.sentenceBoundaryOnly))
  }
  // Note: Turn detection settings are sent via config message after connect, not query params

  // Generate a unique connection ID for DevTools tracking
  const connectionId = crypto.randomUUID
    ? crypto.randomUUID()
    : `ws-${Date.now()}-${Math.random().toString(36).slice(2)}`

  const ws = new WebSocket(url.toString())
  ws.binaryType = 'arraybuffer'

  // Emit WebSocket open event to DevTools
  ws.onopen = () => {
    if (devToolsEmitter.hasSubscribers()) {
      devToolsEmitter.emit({
        type: 'ws_open',
        id: connectionId,
        url: url.toString(),
      })
    }
  }

  ws.onmessage = (event) => {
    // Handle binary audio data
    if (event.data instanceof ArrayBuffer) {
      // Emit binary message to DevTools (just indicate size, not full data)
      if (devToolsEmitter.hasSubscribers()) {
        devToolsEmitter.emit({
          type: 'ws_message',
          connectionId,
          direction: 'receive',
          data: `[Audio: ${event.data.byteLength} bytes]`,
          isBinary: true,
          size: event.data.byteLength,
        })
      }
      callbacks.onAudio?.(event.data)
      return
    }

    // Handle JSON messages
    try {
      const message: VoiceMessage = JSON.parse(event.data)

      // Emit JSON message to DevTools
      if (devToolsEmitter.hasSubscribers()) {
        devToolsEmitter.emit({
          type: 'ws_message',
          connectionId,
          direction: 'receive',
          data: message,
          isBinary: false,
          size: event.data.length,
        })
      }

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
        case 'emotion':
          callbacks.onEmotion?.(message.emotion, message.confidence, message.all_scores)
          break
        case 'tool_call':
          callbacks.onToolCall?.(message.tool_call_id, message.function_name, message.arguments)
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
      // Propagate parse errors to UI so user knows something went wrong
      callbacks.onError?.(`Failed to parse server message: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  ws.onerror = (event) => {
    console.error('Voice WebSocket error:', event)
    // Emit close with error to DevTools
    if (devToolsEmitter.hasSubscribers()) {
      devToolsEmitter.emit({
        type: 'ws_close',
        id: connectionId,
        error: 'WebSocket connection error',
      })
    }
    callbacks.onError?.('WebSocket connection error')
  }

  ws.onclose = () => {
    // Emit close to DevTools
    if (devToolsEmitter.hasSubscribers()) {
      devToolsEmitter.emit({
        type: 'ws_close',
        id: connectionId,
      })
    }
    callbacks.onClose?.()
  }

  // Wrap the send method to capture outgoing messages
  const originalSend = ws.send.bind(ws)
  ws.send = (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    if (devToolsEmitter.hasSubscribers()) {
      if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
        const size = data instanceof ArrayBuffer ? data.byteLength : data.byteLength
        devToolsEmitter.emit({
          type: 'ws_message',
          connectionId,
          direction: 'send',
          data: `[Audio: ${size} bytes]`,
          isBinary: true,
          size,
        })
      } else if (typeof data === 'string') {
        try {
          devToolsEmitter.emit({
            type: 'ws_message',
            connectionId,
            direction: 'send',
            data: JSON.parse(data),
            isBinary: false,
            size: data.length,
          })
        } catch {
          devToolsEmitter.emit({
            type: 'ws_message',
            connectionId,
            direction: 'send',
            data,
            isBinary: false,
            size: data.length,
          })
        }
      }
    }
    return originalSend(data)
  }

  return ws
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
    // Turn detection settings
    turn_detection_enabled: boolean
    base_silence_duration: number
    thinking_silence_duration: number
    max_silence_duration: number
    // Barge-in settings
    barge_in_enabled: boolean
    barge_in_noise_filter: boolean
    barge_in_min_chunks: number
    // Emotion detection settings
    emotion_detection_enabled: boolean
    emotion_model: string
    emotion_confidence_threshold: number
  }>
): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', ...config }))
  }
}
