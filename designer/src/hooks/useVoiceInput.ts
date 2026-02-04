/**
 * useVoiceInput - Lightweight hook for voice-to-text input in Text Generation mode
 *
 * This is a simplified version of useVoiceChat that only handles STT (speech-to-text).
 * It records audio, sends it to the transcription endpoint, and returns the transcribed text.
 *
 * Unlike useVoiceChat, this hook:
 * - Does NOT use WebSockets (uses REST API)
 * - Does NOT handle LLM responses or TTS
 * - Returns transcribed text to be inserted into a text input field
 * - Is designed for "tap mic, speak, get text" flow
 */

import { useState, useCallback, useRef } from 'react'
import { transcribeAudio } from '../api/voiceService'
import type { MicPermissionState, RecordingState } from '../types/ml'

export interface UseVoiceInputOptions {
  /** STT model to use (default: 'base') */
  sttModel?: string
  /** Language code (default: 'en') */
  language?: string
  /** Callback when transcription is complete */
  onTranscriptionComplete?: (text: string) => void
  /** Callback on error */
  onError?: (error: string) => void
}

export interface UseVoiceInputReturn {
  // State
  recordingState: RecordingState
  micPermission: MicPermissionState
  error: string | null
  activeStream: MediaStream | null

  // Actions
  startRecording: () => Promise<void>
  stopRecording: () => Promise<string | null>
  requestMicPermission: () => Promise<boolean>
  cancelRecording: () => void
}

export function useVoiceInput(options: UseVoiceInputOptions = {}): UseVoiceInputReturn {
  const {
    sttModel = 'base',
    language = 'en',
    onTranscriptionComplete,
    onError,
  } = options

  // State
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [micPermission, setMicPermission] = useState<MicPermissionState>('prompt')
  const [error, setError] = useState<string | null>(null)
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null)

  // Refs for recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  // Request mic permission
  const requestMicPermission = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      // Stop the test stream immediately
      stream.getTracks().forEach(track => track.stop())
      setMicPermission('granted')
      return true
    } catch (err) {
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setMicPermission('denied')
        } else {
          setMicPermission('error')
        }
      }
      return false
    }
  }, [])

  // Start recording
  const startRecording = useCallback(async () => {
    // Prevent starting if already recording - avoids leaking previous mic tracks
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      console.warn('Recording already in progress')
      return
    }

    setError(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      setMicPermission('granted')
      streamRef.current = stream
      setActiveStream(stream)
      audioChunksRef.current = []

      // Create MediaRecorder with webm/opus codec for better compression
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.start(100) // Collect data every 100ms
      setRecordingState('recording')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start recording'
      setError(message)
      onError?.(message)

      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setMicPermission('denied')
        } else {
          setMicPermission('error')
        }
      }
    }
  }, [onError])

  // Stop recording and transcribe
  const stopRecording = useCallback(async (): Promise<string | null> => {
    const mediaRecorder = mediaRecorderRef.current
    const stream = streamRef.current

    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      return null
    }

    setRecordingState('processing')

    return new Promise((resolve) => {
      mediaRecorder.onstop = async () => {
        // Stop the media stream
        if (stream) {
          stream.getTracks().forEach(track => track.stop())
          streamRef.current = null
          setActiveStream(null)
        }

        // Create audio blob from chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        audioChunksRef.current = []

        // Skip transcription if audio is too short (likely just noise)
        if (audioBlob.size < 1000) {
          setRecordingState('idle')
          resolve(null)
          return
        }

        try {
          // Transcribe the audio
          const result = await transcribeAudio(audioBlob, {
            model: sttModel,
            language,
            responseFormat: 'verbose_json',
          })

          const transcribedText = result.text?.trim() || ''
          setRecordingState('idle')

          if (transcribedText) {
            onTranscriptionComplete?.(transcribedText)
          }

          resolve(transcribedText)
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Transcription failed'
          setError(message)
          onError?.(message)
          setRecordingState('idle')
          resolve(null)
        }
      }

      mediaRecorder.stop()
    })
  }, [sttModel, language, onTranscriptionComplete, onError])

  // Cancel recording without transcribing
  const cancelRecording = useCallback(() => {
    const mediaRecorder = mediaRecorderRef.current
    const stream = streamRef.current

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop()
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setActiveStream(null)
    }

    audioChunksRef.current = []
    setRecordingState('idle')
  }, [])

  return {
    recordingState,
    micPermission,
    error,
    activeStream,
    startRecording,
    stopRecording,
    requestMicPermission,
    cancelRecording,
  }
}

export default useVoiceInput
