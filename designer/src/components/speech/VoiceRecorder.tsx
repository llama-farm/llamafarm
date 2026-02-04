import { useState, useRef, useCallback, useEffect } from 'react'
import { Mic, Square, Trash2, Save, Play, Pause } from 'lucide-react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import type { RecordingState } from '../../types/ml'

interface VoiceRecorderProps {
  onSave: (name: string, audioBlob: Blob, duration: number) => void
  onCancel: () => void
  sampleText?: string
  className?: string
}

export function VoiceRecorder({
  onSave,
  onCancel,
  sampleText = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
  className = '',
}: VoiceRecorderProps) {
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [voiceName, setVoiceName] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [waveformLevels, setWaveformLevels] = useState<number[]>(new Array(20).fill(0))

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const animationRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (recordedUrl) URL.revokeObjectURL(recordedUrl)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      // Close AudioContext to prevent resource leak
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [recordedUrl])

  const updateWaveform = useCallback(() => {
    if (!analyserRef.current || recordingState !== 'recording') return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)

    // Sample 20 points from the frequency data
    const levels: number[] = []
    const step = Math.floor(dataArray.length / 20)
    for (let i = 0; i < 20; i++) {
      levels.push(dataArray[i * step] / 255)
    }
    setWaveformLevels(levels)

    animationRef.current = requestAnimationFrame(updateWaveform)
  }, [recordingState])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up audio analyser for waveform
      // Close any existing context to prevent leaks
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // Set up media recorder
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setRecordedBlob(blob)
        const url = URL.createObjectURL(blob)
        setRecordedUrl(url)
        setRecordingState('idle')

        // Clean up
        stream.getTracks().forEach(track => track.stop())
        if (animationRef.current) cancelAnimationFrame(animationRef.current)
        setWaveformLevels(new Array(20).fill(0))
      }

      mediaRecorder.start()
      setRecordingState('recording')
      setRecordingTime(0)

      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime(t => t + 0.1)
      }, 100)

      // Start waveform animation
      updateWaveform()
    } catch (err) {
      console.error('Failed to start recording:', err)
      setRecordingState('error')
    }
  }, [updateWaveform])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const handleDiscard = useCallback(() => {
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
    }
    setRecordedBlob(null)
    setRecordedUrl(null)
    setRecordingTime(0)
    setVoiceName('')
  }, [recordedUrl])

  const handleSave = useCallback(() => {
    if (!recordedBlob || !voiceName.trim()) return
    onSave(voiceName.trim(), recordedBlob, recordingTime)
  }, [recordedBlob, voiceName, recordingTime, onSave])

  const togglePlayback = useCallback(() => {
    if (!audioRef.current || !recordedUrl) return

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }, [isPlaying, recordedUrl])

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    const tenths = Math.floor((time % 1) * 10)
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${tenths}`
  }

  // After recording - review and save
  if (recordedBlob && recordedUrl) {
    return (
      <div className={`rounded-xl border border-border bg-card p-4 ${className}`}>
        <h4 className="text-sm font-medium mb-3">Review Recording</h4>

        {/* Audio playback */}
        <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-muted/30">
          <audio
            ref={audioRef}
            src={recordedUrl}
            onEnded={() => setIsPlaying(false)}
          />
          <Button
            variant="outline"
            size="icon"
            className="h-10 w-10 rounded-full"
            onClick={togglePlayback}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </Button>
          <span className="text-sm text-muted-foreground tabular-nums">
            {formatTime(recordingTime)}
          </span>
        </div>

        {/* Name input */}
        <div className="mb-4">
          <label className="text-xs text-muted-foreground mb-1 block">
            Voice Name
          </label>
          <Input
            value={voiceName}
            onChange={(e) => setVoiceName(e.target.value)}
            placeholder="My Voice"
            className="w-full"
          />
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            className="flex-1"
            onClick={handleDiscard}
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Discard
          </Button>
          <Button
            className="flex-1"
            onClick={handleSave}
            disabled={!voiceName.trim()}
          >
            <Save className="w-4 h-4 mr-2" />
            Save Voice
          </Button>
        </div>
      </div>
    )
  }

  // Recording or idle state
  return (
    <div className={`rounded-xl border border-border bg-card p-4 ${className}`}>
      <h4 className="text-sm font-medium mb-3">Record Voice Sample</h4>

      {/* Sample text to read */}
      <div className="mb-4 p-3 rounded-lg bg-muted/30 text-sm text-muted-foreground">
        <div className="text-xs font-medium mb-1">Read this text aloud:</div>
        <div className="italic">{sampleText}</div>
      </div>

      {/* Recording area */}
      <div className="flex flex-col items-center gap-4 py-6">
        {recordingState === 'recording' ? (
          <>
            {/* Recording indicator */}
            <div className="flex items-center gap-2 text-red-500">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-sm font-medium">Recording...</span>
            </div>

            {/* Waveform visualization */}
            <div className="flex items-center gap-0.5 h-12">
              {waveformLevels.map((level, i) => (
                <div
                  key={i}
                  className="w-1.5 bg-red-500/80 rounded-full transition-all duration-75"
                  style={{ height: `${Math.max(4, level * 48)}px` }}
                />
              ))}
            </div>

            {/* Timer */}
            <div className="text-2xl font-mono tabular-nums">
              {formatTime(recordingTime)}
            </div>

            {/* Stop button */}
            <Button
              variant="destructive"
              size="lg"
              onClick={stopRecording}
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Recording
            </Button>
          </>
        ) : (
          <>
            {/* Microphone icon */}
            <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted">
              <Mic className="w-8 h-8 text-muted-foreground" />
            </div>

            <p className="text-sm text-muted-foreground">
              Press to start recording
            </p>

            {/* Start button */}
            <Button
              size="lg"
              onClick={startRecording}
            >
              <Mic className="w-4 h-4 mr-2" />
              Start Recording
            </Button>
          </>
        )}
      </div>

      {/* Cancel button */}
      <div className="flex justify-center mt-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onCancel}
          disabled={recordingState === 'recording'}
        >
          Cancel
        </Button>
      </div>
    </div>
  )
}
