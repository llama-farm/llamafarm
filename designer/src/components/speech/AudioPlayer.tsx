import { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, Volume2, VolumeX } from 'lucide-react'
import { Button } from '../ui/button'

interface AudioPlayerProps {
  src?: string
  blob?: Blob
  compact?: boolean
  autoPlay?: boolean
  showWaveform?: boolean
  onEnded?: () => void
  onPlayingChange?: (isPlaying: boolean) => void
  className?: string
}

export function AudioPlayer({
  src,
  blob,
  compact = false,
  autoPlay = false,
  showWaveform = true,
  onEnded,
  onPlayingChange,
  className = '',
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const animationRef = useRef<number | null>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isMuted, setIsMuted] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | undefined>(src)
  const [waveformLevels, setWaveformLevels] = useState<number[]>(new Array(32).fill(0))

  // Handle blob conversion to URL
  useEffect(() => {
    if (blob) {
      const url = URL.createObjectURL(blob)
      setAudioUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setAudioUrl(src)
    }
  }, [blob, src])

  // Setup audio context and analyser when audio element is ready
  useEffect(() => {
    if (!audioRef.current || !showWaveform) return

    const setupAudioContext = () => {
      if (audioContextRef.current || sourceRef.current) return // Already set up

      try {
        const audioContext = new AudioContext()
        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 128
        analyser.smoothingTimeConstant = 0.7

        const source = audioContext.createMediaElementSource(audioRef.current!)
        source.connect(analyser)
        analyser.connect(audioContext.destination)

        audioContextRef.current = audioContext
        analyserRef.current = analyser
        sourceRef.current = source
      } catch (err) {
        // Audio element might already be connected
        console.warn('Could not create audio context:', err)
      }
    }

    // Set up on first play
    const handleFirstPlay = () => {
      setupAudioContext()
    }

    audioRef.current.addEventListener('play', handleFirstPlay, { once: true })

    return () => {
      audioRef.current?.removeEventListener('play', handleFirstPlay)
    }
  }, [showWaveform, audioUrl])

  // Cleanup audio context
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      // Note: Don't close audio context as it can't be reconnected
    }
  }, [])

  // Waveform animation
  useEffect(() => {
    if (!showWaveform) return

    const animate = () => {
      if (!analyserRef.current || !isPlaying) {
        // Fade out when not playing
        setWaveformLevels(prev => prev.map(level => Math.max(0, level * 0.85)))
        if (isPlaying) {
          animationRef.current = requestAnimationFrame(animate)
        }
        return
      }

      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
      analyserRef.current.getByteFrequencyData(dataArray)

      // Sample frequency data
      const levels: number[] = []
      const barCount = 32
      const step = Math.floor(dataArray.length / barCount)

      for (let i = 0; i < barCount; i++) {
        let sum = 0
        for (let j = 0; j < step; j++) {
          sum += dataArray[i * step + j]
        }
        levels.push(sum / step / 255)
      }

      setWaveformLevels(levels)
      animationRef.current = requestAnimationFrame(animate)
    }

    if (isPlaying) {
      animate()
    } else {
      // Fade out
      const fadeOut = () => {
        setWaveformLevels(prev => {
          const next = prev.map(level => Math.max(0, level * 0.85))
          if (next.some(l => l > 0.01)) {
            animationRef.current = requestAnimationFrame(fadeOut)
          }
          return next
        })
      }
      fadeOut()
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, showWaveform])

  // Auto-play handling
  useEffect(() => {
    if (autoPlay && audioRef.current && audioUrl) {
      audioRef.current.play().catch(() => {
        // Auto-play was blocked
      })
    }
  }, [autoPlay, audioUrl])

  const togglePlay = useCallback(() => {
    if (!audioRef.current) return

    // Resume audio context if suspended (browser autoplay policy)
    if (audioContextRef.current?.state === 'suspended') {
      audioContextRef.current.resume()
    }

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play().catch(() => {
        // Play was blocked
      })
    }
  }, [isPlaying])

  const toggleMute = useCallback(() => {
    if (!audioRef.current) return
    audioRef.current.muted = !isMuted
    setIsMuted(!isMuted)
  }, [isMuted])

  const handleTimeUpdate = useCallback(() => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }, [])

  const handleLoadedMetadata = useCallback(() => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }, [])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value)
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }, [])

  const handleEnded = useCallback(() => {
    setIsPlaying(false)
    setCurrentTime(0)
    onEnded?.()
  }, [onEnded])

  const handlePlayStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing)
    onPlayingChange?.(playing)
  }, [onPlayingChange])

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  if (!audioUrl) {
    return null
  }

  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <audio
          ref={audioRef}
          src={audioUrl}
          onPlay={() => handlePlayStateChange(true)}
          onPause={() => handlePlayStateChange(false)}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          crossOrigin="anonymous"
        />
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={togglePlay}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        {showWaveform && isPlaying && (
          <div className="flex items-center gap-[1px] h-4">
            {waveformLevels.slice(0, 12).map((level, i) => (
              <div
                key={i}
                className="w-0.5 rounded-full bg-primary transition-all duration-75"
                style={{ height: `${Math.max(2, level * 16)}px` }}
              />
            ))}
          </div>
        )}
        {duration > 0 && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        )}
      </div>
    )
  }

  return (
    <div className={`flex items-center gap-3 p-3 rounded-lg border border-border bg-muted/30 ${className}`}>
      <audio
        ref={audioRef}
        src={audioUrl}
        onPlay={() => handlePlayStateChange(true)}
        onPause={() => handlePlayStateChange(false)}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        crossOrigin="anonymous"
      />

      {/* Play/Pause button */}
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full flex-shrink-0"
        onClick={togglePlay}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
      </Button>

      {/* Waveform or progress bar */}
      <div className="flex-1 flex flex-col gap-1">
        {showWaveform ? (
          <div className="relative h-8">
            {/* Waveform bars */}
            <div className="absolute inset-0 flex items-center gap-[2px]">
              {waveformLevels.map((level, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-full transition-all duration-75"
                  style={{
                    height: `${Math.max(4, level * 32)}px`,
                    backgroundColor: isPlaying ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))',
                    opacity: isPlaying ? 0.6 + level * 0.4 : 0.3,
                  }}
                />
              ))}
            </div>
            {/* Clickable progress overlay */}
            <input
              type="range"
              min={0}
              max={duration || 100}
              value={currentTime}
              onChange={handleSeek}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              aria-label="Audio progress"
            />
          </div>
        ) : (
          <input
            type="range"
            min={0}
            max={duration || 100}
            value={currentTime}
            onChange={handleSeek}
            className="w-full h-1.5 bg-muted rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-primary
              [&::-webkit-slider-thumb]:cursor-pointer"
            aria-label="Audio progress"
          />
        )}
        <div className="flex justify-between text-xs text-muted-foreground tabular-nums">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Mute button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 flex-shrink-0"
        onClick={toggleMute}
        aria-label={isMuted ? 'Unmute' : 'Mute'}
      >
        {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
      </Button>
    </div>
  )
}
