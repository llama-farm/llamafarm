import { useEffect, useRef, useState, useCallback } from 'react'

interface WaveformProps {
  /** Audio stream for live recording visualization */
  stream?: MediaStream | null
  /** Audio element for playback visualization */
  audioElement?: HTMLAudioElement | null
  /** Whether to animate (for recording or playback) */
  isActive?: boolean
  /** Height of the waveform in pixels */
  height?: number
  /** Number of bars to display */
  barCount?: number
  /** Color of the bars */
  color?: string
  /** Gap between bars in pixels */
  gap?: number
  className?: string
}

export function Waveform({
  stream,
  audioElement,
  isActive = false,
  height = 48,
  barCount = 32,
  color = 'currentColor',
  gap = 2,
  className = '',
}: WaveformProps) {
  const animationRef = useRef<number | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const [levels, setLevels] = useState<number[]>(new Array(barCount).fill(0))

  // Set up audio analyser for stream (recording)
  useEffect(() => {
    if (!stream || !isActive) {
      analyserRef.current = null
      return
    }

    try {
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.7
      source.connect(analyser)
      analyserRef.current = analyser
    } catch (err) {
      console.error('Failed to create audio analyser:', err)
    }

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [stream, isActive])

  // Set up audio analyser for audio element (playback)
  useEffect(() => {
    if (!audioElement || !isActive) {
      return
    }

    try {
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaElementSource(audioElement)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.7
      source.connect(analyser)
      analyser.connect(audioContext.destination) // Connect to output so audio plays
      analyserRef.current = analyser
    } catch (err) {
      // Audio element might already be connected to a context
      console.error('Failed to create audio analyser for playback:', err)
    }

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [audioElement, isActive])

  // Animation loop
  const animate = useCallback(() => {
    if (!analyserRef.current || !isActive) {
      // Reset to flat when not active
      setLevels(new Array(barCount).fill(0))
      return
    }

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)

    // Sample frequency data into bar count
    const newLevels: number[] = []
    // Guard against barCount exceeding array length to prevent step=0 and NaN levels
    const effectiveBarCount = Math.min(barCount, dataArray.length)
    const step = Math.max(1, Math.floor(dataArray.length / effectiveBarCount))

    for (let i = 0; i < effectiveBarCount; i++) {
      // Average a range of frequencies for each bar
      let sum = 0
      const endIdx = Math.min(i * step + step, dataArray.length)
      for (let j = i * step; j < endIdx; j++) {
        sum += dataArray[j]
      }
      const avg = sum / step / 255 // Normalize to 0-1
      newLevels.push(avg)
    }
    // Pad with zeros if barCount > effectiveBarCount
    while (newLevels.length < barCount) {
      newLevels.push(0)
    }

    setLevels(newLevels)
    animationRef.current = requestAnimationFrame(animate)
  }, [isActive, barCount])

  // Start/stop animation
  useEffect(() => {
    if (isActive) {
      animate()
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      // Animate back to zero
      setLevels(new Array(barCount).fill(0))
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isActive, animate, barCount])

  // Calculate bar width based on container
  const barWidth = `calc((100% - ${(barCount - 1) * gap}px) / ${barCount})`

  return (
    <div
      className={`flex items-center justify-center ${className}`}
      style={{ height }}
      role="img"
      aria-label={isActive ? 'Audio waveform visualization' : 'Audio waveform idle'}
    >
      <div className="flex items-center gap-[2px] h-full w-full">
        {levels.map((level, i) => {
          // Minimum height of 4px, scale up based on level
          const barHeight = Math.max(4, level * height * 0.9)

          return (
            <div
              key={i}
              className="rounded-full transition-all duration-75"
              style={{
                width: barWidth,
                height: barHeight,
                backgroundColor: color,
                opacity: isActive ? 0.6 + level * 0.4 : 0.3,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}

/**
 * Simpler waveform that uses pre-computed levels (for when we don't have direct audio access)
 */
interface SimpleWaveformProps {
  levels: number[]
  isActive?: boolean
  height?: number
  color?: string
  className?: string
}

export function SimpleWaveform({
  levels,
  isActive = false,
  height = 48,
  color = 'currentColor',
  className = '',
}: SimpleWaveformProps) {
  return (
    <div
      className={`flex items-center justify-center ${className}`}
      style={{ height }}
      role="img"
      aria-label={isActive ? 'Audio waveform visualization' : 'Audio waveform idle'}
    >
      <div className="flex items-center gap-[2px] h-full w-full">
        {levels.map((level, i) => {
          const barHeight = Math.max(4, level * height * 0.9)

          return (
            <div
              key={i}
              className="flex-1 rounded-full transition-all duration-75"
              style={{
                height: barHeight,
                backgroundColor: color,
                opacity: isActive ? 0.6 + level * 0.4 : 0.3,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}
