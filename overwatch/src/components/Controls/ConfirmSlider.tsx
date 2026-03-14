import { useState, useRef, useCallback } from 'react'

interface ConfirmSliderProps {
  label: string
  onConfirm: () => void
  onCancel: () => void
  variant?: 'danger' | 'warning'
}

export function ConfirmSlider({ label, onConfirm, onCancel, variant = 'danger' }: ConfirmSliderProps) {
  const [progress, setProgress] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const trackRef = useRef<HTMLDivElement>(null)

  const colors = {
    danger: {
      track: 'bg-[#1a0f0f]',
      fill: 'bg-kill',
      thumb: 'bg-kill-hover',
      text: 'text-text-secondary'
    },
    warning: {
      track: 'bg-[#1a1508]',
      fill: 'bg-status-warning',
      thumb: 'bg-status-warning',
      text: 'text-text-secondary'
    }
  }

  const style = colors[variant]

  const handleStart = useCallback((clientX: number) => {
    if (!trackRef.current) return
    setIsDragging(true)
    updateProgress(clientX)
  }, [])

  const handleMove = useCallback((clientX: number) => {
    if (!isDragging || !trackRef.current) return
    updateProgress(clientX)
  }, [isDragging])

  const handleEnd = useCallback(() => {
    if (progress >= 95) {
      onConfirm()
    } else {
      setProgress(0)
    }
    setIsDragging(false)
  }, [progress, onConfirm])

  const updateProgress = (clientX: number) => {
    if (!trackRef.current) return
    const rect = trackRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setProgress(percentage)
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-[4000]" onClick={onCancel}>
      <div className="bg-surface-raised rounded-2xl p-6 w-96 max-w-[90vw] border border-surface-border" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-xl font-bold text-text-primary mb-1">{label}</h2>
          <p className="text-text-dim text-sm">Slide to confirm</p>
        </div>

        {/* Slider track */}
        <div
          ref={trackRef}
          className={`relative h-14 ${style.track} rounded-full overflow-hidden cursor-pointer mb-4 border border-surface-border`}
          onMouseDown={(e) => handleStart(e.clientX)}
          onMouseMove={(e) => handleMove(e.clientX)}
          onMouseUp={handleEnd}
          onMouseLeave={() => isDragging && handleEnd()}
          onTouchStart={(e) => handleStart(e.touches[0].clientX)}
          onTouchMove={(e) => handleMove(e.touches[0].clientX)}
          onTouchEnd={handleEnd}
        >
          {/* Fill */}
          <div
            className={`absolute inset-y-0 left-0 ${style.fill} transition-all ${isDragging ? '' : 'duration-300'}`}
            style={{ width: `${progress}%` }}
          />

          {/* Label — centered in remaining space after thumb */}
          <div className={`absolute inset-0 flex items-center ${style.text} font-medium pointer-events-none`}
            style={{ paddingLeft: '60px' }}
          >
            <span className="mx-auto">
              {progress < 30 && 'Slide to confirm'}
              {progress >= 30 && progress < 95 && 'Keep sliding...'}
              {progress >= 95 && 'Release to confirm'}
            </span>
          </div>

          {/* Thumb */}
          <div
            className={`absolute top-1 bottom-1 w-12 ${style.thumb} rounded-full shadow-lg flex items-center justify-center transition-all ${isDragging ? '' : 'duration-300'}`}
            style={{ left: `max(4px, calc(${progress}% - 24px))` }}
          >
            <svg className="w-5 h-5 text-text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>

        {/* Cancel button */}
        <button
          onClick={onCancel}
          className="w-full py-3 text-text-dim hover:text-text-secondary hover:bg-surface-overlay rounded-lg transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
