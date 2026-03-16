import { useEffect, useRef, useState } from 'react'

interface AltitudeSparklineProps {
  altitude: number
  maxPoints?: number
}

export function AltitudeSparkline({ altitude, maxPoints = 40 }: AltitudeSparklineProps) {
  const [history, setHistory] = useState<number[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Accumulate altitude history
  useEffect(() => {
    setHistory(prev => {
      const next = [...prev, altitude]
      return next.length > maxPoints ? next.slice(-maxPoints) : next
    })
  }, [altitude, maxPoints])

  // Draw sparkline
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || history.length < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Fixed logical size — DPR handled once
    const w = 120
    const h = 32
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== w * dpr) {
      canvas.width = w * dpr
      canvas.height = h * dpr
      ctx.scale(dpr, dpr)
    }
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`

    ctx.clearRect(0, 0, w, h)

    const min = Math.min(...history) - 5
    const max = Math.max(...history) + 5
    const range = max - min || 1

    // Fill gradient
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, 'rgba(61, 124, 199, 0.3)')
    grad.addColorStop(1, 'rgba(61, 124, 199, 0)')

    ctx.beginPath()
    ctx.moveTo(0, h)
    for (let i = 0; i < history.length; i++) {
      const x = (i / (maxPoints - 1)) * w
      const y = h - ((history[i] - min) / range) * (h - 4)
      if (i === 0) ctx.lineTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.lineTo(((history.length - 1) / (maxPoints - 1)) * w, h)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    // Line
    ctx.beginPath()
    for (let i = 0; i < history.length; i++) {
      const x = (i / (maxPoints - 1)) * w
      const y = h - ((history[i] - min) / range) * (h - 4)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.strokeStyle = '#3d7cc7'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Current value dot
    const lastX = ((history.length - 1) / (maxPoints - 1)) * w
    const lastY = h - ((history[history.length - 1] - min) / range) * (h - 4)
    ctx.beginPath()
    ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2)
    ctx.fillStyle = '#3d7cc7'
    ctx.fill()
  }, [history, maxPoints])

  return (
    <div className="bg-black/70 backdrop-blur-sm border border-surface-border rounded-lg px-2 py-1.5 pointer-events-none">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[9px] text-text-dim font-mono tracking-wider">ALT</span>
        <span className="text-[10px] text-accent font-mono font-bold">{Math.round(altitude)}m</span>
      </div>
      <canvas
        ref={canvasRef}
        width={120}
        height={32}
        className="block"
      />
    </div>
  )
}
