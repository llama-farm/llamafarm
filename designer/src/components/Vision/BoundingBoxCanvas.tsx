import { useEffect, useRef } from 'react'
import type { Detection } from '../../types/vision'

interface BoundingBoxCanvasProps {
  imageSrc: string
  detections: Detection[]
  className?: string
}

const COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7',
]

export function BoundingBoxCanvas({ imageSrc, detections, className }: BoundingBoxCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const prevSrcRef = useRef<string | null>(null)
  const detectionsRef = useRef<Detection[]>(detections)
  detectionsRef.current = detections

  // Load image only when imageSrc changes
  useEffect(() => {
    if (imageSrc === prevSrcRef.current && imgRef.current) return

    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      prevSrcRef.current = imageSrc
      // Use ref to get latest detections, avoiding stale closure
      drawDetections(img, detectionsRef.current)
    }
    img.src = imageSrc
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSrc])

  // Redraw detections when they change (without reloading image)
  useEffect(() => {
    if (imgRef.current) {
      drawDetections(imgRef.current, detections)
    }
  }, [detections])

  function drawDetections(img: HTMLImageElement, dets: Detection[]) {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0)

    dets.forEach((det, i) => {
      const color = COLORS[i % COLORS.length]
      const { x1, y1, x2, y2 } = det.box
      const w = x2 - x1
      const h = y2 - y1

      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(x1, y1, w, h)

      // Label background
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`
      ctx.font = 'bold 14px sans-serif'
      const textWidth = ctx.measureText(label).width
      ctx.fillStyle = color
      ctx.fillRect(x1, y1 - 22, textWidth + 8, 22)

      // Label text
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, x1 + 4, y1 - 6)
    })
  }

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ maxWidth: '100%', height: 'auto' }}
    />
  )
}
