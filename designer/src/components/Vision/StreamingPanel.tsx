import { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '../ui/button'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { cn } from '@/lib/utils'
import { useStreamStart, useStreamFrame, useStreamStop, useStreamSessions } from '../../hooks/useVision'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import type { Detection } from '../../types/vision'
import { useToast } from '../ui/toast'

export function StreamingPanel() {
  const [targetFps, setTargetFps] = useState(5)
  const [confidence, setConfidence] = useState(0.5)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [frameDataUrl, setFrameDataUrl] = useState<string | null>(null)
  const [framesProcessed, setFramesProcessed] = useState(0)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const startMutation = useStreamStart()
  const frameMutation = useStreamFrame()
  const stopMutation = useStreamStop()
  const { toast } = useToast()
  useStreamSessions({ enabled: isStreaming })

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    ctx.drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
    setFrameDataUrl(dataUrl)
    return dataUrl.split(',')[1]
  }, [])

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      const result = await startMutation.mutateAsync({
        config: { confidence_threshold: confidence },
        target_fps: targetFps,
      })

      setSessionId(result.session_id)
      setIsStreaming(true)

      // Start sending frames
      intervalRef.current = setInterval(() => {
        const b64 = captureFrame()
        if (b64 && result.session_id) {
          frameMutation.mutate(
            { session_id: result.session_id, image: b64 },
            {
              onSuccess: data => {
                setDetections(data.detections)
                setFramesProcessed(prev => prev + 1)
              },
            }
          )
        }
      }, 1000 / targetFps)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start camera'
      toast({ message, variant: 'destructive' })
    }
  }

  const handleStop = async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }

    if (sessionId) {
      try {
        await stopMutation.mutateAsync({ session_id: sessionId })
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to stop stream session'
        toast({ message, variant: 'destructive' })
      }
    }

    setIsStreaming(false)
    setSessionId(null)
    setDetections([])
    setFramesProcessed(0)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    }
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="flex flex-col gap-4">
        <div>
          <Label className="text-sm">Target FPS: {targetFps}</Label>
          <Slider
            value={[targetFps]}
            onValueChange={([v]) => setTargetFps(v)}
            min={1}
            max={30}
            step={1}
            className="mt-2"
            disabled={isStreaming}
          />
        </div>

        <div>
          <Label className="text-sm">Confidence: {(confidence * 100).toFixed(0)}%</Label>
          <Slider
            value={[confidence]}
            onValueChange={([v]) => setConfidence(v)}
            min={0}
            max={1}
            step={0.05}
            className="mt-2"
            disabled={isStreaming}
          />
        </div>

        {!isStreaming ? (
          <Button onClick={handleStart} disabled={startMutation.isPending} className="w-full">
            {startMutation.isPending ? 'Starting...' : 'Start Stream'}
          </Button>
        ) : (
          <Button onClick={handleStop} variant="destructive" disabled={stopMutation.isPending} className="w-full">
            {stopMutation.isPending ? 'Stopping...' : 'Stop Stream'}
          </Button>
        )}

        {isStreaming && (
          <div className="rounded-lg border border-border p-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <span className="text-muted-foreground">Session</span>
              <span className="truncate font-mono text-xs">{sessionId}</span>
              <span className="text-muted-foreground">Frames</span>
              <span>{framesProcessed}</span>
              <span className="text-muted-foreground">Detections</span>
              <span>{detections.length}</span>
            </div>
          </div>
        )}

        {detections.length > 0 && (
          <div className="flex flex-col gap-1">
            {detections.map((d, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="font-medium">{d.class_name}</span>
                <span className="text-muted-foreground">{(d.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Video / Detection display */}
      <div className="flex flex-col items-center justify-start relative">
        <video ref={videoRef} className={cn('rounded-lg border border-border w-full', !isStreaming && 'hidden')} muted playsInline />
        <canvas ref={canvasRef} className="hidden" />

        {isStreaming && frameDataUrl && detections.length > 0 ? (
          <BoundingBoxCanvas
            imageSrc={frameDataUrl}
            detections={detections}
            className="rounded-lg border border-border absolute inset-0 w-full"
          />
        ) : null}

        {!isStreaming && (
          <div className={cn(
            'w-full aspect-video rounded-lg border-2 border-dashed border-border',
            'flex items-center justify-center text-muted-foreground text-sm'
          )}>
            Start streaming to see live detections
          </div>
        )}
      </div>
    </div>
  )
}
