import { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '../ui/button'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { cn } from '@/lib/utils'
import { useStreamStart, useStreamFrame, useStreamStop, useStreamSessions } from '../../hooks/useVision'
import visionService from '../../api/visionService'
import { PanelIntro } from './PanelIntro'
import FontIcon from '../../common/FontIcon'
import type { Detection } from '../../types/vision'

export function StreamingPanel() {
  const [targetFps, setTargetFps] = useState(5)
  const [confidence, setConfidence] = useState(0.25)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [framesProcessed, setFramesProcessed] = useState(0)
  const [startError, setStartError] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const frameInFlightRef = useRef(false)

  const startMutation = useStreamStart()
  const frameMutation = useStreamFrame()
  const stopMutation = useStreamStop()
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
    return dataUrl.split(',')[1]
  }, [])

  const handleStart = async () => {
    setStartError(null)
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
        cooldown_seconds: 0.5,
      })

      setSessionId(result.session_id)
      sessionIdRef.current = result.session_id
      setIsStreaming(true)

      // Start sending frames with backpressure guard
      intervalRef.current = setInterval(() => {
        if (frameInFlightRef.current) return // skip if previous frame still processing
        const b64 = captureFrame()
        if (b64 && result.session_id) {
          frameInFlightRef.current = true
          frameMutation.mutate(
            { session_id: result.session_id, image: b64 },
            {
              onSuccess: data => {
                frameInFlightRef.current = false
                // Only update detections when the backend actually returns them;
                // cooldown-suppressed frames return status:"ok" with no detections
                // and we don't want to flash-clear the display
                if (data.detections && data.detections.length > 0) {
                  setDetections(data.detections)
                }
                setFramesProcessed(prev => prev + 1)
              },
              onError: () => {
                frameInFlightRef.current = false
              },
            }
          )
        }
      }, 1000 / targetFps)
    } catch (err) {
      // Clean up camera if API call failed after getUserMedia succeeded
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }
      const message = err instanceof Error ? err.message : 'Failed to start camera'
      setStartError(message)
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
        console.error('Failed to stop stream session:', err)
      }
    }

    sessionIdRef.current = null
    setIsStreaming(false)
    setSessionId(null)
    setDetections([])
    setFramesProcessed(0)
  }

  // Cleanup on unmount — also stop backend session
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
      if (sessionIdRef.current) {
        visionService.streamStop({ session_id: sessionIdRef.current }).catch(() => {})
      }
    }
  }, [])

  return (
    <div className="flex flex-col gap-3">
      <PanelIntro title="Stream">
        Connect a live camera or video feed for real-time object detection. Set up detection cascades that trigger actions when objects are detected with high confidence.
      </PanelIntro>

      {/* Always-mounted video (hidden when not streaming) + hidden canvas for frame capture */}
      <video ref={videoRef} className={cn(
        'rounded-lg border border-border object-contain bg-black',
        isStreaming ? 'max-h-[360px] w-full' : 'hidden'
      )} muted playsInline />
      <canvas ref={canvasRef} className="hidden" />

      {!isStreaming ? (
        <div className="flex flex-col gap-6">
          <div className="rounded-lg border border-border p-6">
            <h3 className="text-sm font-medium mb-3">Real-time Vision Streaming</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Stream video from your camera for continuous object detection. The pipeline works as:
            </p>
            <div className="flex items-center gap-3 text-sm text-muted-foreground mb-4">
              <span className="px-2 py-1 rounded bg-secondary text-secondary-foreground text-xs font-medium">YOLO Detect</span>
              <span>&rarr;</span>
              <span className="px-2 py-1 rounded bg-secondary text-secondary-foreground text-xs font-medium">CLIP Classify</span>
              <span>&rarr;</span>
              <span className="px-2 py-1 rounded bg-secondary text-secondary-foreground text-xs font-medium">Action Triggers</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Objects detected above your confidence threshold are classified, and you can configure actions (alerts, logging, webhooks) for specific classes.
            </p>
          </div>

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
                />
              </div>

              <Button onClick={handleStart} disabled={startMutation.isPending} className="w-full">
                {startMutation.isPending ? 'Starting...' : 'Start Stream'}
              </Button>

              {(startMutation.isError || startError) && (
                <p className="text-sm text-destructive">
                  {startError ?? (startMutation.error as Error).message}
                </p>
              )}
            </div>

            <div className="flex flex-col items-center justify-center">
              <div className={cn(
                'w-full aspect-video rounded-lg border-2 border-dashed border-border',
                'flex flex-col items-center justify-center text-muted-foreground gap-2'
              )}>
                <FontIcon type="eye" className="w-8 h-8" />
                <p className="text-sm">Camera preview will appear here</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {/* Stats bar */}
          <div className="flex items-center gap-4 text-sm">
            <span className="text-muted-foreground text-xs">Session <span className="font-mono text-foreground">{sessionId?.slice(0, 8)}</span></span>
            <Button onClick={handleStop} variant="destructive" size="sm" disabled={stopMutation.isPending} className="ml-auto">
              {stopMutation.isPending ? 'Stopping...' : 'Stop Stream'}
            </Button>
          </div>

          {/* Detection list — compact table below video */}
          <div className="rounded-lg border border-border overflow-hidden flex flex-col max-h-[280px]">
            <div className="px-3 py-1.5 border-b border-border bg-secondary/30 flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                {detections.length} object{detections.length !== 1 ? 's' : ''} detected
              </span>
              <span className="text-xs text-muted-foreground font-mono">{framesProcessed} frames</span>
            </div>
            {detections.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground p-4">
                Waiting for detections…
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto divide-y divide-border">
                {detections.map((d, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-1.5 text-sm">
                    <div className="flex items-center gap-2">
                      <div className={cn(
                        'w-2 h-2 rounded-full shrink-0',
                        d.confidence >= 0.8 ? 'bg-green-500' :
                        d.confidence >= 0.5 ? 'bg-yellow-500' :
                        'bg-red-500'
                      )} />
                      <span className="font-medium">{d.class_name}</span>
                    </div>
                    <span className={cn(
                      'text-xs font-mono',
                      d.confidence >= 0.8 ? 'text-green-600' :
                      d.confidence >= 0.5 ? 'text-yellow-600' :
                      'text-red-600'
                    )}>
                      {(d.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
