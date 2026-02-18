import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { Badge } from '../ui/badge'
import { cn } from '@/lib/utils'
import { useDetect, useClassify, useDetectClassify } from '../../hooks/useVision'
import { useImageUpload } from './useImageUpload'
import { ImageDropzone } from './ImageDropzone'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import { PanelIntro } from './PanelIntro'
import type { Detection, ClassifyResponse, DetectClassifyResult } from '../../types/vision'

type AnalyzeMode = 'detect' | 'classify' | 'detect-classify'

const MODES: { id: AnalyzeMode; label: string }[] = [
  { id: 'detect', label: 'Detect' },
  { id: 'classify', label: 'Classify' },
  { id: 'detect-classify', label: 'Detect + Classify' },
]

const YOLO_MODELS = [
  { value: 'yolov8n', label: 'YOLOv8n (fastest)' },
  { value: 'yolov8s', label: 'YOLOv8s (balanced)' },
  { value: 'yolov8m', label: 'YOLOv8m (accurate)' },
]

const PRESETS: Record<string, string[]> = {
  Animals: ['cat', 'dog', 'bird', 'fish', 'horse'],
  Vehicles: ['car', 'truck', 'bike', 'bus', 'plane'],
  Food: ['pizza', 'burger', 'salad', 'sushi', 'cake'],
}

function confidenceBadgeColor(c: number): string {
  if (c >= 0.8) return 'bg-green-500/15 text-green-700 border-green-500/30'
  if (c >= 0.5) return 'bg-yellow-500/15 text-yellow-700 border-yellow-500/30'
  return 'bg-red-500/15 text-red-700 border-red-500/30'
}

function summarizeDetections(detections: Detection[]): string {
  const counts: Record<string, number> = {}
  for (const d of detections) {
    counts[d.class_name] = (counts[d.class_name] || 0) + 1
  }
  const parts = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `${count} ${name}`)
  return `Found ${detections.length} object${detections.length !== 1 ? 's' : ''}: ${parts.join(', ')}`
}

export function AnalyzePanel() {
  const [mode, setMode] = useState<AnalyzeMode>('detect')

  // Shared state
  const { imageBase64, imagePreview, fileName, fileSize, handleFileChange, handleFileDirect, clear } = useImageUpload()

  // Detect state
  const [detectModel, setDetectModel] = useState('yolov8s')
  const [confidence, setConfidence] = useState(0.25)
  const [detections, setDetections] = useState<Detection[]>([])

  // Classify state
  const [classInput, setClassInput] = useState('')
  const [classes, setClasses] = useState<string[]>([])
  const [classifyResult, setClassifyResult] = useState<ClassifyResponse | null>(null)

  // Detect+Classify state
  const [dcResults, setDcResults] = useState<DetectClassifyResult[]>([])

  const detectMutation = useDetect()
  const classifyMutation = useClassify()
  const dcMutation = useDetectClassify()

  const handleClear = () => {
    clear()
    setDetections([])
    setClassifyResult(null)
    setDcResults([])
  }

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDetections([])
    setClassifyResult(null)
    setDcResults([])
    handleFileChange(e)
  }

  const handleFileDirectUpload = (file: File) => {
    setDetections([])
    setClassifyResult(null)
    setDcResults([])
    handleFileDirect(file)
  }

  // Classify helpers
  const addCls = () => {
    const val = classInput.trim()
    if (val && !classes.includes(val)) {
      setClasses(prev => [...prev, val])
      setClassInput('')
    }
  }
  const removeCls = (cls: string) => setClasses(prev => prev.filter(c => c !== cls))
  const addPreset = (preset: string[]) => {
    setClasses(prev => [...new Set([...prev, ...preset])])
  }

  // Actions
  const handleDetect = () => {
    if (!imageBase64) return
    detectMutation.mutate(
      { image: imageBase64, model: detectModel || undefined, confidence_threshold: confidence },
      { onSuccess: data => setDetections(data.detections) }
    )
  }

  const handleClassify = () => {
    if (!imageBase64) return
    classifyMutation.mutate(
      { image: imageBase64, classes: classes.length > 0 ? classes : undefined, top_k: 10 },
      { onSuccess: setClassifyResult }
    )
  }

  const handleDetectClassify = () => {
    if (!imageBase64) return
    dcMutation.mutate(
      {
        image: imageBase64,
        detect_model: detectModel || undefined,
        confidence_threshold: confidence,
        classify_classes: classes.length > 0 ? classes : undefined,
      },
      { onSuccess: data => setDcResults(data.results) }
    )
  }

  const isPending = detectMutation.isPending || classifyMutation.isPending || dcMutation.isPending
  const mutationError = detectMutation.error || classifyMutation.error || dcMutation.error

  const handleRun = () => {
    if (mode === 'detect') handleDetect()
    else if (mode === 'classify') handleClassify()
    else handleDetectClassify()
  }

  const runLabel =
    mode === 'detect'
      ? detectMutation.isPending ? 'Detecting...' : 'Detect'
      : mode === 'classify'
        ? classifyMutation.isPending ? 'Classifying...' : 'Classify'
        : dcMutation.isPending ? 'Processing...' : 'Detect + Classify'

  // Build detections for D+C overlay
  const dcDetections: Detection[] = dcResults.map(r => ({
    ...r.detection,
    class_name: `${r.detection.class_name} → ${r.classification.class_name}`,
  }))

  const activeDetections = mode === 'detect' ? detections : mode === 'detect-classify' ? dcDetections : []
  return (
    <div className="flex flex-col gap-4">
      <PanelIntro>
        Upload and analyze images using state-of-the-art models. Your images are processed locally — nothing leaves your machine.
      </PanelIntro>

      {/* Mode selector */}
      <div className="flex rounded-lg border border-border p-1 bg-muted/30 w-fit">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={cn(
              'px-4 py-1.5 text-sm rounded-md transition-colors',
              mode === m.id
                ? 'bg-background text-foreground font-medium shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: controls */}
        <div className="flex flex-col gap-4">
          {/* Upload */}
          <ImageDropzone
            onFileSelected={handleFileSelected}
            onFileDirect={handleFileDirectUpload}
            fileName={fileName}
            fileSize={fileSize}
            imagePreview={imagePreview}
            onClear={handleClear}
          />

          {/* Detect config */}
          {(mode === 'detect' || mode === 'detect-classify') && (
            <>
              <div>
                <Label className="text-sm">Detection Model</Label>
                <select
                  value={detectModel}
                  onChange={e => setDetectModel(e.target.value)}
                  className={cn(
                    'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm',
                    'ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
                  )}
                >
                  {YOLO_MODELS.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <Label className="text-sm">
                  Confidence Threshold: {(confidence * 100).toFixed(0)}%
                </Label>
                <Slider
                  value={[confidence]}
                  onValueChange={([v]) => setConfidence(v)}
                  min={0}
                  max={1}
                  step={0.05}
                  className="mt-2"
                />
              </div>
            </>
          )}

          {/* Classify config */}
          {(mode === 'classify' || mode === 'detect-classify') && (
            <div>
              <Label className="text-sm">Classification Labels</Label>
              <p className="text-xs text-muted-foreground mt-1 mb-2">
                {mode === 'classify'
                  ? 'CLIP understands natural language — describe what you\'re looking for and it will match against your image. No training required for common concepts.'
                  : 'Detect objects first, then classify each one. Example: detect all birds, then identify each species.'}
              </p>
              <div className="flex gap-2">
                <Input
                  value={classInput}
                  onChange={e => setClassInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), addCls())}
                  placeholder="Type a label and press Enter..."
                  className="flex-1"
                />
              </div>
              {classes.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {classes.map(cls => (
                    <Badge key={cls} variant="secondary" className="cursor-pointer gap-1" onClick={() => removeCls(cls)}>
                      {cls}
                      <span className="text-muted-foreground">×</span>
                    </Badge>
                  ))}
                </div>
              )}
              <div className="flex gap-2 mt-2">
                {Object.entries(PRESETS).map(([name, labels]) => (
                  <Button key={name} variant="outline" size="sm" onClick={() => addPreset(labels)}>
                    {name}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Run button */}
          <Button onClick={handleRun} disabled={!imageBase64 || isPending} className="w-full">
            {runLabel}
          </Button>

          {mutationError && (
            <p className="text-sm text-destructive">{(mutationError as Error).message}</p>
          )}

          {/* Detect results list */}
          {mode === 'detect' && detections.length > 0 && (
            <div className="rounded-lg border border-border p-3">
              <p className="text-sm font-medium mb-2">{summarizeDetections(detections)}</p>
              <div className="flex flex-col gap-1">
                {detections.map((d, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="font-medium">{d.class_name}</span>
                    <span className={cn('text-xs px-2 py-0.5 rounded-full border', confidenceBadgeColor(d.confidence))}>
                      {(d.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Classify results */}
          {mode === 'classify' && classifyResult && (
            <div className="rounded-lg border border-border p-3">
              <p className="text-sm font-medium mb-3">
                Top result: <span className="text-primary">{classifyResult.class_name}</span>{' '}
                ({(classifyResult.confidence * 100).toFixed(1)}%)
              </p>
              <div className="flex flex-col gap-2">
                {classifyResult.all_scores.map((s, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-sm w-24 truncate">{s.class_name}</span>
                    <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={cn(
                          'h-full rounded-full transition-all',
                          i === 0 ? 'bg-primary' : 'bg-primary/60'
                        )}
                        style={{ width: `${s.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground w-12 text-right">
                      {(s.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* D+C results list */}
          {mode === 'detect-classify' && dcResults.length > 0 && (
            <div className="rounded-lg border border-border p-3">
              <p className="text-sm font-medium mb-2">
                {dcResults.length} object{dcResults.length !== 1 ? 's' : ''} found
              </p>
              <div className="flex flex-col gap-2">
                {dcResults.map((r, i) => (
                  <div key={i} className="rounded-md border border-border p-2 text-sm">
                    <div className="flex justify-between">
                      <span className="font-medium">{r.detection.class_name}</span>
                      <span className={cn('text-xs px-2 py-0.5 rounded-full border', confidenceBadgeColor(r.detection.confidence))}>
                        {(r.detection.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Classified as: <span className="text-primary">{r.classification.class_name}</span>{' '}
                      ({(r.classification.confidence * 100).toFixed(1)}%)
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: preview / results */}
        <div className="flex flex-col items-center justify-start">
          {imagePreview && activeDetections.length > 0 ? (
            <BoundingBoxCanvas
              imageSrc={imagePreview}
              detections={activeDetections}
              className="rounded-lg border border-border"
            />
          ) : imagePreview ? (
            <img
              src={imagePreview}
              alt="Uploaded"
              className="max-w-full rounded-lg border border-border"
            />
          ) : (
            <div className={cn(
              'w-full aspect-video rounded-lg border-2 border-dashed border-border',
              'flex items-center justify-center text-muted-foreground text-sm'
            )}>
              Upload an image to get started
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
