import { useState, useCallback } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { Badge } from '../ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { Upload, X, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useDetect, useClassify, useDetectClassify } from '../../hooks/useVision'
import { useImageUpload } from './useImageUpload'
import { PanelIntro } from './PanelIntro'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import type { Detection, ClassifyResponse, DetectClassifyResult } from '../../types/vision'

type AnalyzeMode = 'detect' | 'classify' | 'detect-classify'

const MODES: { id: AnalyzeMode; label: string; desc: string }[] = [
  { id: 'detect', label: 'Detect', desc: 'Find objects and their locations' },
  { id: 'classify', label: 'Classify', desc: 'Categorize the whole image' },
  { id: 'detect-classify', label: 'Detect + Classify', desc: 'Find objects, then classify each one' },
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
  const [isDragOver, setIsDragOver] = useState(false)
  const [previewError, setPreviewError] = useState(false)

  const { imageBase64, imagePreview, fileName, fileSize, handleFileDirect, clear } = useImageUpload()

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

  const handleNewFile = (file: File) => {
    setDetections([])
    setClassifyResult(null)
    setDcResults([])
    setPreviewError(false)
    handleFileDirect(file)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) handleNewFile(file)
  }, [])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) handleNewFile(file)
        break
      }
    }
  }, [])

  // Classify helpers
  const addCls = () => {
    const val = classInput.trim()
    if (val && !classes.includes(val)) {
      setClasses(prev => [...prev, val])
      setClassInput('')
    }
  }
  const removeCls = (cls: string) => setClasses(prev => prev.filter(c => c !== cls))
  const addPreset = (preset: string[]) => setClasses(prev => [...new Set([...prev, ...preset])])

  // Actions
  const handleDetect = () => {
    if (!imageBase64) return
    detectMutation.mutate(
      { image: imageBase64, model: detectModel || undefined, confidence_threshold: confidence },
      { onSuccess: data => setDetections(data.detections) }
    )
  }

  const handleClassify = () => {
    if (!imageBase64 || classes.length === 0) return
    classifyMutation.mutate(
      { image: imageBase64, classes, top_k: 10 },
      { onSuccess: setClassifyResult }
    )
  }

  const [dcPending, setDcPending] = useState(false)
  const [dcError, setDcError] = useState<Error | null>(null)

  const handleDetectClassify = async () => {
    if (!imageBase64) return
    setDcPending(true)
    setDcError(null)
    setDcResults([])
    try {
      // Try combo endpoint first (available when PR #774 is merged)
      dcMutation.mutate(
        {
          image: imageBase64,
          detect_model: detectModel || undefined,
          confidence_threshold: confidence,
          classify_classes: classes.length > 0 ? classes : undefined,
        },
        {
          onSuccess: data => { setDcResults(data.results); setDcPending(false) },
          onError: async () => {
            // Fallback: run detect then classify sequentially
            try {
              const { detect, classify } = await import('../../api/visionService')
              const detectRes = await detect({
                image: imageBase64,
                model: detectModel || undefined,
                confidence_threshold: confidence,
              })
              const detectedObjects = detectRes.detections || []
              if (detectedObjects.length === 0) {
                setDcResults([])
                setDcPending(false)
                return
              }
              // Classify the whole image with detected class names + user classes
              const detectedClasses = [...new Set(detectedObjects.map(d => d.class_name))]
              const allClasses = [...new Set([...detectedClasses, ...classes])]
              const classifyRes = await classify({
                image: imageBase64,
                classes: allClasses.length > 0 ? allClasses : ['object'],
                top_k: 10,
              })
              // Merge: each detection gets the classification result
              const results: DetectClassifyResult[] = detectedObjects.map(det => ({
                detection: det,
                classification: classifyRes,
              }))
              setDcResults(results)
            } catch (fallbackErr) {
              setDcError(fallbackErr instanceof Error ? fallbackErr : new Error(String(fallbackErr)))
            } finally {
              setDcPending(false)
            }
          },
        }
      )
    } catch (err) {
      setDcError(err instanceof Error ? err : new Error(String(err)))
      setDcPending(false)
    }
  }

  const isPending = detectMutation.isPending || classifyMutation.isPending || dcMutation.isPending || dcPending
  const mutationError = detectMutation.error || classifyMutation.error || dcError

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
        : (dcMutation.isPending || dcPending) ? 'Processing...' : 'Detect + Classify'

  const dcDetections: Detection[] = dcResults.map(r => ({
    ...r.detection,
    class_name: `${r.detection.class_name} → ${r.classification.class_name}`,
  }))

  const activeDetections = mode === 'detect' ? detections : mode === 'detect-classify' ? dcDetections : []
  return (
    <div
      className="flex flex-col gap-4"
      onPaste={handlePaste}
      tabIndex={0}
    >
      <PanelIntro title="Analyze">
        Analyze images using state-of-the-art vision models. Your images are processed locally — nothing leaves your machine.
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
      <p className="text-xs text-muted-foreground -mt-2">{MODES.find(m => m.id === mode)?.desc}</p>

      {/* Image area — drop zone that becomes the preview */}
      {!imagePreview ? (
        <>
          <label
            className={cn(
              'relative flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 cursor-pointer transition-colors',
              isDragOver
                ? 'border-primary bg-primary/5'
                : 'border-border hover:border-primary/50 hover:bg-muted/30'
            )}
            onDragOver={e => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
          >
            <Upload className="w-8 h-8 text-muted-foreground" />
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">Drop an image here, click to browse, or paste from clipboard</p>
              <p className="text-xs text-muted-foreground mt-1">Supports JPG, PNG, WEBP, GIF</p>
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleNewFile(f) }}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
          </label>
          <div className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground -mt-2">
            <span>Need test images?</span>
            <a
              href="https://github.com/llama-farm/vision-sample-data"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              Sample data repo <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </>
      ) : (
        <div className="relative group max-h-[350px] overflow-hidden rounded-lg border border-border">
          {activeDetections.length > 0 ? (
            <BoundingBoxCanvas
              imageSrc={imagePreview}
              detections={activeDetections}
              className="w-full max-h-[350px] object-contain"
            />
          ) : previewError ? (
            <div className="w-full bg-muted/30 p-6 flex flex-col items-center gap-2">
              <span className="text-2xl">📄</span>
              <p className="text-sm font-medium">{fileName}</p>
              <p className="text-xs text-muted-foreground">{fileSize ? `${(fileSize / 1024).toFixed(0)} KB` : ''} — Preview not available (file will still be sent for analysis)</p>
            </div>
          ) : (
            <img
              src={imagePreview}
              alt="Uploaded"
              className="w-full max-h-[350px] object-contain"
              onError={() => setPreviewError(true)}
            />
          )}
          <div className="absolute top-2 right-2 flex gap-1">
            <label className="p-1.5 rounded-md bg-background/80 backdrop-blur border border-border cursor-pointer hover:bg-background" title="Upload another image">
              <Upload className="w-4 h-4" />
              <input
                type="file"
                accept="image/*"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleNewFile(f) }}
                className="hidden"
              />
            </label>
            <button
              onClick={handleClear}
              className="p-1.5 rounded-md bg-background/80 backdrop-blur border border-border hover:bg-background"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          {fileName && (
            <p className="text-xs text-muted-foreground mt-1">{fileName} {fileSize ? `(${(fileSize / 1024).toFixed(0)} KB)` : ''}</p>
          )}
        </div>
      )}

      {/* Configuration */}
      {(mode === 'detect' || mode === 'detect-classify') && (
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <Label className="text-sm">Model</Label>
            <Select value={detectModel} onValueChange={setDetectModel}>
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {YOLO_MODELS.map(m => (
                  <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex-1">
            <Label className="text-sm">Confidence: {(confidence * 100).toFixed(0)}%</Label>
            <Slider
              value={[confidence]}
              onValueChange={([v]) => setConfidence(v)}
              min={0}
              max={1}
              step={0.05}
              className="mt-3"
            />
          </div>
        </div>
      )}

      {(mode === 'classify' || mode === 'detect-classify') && (
        <div>
          <Label className="text-sm">Classification Labels</Label>
          <p className="text-xs text-muted-foreground mt-1 mb-2">
            {mode === 'classify'
              ? 'CLIP understands natural language — type what you\'re looking for. No training needed for common concepts. For domain-specific accuracy, train a custom model.'
              : 'Each detected object will be classified against these labels.'}
          </p>
          <div className="flex gap-2">
            <Input
              value={classInput}
              onChange={e => setClassInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), addCls())}
              placeholder="Type a label and press Enter..."
              className="flex-1"
            />
            <Button variant="outline" size="sm" onClick={addCls} disabled={!classInput.trim()}>Add</Button>
          </div>
          {classes.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {classes.map(cls => (
                <Badge key={cls} variant="secondary" className="cursor-pointer gap-1" onClick={() => removeCls(cls)}>
                  {cls} <span className="text-muted-foreground">×</span>
                </Badge>
              ))}
            </div>
          )}
          <div className="flex gap-2 mt-2">
            <span className="text-xs text-muted-foreground self-center">Presets:</span>
            {Object.entries(PRESETS).map(([name, labels]) => (
              <Button key={name} variant="ghost" size="sm" className="text-xs h-7" onClick={() => addPreset(labels)}>
                {name}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Run button */}
      <Button onClick={handleRun} disabled={!imageBase64 || isPending || ((mode === 'classify' || mode === 'detect-classify') && classes.length === 0)} className="w-fit">
        {runLabel}
      </Button>

      {mutationError && (
        <p className="text-sm text-destructive">{(mutationError as Error).message}</p>
      )}

      {/* Results */}
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

      {mode === 'classify' && classifyResult && (
        <div className="rounded-lg border border-border p-3">
          <p className="text-sm font-medium mb-3">
            Top result: <span className="text-primary">{classifyResult.class_name}</span>{' '}
            ({(classifyResult.confidence * 100).toFixed(1)}%)
          </p>
          <div className="flex flex-col gap-2">
            {Object.entries(classifyResult.all_scores)
              .sort(([, a], [, b]) => b - a)
              .map(([name, score], i) => (
              <div key={name} className="flex items-center gap-2">
                <span className="text-sm w-24 truncate">{name}</span>
                <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className={cn('h-full rounded-full transition-all', i === 0 ? 'bg-primary' : 'bg-primary/60')}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-12 text-right">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {mode === 'detect-classify' && dcResults.length > 0 && (
        <div className="rounded-lg border border-border p-3">
          <p className="text-sm font-medium mb-2">{dcResults.length} object{dcResults.length !== 1 ? 's' : ''} found</p>
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
                  → {r.classification.class_name} ({(r.classification.confidence * 100).toFixed(1)}%)
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
