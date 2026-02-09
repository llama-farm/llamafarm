import { useState, useCallback, useRef, useEffect } from 'react'
import { Button } from '../ui/button'
import { Label } from '../ui/label'
import { Input } from '../ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { useToast } from '../ui/toast'
import { Slider } from '../ui/slider'
import { Switch } from '../ui/switch'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'

const RUNTIME_URL = ''

type VisionMode = 'detect' | 'classify' | 'segment' | 'ocr'

interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
}

interface Detection {
  box: Box
  class_name: string
  confidence: number
  mask?: number[][] // RLE or polygon points for segmentation
}

interface ClassificationResult {
  label: string
  confidence: number
}

interface OcrRegion {
  text: string
  box?: Box
  confidence?: number
}

interface SessionStats {
  session_id: string
  frames_processed: number
  actions_triggered: number
  escalations_to_secondary: number
  secondary_successes: number
  secondary_success_rate: number
  review_queue_count: number
}

interface ReplayBufferStats {
  size: number
  max_size: number
  by_source: Record<string, number>
  avg_priority: number
}

interface QueuedImage {
  id: string
  data: string
  name: string
  status: 'pending' | 'processing' | 'done' | 'error'
  // Results vary by mode
  detections?: Detection[]
  classifications?: ClassificationResult[]
  segments?: Detection[] // Same structure but with masks
  ocrRegions?: OcrRegion[]
  ocrFullText?: string
}

const DETECTION_MODELS = [
  { id: 'yolov8n', label: 'YOLOv8 Nano (fastest)' },
  { id: 'yolov8s', label: 'YOLOv8 Small' },
  { id: 'yolov8m', label: 'YOLOv8 Medium' },
  { id: 'yolov8l', label: 'YOLOv8 Large' },
]

const CLASSIFICATION_MODELS = [
  { id: 'clip-vit-base', label: 'CLIP ViT-Base (general)' },
  { id: 'clip-vit-large', label: 'CLIP ViT-Large (accurate)' },
  { id: 'blip-base', label: 'BLIP Base (captioning)' },
]

const SEGMENTATION_MODELS = [
  { id: 'yolov8n-seg', label: 'YOLOv8 Nano Seg' },
  { id: 'yolov8s-seg', label: 'YOLOv8 Small Seg' },
  { id: 'yolov8m-seg', label: 'YOLOv8 Medium Seg' },
  { id: 'sam-base', label: 'SAM Base (interactive)' },
]

const OCR_MODELS = [
  { id: 'easyocr', label: 'EasyOCR (multilingual)' },
  { id: 'tesseract', label: 'Tesseract (fast)' },
  { id: 'paddleocr', label: 'PaddleOCR (accurate)' },
]

const MODE_INFO: Record<VisionMode, { 
  label: string
  description: string
  icon: string
  models: { id: string; label: string }[]
}> = {
  detect: { 
    label: 'Detection', 
    description: 'Find objects with bounding boxes', 
    icon: 'search',
    models: DETECTION_MODELS,
  },
  classify: { 
    label: 'Classification', 
    description: 'What is this image?', 
    icon: 'data',
    models: CLASSIFICATION_MODELS,
  },
  segment: { 
    label: 'Segmentation', 
    description: 'Pixel-level object masks', 
    icon: 'fade',
    models: SEGMENTATION_MODELS,
  },
  ocr: { 
    label: 'OCR', 
    description: 'Extract text from images', 
    icon: 'prompt',
    models: OCR_MODELS,
  },
}

// Distinct colors for visualization
const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
]

export function StreamingPanel() {
  const { toast } = useToast()
  
  const [mode, setMode] = useState<VisionMode>('detect')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [stats, setStats] = useState<SessionStats | null>(null)
  const [replayStats, setReplayStats] = useState<ReplayBufferStats | null>(null)
  
  // Model per mode
  const [detectModel, setDetectModel] = useState('yolov8n')
  const [classifyModel, setClassifyModel] = useState('clip-vit-base')
  const [segmentModel, setSegmentModel] = useState('yolov8n-seg')
  const [ocrModel, setOcrModel] = useState('easyocr')
  
  // Config
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [topK, setTopK] = useState(5) // For classification
  const [classifyClasses, setClassifyClasses] = useState('cat, dog, bird, car, person, food, building, nature')
  const [enableCascade, setEnableCascade] = useState(true)
  
  // Image queue
  const [imageQueue, setImageQueue] = useState<QueuedImage[]>([])
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const [isProcessingQueue, setIsProcessingQueue] = useState(false)
  
  // Drag state
  const [isDragging, setIsDragging] = useState(false)
  
  // Correction state
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editLabel, setEditLabel] = useState('')
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)
  
  const selectedImage = imageQueue.find(img => img.id === selectedImageId)
  const currentModel = mode === 'detect' ? detectModel : mode === 'classify' ? classifyModel : mode === 'segment' ? segmentModel : ocrModel
  
  // Fetch stats
  const fetchReplayStats = useCallback(async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/replay-buffer`)
      if (response.ok) {
        const data = await response.json()
        setReplayStats(data.stats)
      }
    } catch (error) {}
  }, [])
  
  const fetchSessionStats = useCallback(async () => {
    if (!sessionId) return
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/stream/sessions/${sessionId}/stats`)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      }
    } catch (error) {}
  }, [sessionId])
  
  useEffect(() => {
    if (!isStreaming || !sessionId) return
    const interval = setInterval(() => {
      fetchSessionStats()
      fetchReplayStats()
    }, 2000)
    return () => clearInterval(interval)
  }, [isStreaming, sessionId, fetchSessionStats, fetchReplayStats])
  
  useEffect(() => { fetchReplayStats() }, [fetchReplayStats])
  
  // Draw overlays based on mode
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !selectedImage?.data) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = imageRef.current

    const draw = () => {
      if (!ctx || !img || !canvas) return

      // Match canvas pixel size to the container so coordinates align
      const container = canvas.parentElement
      if (!container) return
      const cw = container.clientWidth
      const ch = container.clientHeight
      canvas.width = cw
      canvas.height = ch
      ctx.clearRect(0, 0, cw, ch)

      // Compute where object-contain places the image inside the container
      const natW = img.naturalWidth || 1
      const natH = img.naturalHeight || 1
      const scale = Math.min(cw / natW, ch / natH)
      const drawW = natW * scale
      const drawH = natH * scale
      const offsetX = (cw - drawW) / 2
      const offsetY = (ch - drawH) / 2

      // Map natural-pixel coords to canvas coords
      const mapX = (x: number) => offsetX + x * scale
      const mapY = (y: number) => offsetY + y * scale
      
      if (mode === 'detect' && selectedImage.detections) {
        // DETECTION: Bounding boxes with labels
        selectedImage.detections.forEach((det, i) => {
          const color = COLORS[i % COLORS.length]
          const { x1, y1, x2, y2 } = det.box
          const mx1 = mapX(x1), my1 = mapY(y1), mx2 = mapX(x2), my2 = mapY(y2)

          // Box
          ctx.strokeStyle = color
          ctx.lineWidth = 3
          ctx.strokeRect(mx1, my1, mx2 - mx1, my2 - my1)

          // Label background
          const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`
          ctx.font = 'bold 14px sans-serif'
          const tm = ctx.measureText(label)
          ctx.fillStyle = color
          ctx.fillRect(mx1, my1 - 22, tm.width + 8, 22)

          // Label text
          ctx.fillStyle = '#FFF'
          ctx.fillText(label, mx1 + 4, my1 - 6)
        })
      }

      if (mode === 'segment' && selectedImage.segments) {
        // SEGMENTATION: Filled masks with semi-transparency
        selectedImage.segments.forEach((seg, i) => {
          const color = COLORS[i % COLORS.length]
          const { x1, y1, x2, y2 } = seg.box
          const mx1 = mapX(x1), my1 = mapY(y1), mx2 = mapX(x2), my2 = mapY(y2)

          // Draw mask if available
          if (seg.mask && seg.mask.length > 0) {
            ctx.beginPath()
            // mask is list of [x, y] points
            const points = seg.mask as unknown as number[][]
            if (points.length > 0) {
              ctx.moveTo(mapX(points[0][0]), mapY(points[0][1]))
              for (let j = 1; j < points.length; j++) {
                ctx.lineTo(mapX(points[j][0]), mapY(points[j][1]))
              }
              ctx.closePath()

              // Fill with semi-transparent color
              ctx.fillStyle = color + '80' // 50% opacity
              ctx.fill()

              // Stroke border
              ctx.strokeStyle = color
              ctx.lineWidth = 2
              ctx.stroke()
            }
          } else {
            // Fallback: Fill box area
            ctx.fillStyle = color + '60' // 40% opacity
            ctx.fillRect(mx1, my1, mx2 - mx1, my2 - my1)
            ctx.strokeStyle = color
            ctx.lineWidth = 2
            ctx.strokeRect(mx1, my1, mx2 - mx1, my2 - my1)
          }

          // Label
          const label = seg.class_name
          ctx.font = 'bold 12px sans-serif'
          const tm = ctx.measureText(label)
          ctx.fillStyle = color
          ctx.fillRect(mx1, my2 - 20, tm.width + 8, 20)
          ctx.fillStyle = '#FFF'
          ctx.fillText(label, mx1 + 4, my2 - 6)
        })
      }

      if (mode === 'ocr' && selectedImage.ocrRegions) {
        // OCR: Highlight text regions
        selectedImage.ocrRegions.forEach((region, i) => {
          if (!region.box) return
          const color = '#45B7D1'
          const { x1, y1, x2, y2 } = region.box
          const mx1 = mapX(x1), my1 = mapY(y1), mx2 = mapX(x2), my2 = mapY(y2)

          // Highlight box
          ctx.fillStyle = color + '30'
          ctx.fillRect(mx1, my1, mx2 - mx1, my2 - my1)
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.strokeRect(mx1, my1, mx2 - mx1, my2 - my1)
        })
      }
      
      // CLASSIFICATION: No overlay - it's about the whole image
    }
    
    if (!img.complete) {
      img.onload = draw
    } else {
      draw()
    }

    // Redraw when container resizes so overlays stay aligned
    const container = canvas.parentElement
    if (container) {
      const ro = new ResizeObserver(() => { if (img.complete) draw() })
      ro.observe(container)
      return () => ro.disconnect()
    }
  }, [selectedImage, mode])
  
  // Process image based on mode
  const processImage = async (queuedImage: QueuedImage): Promise<QueuedImage> => {
    try {
      let endpoint = ''
      let body: Record<string, unknown> = {}
      
      switch (mode) {
        case 'detect':
          endpoint = '/v1/vision/detect'
          body = {
            image: queuedImage.data,
            model: detectModel,
            confidence_threshold: confidenceThreshold,
          }
          break
          
        case 'classify':
          // Classification: whole image → top-K labels (needs classes for CLIP zero-shot)
          endpoint = '/v1/vision/classify'
          body = {
            image: queuedImage.data,
            model: classifyModel,
            classes: classifyClasses.split(',').map(c => c.trim()).filter(c => c),
            top_k: topK,
          }
          break
          
        case 'segment':
          // Segmentation: objects with pixel masks
          endpoint = '/v1/vision/segment'
          body = {
            image: queuedImage.data,
            model: segmentModel,
            confidence_threshold: confidenceThreshold,
            return_masks: true,
          }
          break
          
        case 'ocr':
          // OCR: extract text
          endpoint = '/v1/vision/ocr'
          body = {
            images: [queuedImage.data], // API expects list of images
            model: ocrModel,
            return_boxes: true, // Renamed from return_regions
          }
          break
      }
      
      const response = await fetch(`${RUNTIME_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      
      if (!response.ok) throw new Error(`${response.status}`)
      
      const data = await response.json()
      
      // Stream session tracking (detection mode)
      if (sessionId && mode === 'detect') {
        await fetch(`${RUNTIME_URL}/v1/vision/stream/frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, image: queuedImage.data }),
        })
        fetchSessionStats()
        fetchReplayStats()
      }
      
      // Map response to appropriate field
      const result: QueuedImage = { ...queuedImage, status: 'done' }
      
      switch (mode) {
        case 'detect':
          result.detections = data.detections || []
          break
        case 'classify':
          // API returns { class_name, all_scores: {label: score, ...} }
          // Convert to array format: [{ label, confidence }, ...]
          if (data.all_scores) {
            result.classifications = Object.entries(data.all_scores)
              .map(([label, confidence]) => ({ label, confidence: confidence as number }))
              .sort((a, b) => b.confidence - a.confidence)
          } else {
            result.classifications = data.classifications || data.results || []
          }
          break
        case 'segment':
          result.segments = data.segments || data.detections || []
          break
        case 'ocr':
          // API returns { data: [{ index, text, confidence, boxes }] }
          if (data.data && Array.isArray(data.data)) {
            result.ocrRegions = data.data.flatMap((item: any) => 
              (item.boxes || []).map((box: any) => ({
                text: box.text,
                box: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
                confidence: box.confidence
              }))
            )
            result.ocrFullText = data.data.map((item: any) => item.text).join('\n')
          } else {
            result.ocrRegions = data.regions || []
            result.ocrFullText = data.text || data.regions?.map((r: OcrRegion) => r.text).join('\n') || ''
          }
          break
      }
      
      return result
    } catch (error) {
      console.error('Vision process error:', error)
      return { ...queuedImage, status: 'error' }
    }
  }
  
  // Process queue
  const processQueue = async () => {
    setIsProcessingQueue(true)
    
    for (let i = 0; i < imageQueue.length; i++) {
      const img = imageQueue[i]
      // Skip only if currently processing
      if (img.status === 'processing') continue
      
      setImageQueue(prev => prev.map((item, idx) => 
        idx === i ? { ...item, status: 'processing' } : item
      ))
      setSelectedImageId(img.id)
      
      const result = await processImage(img)
      
      setImageQueue(prev => prev.map((item, idx) => 
        idx === i ? result : item
      ))
      
      await new Promise(r => setTimeout(r, 150))
    }
    
    setIsProcessingQueue(false)
    toast({ message: `Processed ${imageQueue.length} images` })
  }
  
  // File handling
  const handleFiles = useCallback((files: FileList | File[]) => {
    Array.from(files).forEach(file => {
      if (!file.type.startsWith('image/')) return
      
      const reader = new FileReader()
      reader.onload = () => {
        const newImage: QueuedImage = {
          id: Math.random().toString(36).substr(2, 9),
          data: reader.result as string,
          name: file.name,
          status: 'pending',
        }
        setImageQueue(prev => {
          const updated = [...prev, newImage]
          if (prev.length === 0) setSelectedImageId(newImage.id)
          return updated
        })
      }
      reader.readAsDataURL(file)
    })
  }, [])
  
  const handleDragEnter = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true) }
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); if (e.currentTarget === dropZoneRef.current) setIsDragging(false) }
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault() }
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files)
  }
  
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) handleFiles(e.target.files)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }
  
  const clearQueue = () => { setImageQueue([]); setSelectedImageId(null) }
  
  // Session controls
  const startSession = async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/stream/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: detectModel,
          config: { confidence_threshold: confidenceThreshold, cascade: enableCascade ? { secondary_model_id: 'yolov8m' } : null },
        }),
      })
      if (!response.ok) throw new Error()
      const data = await response.json()
      setSessionId(data.session_id)
      setIsStreaming(true)
      toast({ message: `Session started` })
    } catch (error) {
      toast({ message: 'Failed to start session', variant: 'destructive' })
    }
  }
  
  const stopSession = async () => {
    if (!sessionId) return
    await fetch(`${RUNTIME_URL}/v1/vision/stream/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    })
    setIsStreaming(false)
    toast({ message: 'Session stopped' })
  }
  
  // Corrections
  const submitCorrection = async (original: string, corrected: string, index: number) => {
    if (!selectedImage?.data) return

    try {
      // Build image_id from selected image name + index
      const imageId = `${selectedImage.id}_${index}`

      // Get original confidence from the detection/segment
      let originalConfidence = 0.0
      let box: Box | null = null
      if (mode === 'detect' && selectedImage.detections?.[index]) {
        originalConfidence = selectedImage.detections[index].confidence
        box = selectedImage.detections[index].box
      } else if (mode === 'segment' && selectedImage.segments?.[index]) {
        originalConfidence = selectedImage.segments[index].confidence
        box = selectedImage.segments[index].box
      } else if (mode === 'classify' && selectedImage.classifications?.[index]) {
        originalConfidence = selectedImage.classifications[index].confidence
      }

      await fetch(`${RUNTIME_URL}/v1/vision/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          corrected_class: corrected,
          original_confidence: originalConfidence,
          box,
          session_id: sessionId,
        }),
      })
      
      // Update local state
      setImageQueue(prev => prev.map(img => {
        if (img.id !== selectedImageId) return img
        
        if (mode === 'detect' && img.detections) {
          return { ...img, detections: img.detections.map((d, i) => i === index ? { ...d, class_name: corrected } : d) }
        }
        if (mode === 'classify' && img.classifications) {
          return { ...img, classifications: img.classifications.map((c, i) => i === index ? { ...c, label: corrected } : c) }
        }
        if (mode === 'segment' && img.segments) {
          return { ...img, segments: img.segments.map((s, i) => i === index ? { ...s, class_name: corrected } : s) }
        }
        return img
      }))
      
      setEditingIndex(null)
      setEditLabel('')
      toast({ message: `Saved: "${original}" → "${corrected}"` })
      fetchReplayStats()
    } catch (error) {
      toast({ message: 'Failed to save', variant: 'destructive' })
    }
  }
  
  // Results for current selection
  const detections = selectedImage?.detections || []
  const classifications = selectedImage?.classifications || []
  const segments = selectedImage?.segments || []
  const ocrText = selectedImage?.ocrFullText || ''
  const ocrRegions = selectedImage?.ocrRegions || []
  
  return (
    <div className="space-y-6">
      {/* Mode Tabs */}
      <div className="flex gap-1 p-1 bg-muted rounded-lg">
        {(Object.keys(MODE_INFO) as VisionMode[]).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 flex flex-col items-center gap-1 px-4 py-3 rounded-md transition-colors ${
              mode === m ? 'bg-background shadow-sm' : 'hover:bg-background/50'
            }`}
          >
            <FontIcon type={MODE_INFO[m].icon as any} className="w-5 h-5" />
            <span className={`text-sm font-medium ${mode === m ? '' : 'text-muted-foreground'}`}>
              {MODE_INFO[m].label}
            </span>
            <span className="text-xs text-muted-foreground">{MODE_INFO[m].description}</span>
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-4">
          {/* Drop Zone */}
          <div
            ref={dropZoneRef}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
              isDragging ? 'border-primary bg-primary/10 scale-[1.02]' : 'border-border hover:border-primary/50'
            }`}
          >
            <FontIcon type="upload" className="w-10 h-10 mx-auto mb-2 text-muted-foreground" />
            <p className="font-medium">Drop images here</p>
            <p className="text-sm text-muted-foreground">or click to browse • multiple files supported</p>
            <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={handleFileInput} />
          </div>
          
          {/* Image Queue */}
          {imageQueue.length > 0 && (
            <div className="bg-card border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="font-medium">
                  {imageQueue.filter(i => i.status === 'done').length}/{imageQueue.length} processed
                </span>
                <div className="flex gap-2">
                  <Button size="sm" onClick={processQueue} disabled={isProcessingQueue}>
                    {isProcessingQueue ? <><Loader className="w-4 h-4 mr-2" />Running...</> : 'Process All'}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={clearQueue}>Clear</Button>
                </div>
              </div>
              
              <div className="flex gap-2 overflow-x-auto pb-2">
                {imageQueue.map(img => (
                  <div
                    key={img.id}
                    onClick={() => setSelectedImageId(img.id)}
                    className={`relative flex-shrink-0 w-16 h-16 rounded overflow-hidden cursor-pointer border-2 ${
                      selectedImageId === img.id ? 'border-primary' : 'border-transparent'
                    }`}
                  >
                    <img src={img.data} alt="" className="w-full h-full object-cover" />
                    <div className={`absolute inset-0 flex items-center justify-center ${
                      img.status === 'processing' ? 'bg-black/50' : img.status === 'done' ? 'bg-green-500/20' : img.status === 'error' ? 'bg-red-500/30' : ''
                    }`}>
                      {img.status === 'processing' && <Loader className="w-5 h-5 text-white" />}
                      {img.status === 'done' && <FontIcon type="checkmark-filled" className="w-5 h-5 text-green-500" />}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Image Preview with Overlays */}
          {selectedImage && (
            <div className="bg-card border rounded-lg p-4">
              <div className="relative aspect-video bg-muted rounded overflow-hidden">
                <img ref={imageRef} src={selectedImage.data} alt="" className="max-w-full max-h-full object-contain mx-auto" />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                
                {/* Classification: Show top label as overlay */}
                {mode === 'classify' && classifications.length > 0 && selectedImage.status === 'done' && (
                  <div className="absolute bottom-4 left-4 right-4 bg-black/70 rounded-lg p-3 text-white">
                    <div className="text-lg font-bold">{classifications[0]?.label}</div>
                    <div className="text-sm opacity-80">{(classifications[0]?.confidence * 100).toFixed(1)}% confidence</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Right Column */}
        <div className="space-y-4">
          {/* Results Panel - Different per mode */}
          <div className="bg-card border rounded-lg p-4">
            <h3 className="font-semibold mb-4">
              {mode === 'detect' && 'Detected Objects'}
              {mode === 'classify' && 'Classifications'}
              {mode === 'segment' && 'Segments'}
              {mode === 'ocr' && 'Extracted Text'}
            </h3>
            
            <div className="space-y-2 max-h-[280px] overflow-y-auto">
              {/* DETECTION: List of objects with boxes */}
              {mode === 'detect' && detections.map((det, i) => (
                <div key={i} className="flex items-center gap-3 p-2 bg-muted rounded group" style={{ borderLeft: `4px solid ${COLORS[i % COLORS.length]}` }}>
                  {editingIndex === i ? (
                    <div className="flex-1 flex gap-2">
                      <Input value={editLabel} onChange={e => setEditLabel(e.target.value)} className="h-8" autoFocus 
                        onKeyDown={e => { if (e.key === 'Enter') submitCorrection(det.class_name, editLabel, i); if (e.key === 'Escape') setEditingIndex(null) }} />
                      <Button size="sm" onClick={() => submitCorrection(det.class_name, editLabel, i)}>Save</Button>
                    </div>
                  ) : (
                    <>
                      <span className="font-medium flex-1">{det.class_name}</span>
                      <span className="text-sm text-muted-foreground">{(det.confidence * 100).toFixed(0)}%</span>
                      <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100" onClick={() => { setEditingIndex(i); setEditLabel(det.class_name) }}>
                        <FontIcon type="edit" className="w-4 h-4" />
                      </Button>
                    </>
                  )}
                </div>
              ))}
              
              {/* CLASSIFICATION: Ranked list for whole image */}
              {mode === 'classify' && classifications.map((cls, i) => (
                <div key={i} className="p-2 bg-muted rounded group">
                  {editingIndex === i ? (
                    <div className="flex gap-2">
                      <Input value={editLabel} onChange={e => setEditLabel(e.target.value)} className="h-8" autoFocus
                        onKeyDown={e => { if (e.key === 'Enter') submitCorrection(cls.label, editLabel, i); if (e.key === 'Escape') setEditingIndex(null) }} />
                      <Button size="sm" onClick={() => submitCorrection(cls.label, editLabel, i)}>Save</Button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-bold text-muted-foreground w-6">{i + 1}</span>
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <span className="font-medium">{cls.label}</span>
                          <span className="text-sm text-muted-foreground">{(cls.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 bg-background rounded-full mt-1 overflow-hidden">
                          <div className="h-full bg-primary" style={{ width: `${cls.confidence * 100}%` }} />
                        </div>
                      </div>
                      <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100" onClick={() => { setEditingIndex(i); setEditLabel(cls.label) }}>
                        <FontIcon type="edit" className="w-4 h-4" />
                      </Button>
                    </div>
                  )}
                </div>
              ))}
              
              {/* SEGMENTATION: Similar to detection but with mask indicator */}
              {mode === 'segment' && segments.map((seg, i) => (
                <div key={i} className="flex items-center gap-3 p-2 bg-muted rounded group">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: COLORS[i % COLORS.length] + '80' }} />
                  {editingIndex === i ? (
                    <div className="flex-1 flex gap-2">
                      <Input value={editLabel} onChange={e => setEditLabel(e.target.value)} className="h-8" autoFocus
                        onKeyDown={e => { if (e.key === 'Enter') submitCorrection(seg.class_name, editLabel, i); if (e.key === 'Escape') setEditingIndex(null) }} />
                      <Button size="sm" onClick={() => submitCorrection(seg.class_name, editLabel, i)}>Save</Button>
                    </div>
                  ) : (
                    <>
                      <span className="font-medium flex-1">{seg.class_name}</span>
                      <span className="text-sm text-muted-foreground">{(seg.confidence * 100).toFixed(0)}%</span>
                      <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100" onClick={() => { setEditingIndex(i); setEditLabel(seg.class_name) }}>
                        <FontIcon type="edit" className="w-4 h-4" />
                      </Button>
                    </>
                  )}
                </div>
              ))}
              
              {/* OCR: Show extracted text */}
              {mode === 'ocr' && ocrText && (
                <div className="space-y-3">
                  <div className="p-3 bg-muted rounded font-mono text-sm whitespace-pre-wrap">{ocrText}</div>
                  {ocrRegions.length > 0 && (
                    <div className="text-xs text-muted-foreground">{ocrRegions.length} text regions detected</div>
                  )}
                  <Button size="sm" variant="outline" onClick={() => navigator.clipboard.writeText(ocrText)}>
                    <FontIcon type="copy" className="w-4 h-4 mr-2" />Copy Text
                  </Button>
                </div>
              )}
              
              {/* Empty states */}
              {!selectedImage && <p className="text-muted-foreground text-center py-4">Drop images to analyze</p>}
              {selectedImage?.status === 'pending' && <p className="text-muted-foreground text-center py-4">Click Process All</p>}
              {selectedImage?.status === 'done' && mode === 'detect' && detections.length === 0 && <p className="text-muted-foreground text-center py-4">No objects detected</p>}
              {selectedImage?.status === 'done' && mode === 'classify' && classifications.length === 0 && <p className="text-muted-foreground text-center py-4">No classifications</p>}
              {selectedImage?.status === 'done' && mode === 'segment' && segments.length === 0 && <p className="text-muted-foreground text-center py-4">No segments</p>}
              {selectedImage?.status === 'done' && mode === 'ocr' && !ocrText && <p className="text-muted-foreground text-center py-4">No text found</p>}
            </div>
          </div>
          
          {/* Configuration - Mode-specific */}
          <div className="bg-card border rounded-lg p-4">
            <h3 className="font-semibold mb-4">Configuration</h3>
            
            <div className="space-y-4">
              {/* Model selector - different per mode */}
              <div className="space-y-2">
                <Label>Model</Label>
                <Select 
                  value={currentModel} 
                  onValueChange={v => {
                    if (mode === 'detect') setDetectModel(v)
                    if (mode === 'classify') setClassifyModel(v)
                    if (mode === 'segment') setSegmentModel(v)
                    if (mode === 'ocr') setOcrModel(v)
                  }}
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {MODE_INFO[mode].models.map(m => <SelectItem key={m.id} value={m.id}>{m.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              
              {/* Confidence threshold - detection/segmentation */}
              {(mode === 'detect' || mode === 'segment') && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Confidence</Label>
                    <span className="text-sm text-muted-foreground">{confidenceThreshold.toFixed(2)}</span>
                  </div>
                  <Slider value={[confidenceThreshold]} onValueChange={([v]) => setConfidenceThreshold(v)} min={0.1} max={1} step={0.05} />
                </div>
              )}
              
              {/* Classification config */}
              {mode === 'classify' && (
                <>
                  <div className="space-y-2">
                    <Label>Classes to classify (comma-separated)</Label>
                    <Input 
                      value={classifyClasses} 
                      onChange={e => setClassifyClasses(e.target.value)}
                      placeholder="cat, dog, bird, car..."
                    />
                    <p className="text-xs text-muted-foreground">
                      CLIP will classify the image against these labels
                    </p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Top K Results</Label>
                      <span className="text-sm text-muted-foreground">{topK}</span>
                    </div>
                    <Slider value={[topK]} onValueChange={([v]) => setTopK(v)} min={1} max={10} step={1} />
                  </div>
                </>
              )}
              
              {/* Cascade - detection only */}
              {mode === 'detect' && (
                <div className="flex items-center justify-between">
                  <Label>Cascade Fallback</Label>
                  <Switch checked={enableCascade} onCheckedChange={setEnableCascade} />
                </div>
              )}
            </div>
          </div>
          
          {/* Streaming Session - Detection mode */}
          {mode === 'detect' && (
            <div className="bg-card border rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Streaming Session</h3>
                {isStreaming && <span className="px-2 py-0.5 bg-green-500/20 text-green-500 text-xs rounded-full animate-pulse">Live</span>}
              </div>
              
              <div className="flex gap-2">
                {!isStreaming ? (
                  <Button onClick={startSession} className="flex-1">Start Session</Button>
                ) : (
                  <Button variant="destructive" onClick={stopSession} className="flex-1">Stop</Button>
                )}
              </div>
              
              {stats && (
                <div className="grid grid-cols-4 gap-2 mt-4 text-center text-sm">
                  <div><div className="font-bold">{stats.frames_processed}</div><div className="text-muted-foreground">Frames</div></div>
                  <div><div className="font-bold text-green-500">{stats.actions_triggered}</div><div className="text-muted-foreground">Actions</div></div>
                  <div><div className="font-bold text-yellow-500">{stats.escalations_to_secondary}</div><div className="text-muted-foreground">Escalated</div></div>
                  <div><div className="font-bold text-blue-500">{stats.review_queue_count}</div><div className="text-muted-foreground">Review</div></div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Training Data */}
      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center gap-3 mb-4">
          <h3 className="font-semibold">Training Data</h3>
          <span className="px-2 py-0.5 bg-muted text-xs rounded-full">{replayStats?.size || 0} / {replayStats?.max_size || 1000}</span>
        </div>
        
        <div className="h-2 bg-muted rounded-full overflow-hidden mb-4">
          <div className="h-full bg-primary transition-all" style={{ width: `${((replayStats?.size || 0) / (replayStats?.max_size || 1000)) * 100}%` }} />
        </div>
        
        <div className="grid grid-cols-4 gap-4 text-center text-sm">
          <div><div className="font-bold">{replayStats?.by_source?.escalation || 0}</div><div className="text-muted-foreground">Cascade</div></div>
          <div><div className="font-bold">{replayStats?.by_source?.correction || 0}</div><div className="text-muted-foreground">Corrections</div></div>
          <div><div className="font-bold">{replayStats?.by_source?.review || 0}</div><div className="text-muted-foreground">Review</div></div>
          <div><div className="font-bold">{replayStats?.avg_priority?.toFixed(2) || '0.00'}</div><div className="text-muted-foreground">Priority</div></div>
        </div>
        
        {(replayStats?.size || 0) >= 50 && (
          <div className="flex items-center justify-between mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded">
            <span className="text-green-500 font-medium">Ready for training!</span>
            <Button size="sm">Start Training</Button>
          </div>
        )}
      </div>
    </div>
  )
}
