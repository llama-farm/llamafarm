import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { cn } from '@/lib/utils'
import { useDetect } from '../../hooks/useVision'
import { useImageUpload } from './useImageUpload'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import type { Detection } from '../../types/vision'

export function DetectPanel() {
  const [model, setModel] = useState('')
  const [confidence, setConfidence] = useState(0.5)
  const [classesInput, setClassesInput] = useState('')
  const [detections, setDetections] = useState<Detection[]>([])

  const { imageBase64, imagePreview, fileName, handleFileChange } = useImageUpload()
  const detectMutation = useDetect()

  const handleDetect = () => {
    if (!imageBase64) return
    const classes = classesInput.trim()
      ? classesInput.split(',').map(c => c.trim()).filter(Boolean)
      : undefined

    detectMutation.mutate(
      {
        image: imageBase64,
        model: model || undefined,
        confidence_threshold: confidence,
        classes,
      },
      {
        onSuccess: data => setDetections(data.detections),
      }
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Controls */}
      <div className="flex flex-col gap-4">
        <div>
          <Label className="text-sm">Image</Label>
          <Input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="mt-1"
          />
          {fileName && (
            <p className="text-xs text-muted-foreground mt-1">{fileName}</p>
          )}
        </div>

        <div>
          <Label className="text-sm">Model (optional)</Label>
          <Input
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder="default"
            className="mt-1"
          />
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

        <div>
          <Label className="text-sm">Filter Classes (comma-separated)</Label>
          <Input
            value={classesInput}
            onChange={e => setClassesInput(e.target.value)}
            placeholder="person, car, dog"
            className="mt-1"
          />
        </div>

        <Button
          onClick={handleDetect}
          disabled={!imageBase64 || detectMutation.isPending}
          className="w-full"
        >
          {detectMutation.isPending ? 'Detecting...' : 'Run Detection'}
        </Button>

        {detectMutation.isError && (
          <p className="text-sm text-destructive">
            {(detectMutation.error as Error).message}
          </p>
        )}

        {/* Results list */}
        {detections.length > 0 && (
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm font-medium mb-2">
              {detections.length} detection{detections.length !== 1 ? 's' : ''} found
            </p>
            <div className="flex flex-col gap-1">
              {detections.map((d, i) => (
                <div key={i} className="flex items-center justify-between text-sm">
                  <span className="font-medium">{d.class_name}</span>
                  <span className="text-muted-foreground">
                    {(d.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Preview / Results */}
      <div className="flex flex-col items-center justify-start">
        {imagePreview && detections.length > 0 ? (
          <BoundingBoxCanvas
            imageSrc={imagePreview}
            detections={detections}
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
  )
}
