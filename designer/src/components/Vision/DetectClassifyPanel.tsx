import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Slider } from '../ui/slider'
import { cn } from '@/lib/utils'
import { useDetectClassify } from '../../hooks/useVision'
import { useImageUpload } from './useImageUpload'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import type { DetectClassifyResult, Detection } from '../../types/vision'

export function DetectClassifyPanel() {
  const [confidence, setConfidence] = useState(0.5)
  const [results, setResults] = useState<DetectClassifyResult[]>([])

  const { imageBase64, imagePreview, fileName, handleFileChange: onFileChange } = useImageUpload()
  const mutation = useDetectClassify()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setResults([])
    onFileChange(e)
  }

  const handleRun = () => {
    if (!imageBase64) return
    mutation.mutate(
      { image: imageBase64, confidence_threshold: confidence },
      { onSuccess: data => setResults(data.results) }
    )
  }

  // Convert results to detections for bounding box overlay
  const detections: Detection[] = results.map(r => ({
    ...r.detection,
    class_name: `${r.detection.class_name} → ${r.classification.class_name}`,
  }))

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">Detect objects first, then classify each detected region. Great for identifying specific types within broader categories — for example, detecting birds then classifying each by species.</p>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="flex flex-col gap-4">
        <div>
          <Label className="text-sm">Image</Label>
          <Input type="file" accept="image/*" onChange={handleFileChange} className="mt-1" />
          {fileName && <p className="text-xs text-muted-foreground mt-1">{fileName}</p>}
        </div>

        <div>
          <Label className="text-sm">Confidence: {(confidence * 100).toFixed(0)}%</Label>
          <Slider value={[confidence]} onValueChange={([v]) => setConfidence(v)} min={0} max={1} step={0.05} className="mt-2" />
        </div>

        <Button onClick={handleRun} disabled={!imageBase64 || mutation.isPending} className="w-full">
          {mutation.isPending ? 'Processing...' : 'Detect + Classify'}
        </Button>

        {mutation.isError && (
          <p className="text-sm text-destructive">{(mutation.error as Error).message}</p>
        )}

        {results.length > 0 && (
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm font-medium mb-2">
              {results.length} object{results.length !== 1 ? 's' : ''} found
            </p>
            <div className="flex flex-col gap-2">
              {results.map((r, i) => (
                <div key={i} className="rounded-md border border-border p-2 text-sm">
                  <div className="flex justify-between">
                    <span className="font-medium">{r.detection.class_name}</span>
                    <span className="text-muted-foreground">
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

      <div className="flex flex-col items-center justify-start">
        {imagePreview && detections.length > 0 ? (
          <BoundingBoxCanvas imageSrc={imagePreview} detections={detections} className="rounded-lg border border-border" />
        ) : imagePreview ? (
          <img src={imagePreview} alt="Uploaded" className="max-w-full rounded-lg border border-border" />
        ) : (
          <div className={cn(
            'w-full aspect-video rounded-lg border-2 border-dashed border-border',
            'flex items-center justify-center text-muted-foreground text-sm'
          )}>
            Upload an image to detect and classify
          </div>
        )}
      </div>
    </div>
    </div>
  )
}
