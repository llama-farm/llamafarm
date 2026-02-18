import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Badge } from '../ui/badge'
import { cn } from '@/lib/utils'
import { useClassify } from '../../hooks/useVision'
import { useImageUpload } from './useImageUpload'
import type { ClassifyResponse } from '../../types/vision'

export function ClassifyPanel() {
  const [model, setModel] = useState('')
  const [classInput, setClassInput] = useState('')
  const [classes, setClasses] = useState<string[]>([])
  const [topK, setTopK] = useState(5)
  const [result, setResult] = useState<ClassifyResponse | null>(null)

  const { imageBase64, imagePreview, fileName, handleFileChange } = useImageUpload()
  const classifyMutation = useClassify()

  const addCls = () => {
    const val = classInput.trim()
    if (val && !classes.includes(val)) {
      setClasses(prev => [...prev, val])
      setClassInput('')
    }
  }

  const removeCls = (cls: string) => {
    setClasses(prev => prev.filter(c => c !== cls))
  }

  const handleClassify = () => {
    if (!imageBase64) return
    classifyMutation.mutate(
      {
        image: imageBase64,
        model: model || undefined,
        classes: classes.length > 0 ? classes : undefined,
        top_k: topK,
      },
      { onSuccess: setResult }
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="flex flex-col gap-4">
        <div>
          <Label className="text-sm">Image</Label>
          <Input type="file" accept="image/*" onChange={handleFileChange} className="mt-1" />
          {fileName && <p className="text-xs text-muted-foreground mt-1">{fileName}</p>}
        </div>

        <div>
          <Label className="text-sm">Model (optional)</Label>
          <Input value={model} onChange={e => setModel(e.target.value)} placeholder="default" className="mt-1" />
        </div>

        <div>
          <Label className="text-sm">Classes</Label>
          <div className="flex gap-2 mt-1">
            <Input
              value={classInput}
              onChange={e => setClassInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), addCls())}
              placeholder="Add a class..."
            />
            <Button variant="outline" onClick={addCls} size="sm">Add</Button>
          </div>
          {classes.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {classes.map(cls => (
                <Badge key={cls} variant="secondary" className="cursor-pointer" onClick={() => removeCls(cls)}>
                  {cls} ×
                </Badge>
              ))}
            </div>
          )}
        </div>

        <div>
          <Label className="text-sm">Top K: {topK}</Label>
          <Input
            type="number"
            value={topK}
            onChange={e => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            max={100}
            className="mt-1 w-24"
          />
        </div>

        <Button onClick={handleClassify} disabled={!imageBase64 || classifyMutation.isPending} className="w-full">
          {classifyMutation.isPending ? 'Classifying...' : 'Run Classification'}
        </Button>

        {classifyMutation.isError && (
          <p className="text-sm text-destructive">{(classifyMutation.error as Error).message}</p>
        )}

        {/* Results */}
        {result && (
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm font-medium mb-3">
              Top result: <span className="text-primary">{result.class_name}</span>{' '}
              ({(result.confidence * 100).toFixed(1)}%)
            </p>
            <div className="flex flex-col gap-2">
              {result.all_scores.map((s, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-sm w-24 truncate">{s.class_name}</span>
                  <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all"
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
      </div>

      <div className="flex flex-col items-center justify-start">
        {imagePreview ? (
          <img src={imagePreview} alt="Uploaded" className="max-w-full rounded-lg border border-border" />
        ) : (
          <div className={cn(
            'w-full aspect-video rounded-lg border-2 border-dashed border-border',
            'flex items-center justify-center text-muted-foreground text-sm'
          )}>
            Upload an image to classify
          </div>
        )}
      </div>
    </div>
  )
}
