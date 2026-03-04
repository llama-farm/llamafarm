import { useState } from 'react'
import { PanelIntro } from './PanelIntro'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import { useVisionModels, useLoadVisionModel, useExportVisionModel, useDeleteVisionModel } from '../../hooks/useVision'

const BUILTIN_MODELS = [
  {
    name: 'yolov8n',
    displayName: 'YOLOv8 Nano',
    description: 'Pre-trained object detection — 80 classes (person, car, dog, etc.). Fastest, ideal for real-time.',
    task: 'detection',
    builtin: true,
    size: '~6 MB',
  },
  {
    name: 'yolov8s',
    displayName: 'YOLOv8 Small',
    description: 'Pre-trained object detection — 80 classes. Good balance of speed and accuracy.',
    task: 'detection',
    builtin: true,
    size: '~22 MB',
  },
  {
    name: 'yolov8m',
    displayName: 'YOLOv8 Medium',
    description: 'Pre-trained object detection — 80 classes. Higher accuracy, more compute.',
    task: 'detection',
    builtin: true,
    size: '~52 MB',
  },
  {
    name: 'clip-vit-base',
    displayName: 'CLIP ViT-Base',
    description: 'Zero-shot image classification — understands natural language labels without training. Trained on 400M image-text pairs.',
    task: 'classification',
    builtin: true,
    size: '~350 MB',
  },
]

export function ModelsPanel({ onNavigateToTrain }: { onNavigateToTrain?: () => void } = {}) {
  const [search, setSearch] = useState('')
  const { data, isLoading, error } = useVisionModels()
  const loadModel = useLoadVisionModel()
  const exportModel = useExportVisionModel()
  const deleteModel = useDeleteVisionModel()
  const { toast } = useToast()

  const customModels = data?.models ?? []

  const handleLoad = (name: string) => {
    loadModel.mutate(
      { name },
      {
        onSuccess: () => toast({ message: `Model "${name}" loaded` }),
        onError: (err: Error) => toast({ message: err.message }),
      }
    )
  }

  const handleDelete = (name: string) => {
    deleteModel.mutate(name, {
      onSuccess: () => toast({ message: `Model "${name}" deleted` }),
      onError: (err: Error) => toast({ message: err.message }),
    })
  }

  const handleExport = (name: string) => {
    exportModel.mutate(
      { name },
      {
        onSuccess: (blob) => {
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `${name}.zip`
          a.click()
          setTimeout(() => URL.revokeObjectURL(url), 30_000)
          toast({ message: `Model "${name}" exported` })
        },
        onError: (err: Error) => toast({ message: err.message }),
      }
    )
  }

  const filteredBuiltin = BUILTIN_MODELS.filter(m =>
    m.displayName.toLowerCase().includes(search.toLowerCase()) ||
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    m.description.toLowerCase().includes(search.toLowerCase())
  )

  const filteredCustom = customModels.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="flex flex-col gap-4">
      <PanelIntro title="Models">
        Models available for vision tasks. Built-in models are downloaded automatically on first use. Train custom models for domain-specific accuracy.
      </PanelIntro>

      <div className="relative">
        <FontIcon type="search" className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
        <Input
          placeholder="Search models..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Built-in Models */}
      {filteredBuiltin.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2">Built-in Models</h3>
          <p className="text-xs text-muted-foreground mb-3">These come with the Vision add-on. Downloaded automatically on first use — no setup needed.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {filteredBuiltin.map(model => (
              <div key={model.name} className="rounded-lg border border-border p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{model.displayName}</span>
                      <Badge variant="secondary" className="text-xs">{model.task}</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">{model.description}</p>
                    <p className="text-xs text-muted-foreground mt-1">Size: {model.size} · ID: <code className="text-xs">{model.name}</code></p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Custom Models */}
      <div>
        <h3 className="text-sm font-medium mb-2">Custom Models</h3>
        {isLoading ? (
          <div className="text-sm text-muted-foreground py-4 text-center">Loading...</div>
        ) : (error || filteredCustom.length === 0) && customModels.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-6 text-center">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-muted">
              <span className="text-lg">🧠</span>
            </div>
            <p className="text-sm font-medium text-foreground mb-1">No custom models yet</p>
            <p className="text-sm text-muted-foreground mb-4">
              Train a model on your own data to create one tailored to your specific use case.
            </p>
            {onNavigateToTrain && (
              <Button variant="outline" size="sm" onClick={onNavigateToTrain}>
                Go to Train
              </Button>
            )}
          </div>
        ) : filteredCustom.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-4 text-center">
            <p className="text-sm text-muted-foreground">No models match your search.</p>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            {filteredCustom.map(model => (
              <div
                key={model.name}
                className="flex items-center justify-between px-3 py-3 text-sm border-b last:border-b-0 border-border hover:bg-accent/40"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">{model.name}</span>
                  {model.task && <Badge variant="secondary" className="text-xs">{model.task}</Badge>}
                  {model.size_mb != null && (
                    <span className="text-xs text-muted-foreground">{model.size_mb.toFixed(1)} MB</span>
                  )}
                  {model.created_at && (
                    <span className="text-xs text-muted-foreground">{new Date(model.created_at).toLocaleDateString()}</span>
                  )}
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30">
                      <FontIcon type="overflow" className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => handleLoad(model.name)}>Load</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExport(model.name)}>Export</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleDelete(model.name)} className="text-destructive">Remove</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
