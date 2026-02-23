import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/collapsible'
import { ChevronDown, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { useStartTraining, useTrainingJobStatus, useSampleDataStatus, useCloneSampleData } from '../../hooks/useVision'
import { Loader2 } from 'lucide-react'
import { PanelIntro } from './PanelIntro'

const SAMPLE_REPO = 'https://github.com/llama-farm/vision-sample-data'
const SAMPLE_CATEGORIES = [
  { id: 'cats', label: '🐱 cats', path: 'vision-sample-data/cats' },
  { id: 'firetrucks', label: '🚒 firetrucks', path: 'vision-sample-data/firetrucks' },
  { id: 'sunflowers', label: '🌻 sunflowers', path: 'vision-sample-data/sunflowers' },
  { id: 'pizza', label: '🍕 pizza', path: 'vision-sample-data/pizza' },
]

const TASK_OPTIONS = [
  { value: 'detection', label: 'Object Detection' },
  { value: 'classification', label: 'Image Classification' },
  { value: 'segmentation', label: 'Instance Segmentation' },
]

export function TrainingPanel() {
  const [modelName, setModelName] = useState('')
  const [task, setTask] = useState('detection')
  const [dataset, setDataset] = useState('')
  const [baseModel, setBaseModel] = useState('yolov8n')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState(0.001)
  const [jobId, setJobId] = useState<string | null>(null)
  const [showDatasetInfo, setShowDatasetInfo] = useState(true)

  const startTraining = useStartTraining()
  const { data: jobStatus } = useTrainingJobStatus(jobId)
  const { data: sampleStatus } = useSampleDataStatus()
  const cloneMutation = useCloneSampleData()

  const handleSampleClick = async (cat: typeof SAMPLE_CATEGORIES[number]) => {
    if (!modelName.trim()) setModelName(cat.id)

    // Sample data has no YOLO labels, so always use classification
    const effectiveTask = 'classification'
    if (task !== effectiveTask) setTask(effectiveTask)

    // If sample data isn't cloned yet, clone first
    if (sampleStatus && !sampleStatus.installed) {
      const result = await cloneMutation.mutateAsync()
      if (result.success) {
        setDataset(effectiveTask === 'classification' ? result.path : `${result.path}/${cat.id}`)
      }
    } else if (sampleStatus?.installed) {
      setDataset(effectiveTask === 'classification' ? sampleStatus.path : `${sampleStatus.path}/${cat.id}`)
    } else {
      setDataset(cat.path)
    }
  }

  const handleTrain = () => {
    if (!modelName.trim() || !dataset.trim()) return
    startTraining.mutate(
      {
        model: modelName.trim(),
        dataset: dataset.trim(),
        task,
        base_model: baseModel,
        config: { epochs, batch_size: batchSize, learning_rate: learningRate },
      },
      { onSuccess: data => setJobId(data.job_id) }
    )
  }

  const isTraining = jobStatus?.status === 'running' || jobStatus?.status === 'pending'
  const isComplete = jobStatus?.status === 'completed'
  const isFailed = jobStatus?.status === 'failed'
  const hasJob = !!jobId

  const handleReset = () => {
    setJobId(null)
    setModelName('')
    setDataset('')
    setShowDatasetInfo(false)
  }

  // When a job is active/complete/failed, show progress view
  if (hasJob && jobStatus) {
    return (
      <div className="max-w-2xl flex flex-col gap-4">
        <PanelIntro title="Train">
          Training a custom model for {task}.
        </PanelIntro>
        <div className="rounded-lg border border-border p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-medium">Training: {modelName || jobStatus.job_id}</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {task} · {dataset.split('/').pop()} dataset
              </p>
            </div>
            <span className={cn(
              'text-xs font-medium px-2.5 py-1 rounded-full',
              isComplete ? 'bg-green-500/15 text-green-700' :
              isFailed ? 'bg-red-500/15 text-red-700' :
              'bg-blue-500/15 text-blue-700'
            )}>
              {jobStatus.status}
            </span>
          </div>

          {jobStatus.progress !== undefined && (
            <div className="mb-3">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>
                  {jobStatus.epoch !== undefined && jobStatus.total_epochs
                    ? `Epoch ${jobStatus.epoch}/${jobStatus.total_epochs}`
                    : 'Progress'}
                </span>
                <span>{Math.round(jobStatus.progress * 100)}%</span>
              </div>
              <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full transition-all duration-300',
                    isFailed ? 'bg-destructive' : 'bg-primary'
                  )}
                  style={{ width: `${Math.max(jobStatus.progress * 100, isFailed ? 100 : 0)}%` }}
                />
              </div>
            </div>
          )}

          {jobStatus.loss !== undefined && (
            <p className="text-xs text-muted-foreground">Loss: {jobStatus.loss.toFixed(4)}</p>
          )}

          {jobStatus.message && (
            <p className="text-xs text-muted-foreground mt-1">{jobStatus.message}</p>
          )}

          {isComplete && (
            <p className="text-sm text-green-700 mt-3">
              ✓ Model saved! Check the <strong>Models</strong> tab to view and manage it.
            </p>
          )}

          {isFailed && (
            <p className="text-sm text-destructive mt-3">
              Training failed. This usually means the dataset format doesn't match the expected structure, or the training backend isn't available on this runtime.
            </p>
          )}
        </div>

        <Button onClick={handleReset} variant="outline" className="w-full">
          {isComplete ? 'Train Another Model' : isFailed ? 'Try Again' : 'Cancel & Start Over'}
        </Button>
      </div>
    )
  }

  return (
    <div className="max-w-2xl flex flex-col gap-4">
      <PanelIntro title="Train">
        Train a custom vision model when the built-in models aren't accurate enough for your specific use case. Start with the Analyze tab to test zero-shot capabilities first — you may not need custom training.
      </PanelIntro>

      {/* Dataset guidance — collapsible, open by default */}
      <Collapsible open={showDatasetInfo} onOpenChange={setShowDatasetInfo}>
        <div className="rounded-lg border border-border bg-muted/20">
          <CollapsibleTrigger className="flex items-center justify-between w-full p-3 text-left">
            <p className="text-sm font-medium">Dataset requirements</p>
            <ChevronDown className="w-4 h-4 text-muted-foreground transition-transform [[data-state=closed]_&]:rotate-[-90deg]" />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-3 pb-3">
              <p className="text-xs text-muted-foreground mb-2">Provide a path to a local folder on this machine containing your training data. Supported formats:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-muted-foreground">
                <div className="rounded border border-border bg-background p-2">
                  <p className="font-medium text-foreground mb-1">YOLO Format</p>
                  <code className="block text-xs">
                    dataset/<br/>
                    ├── images/<br/>
                    │   ├── photo1.jpg<br/>
                    │   └── photo2.jpg<br/>
                    └── labels/<br/>
                    {'    '}├── photo1.txt<br/>
                    {'    '}└── photo2.txt
                  </code>
                  <p className="mt-1">Each .txt: <code>class x_center y_center width height</code> (normalized 0-1)</p>
                </div>
                <div className="rounded border border-border bg-background p-2">
                  <p className="font-medium text-foreground mb-1">COCO Format</p>
                  <code className="block text-xs">
                    dataset/<br/>
                    ├── images/<br/>
                    │   ├── photo1.jpg<br/>
                    │   └── photo2.jpg<br/>
                    └── annotations.json
                  </code>
                  <p className="mt-1">Standard COCO JSON with categories, images, and annotations arrays.</p>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label className="text-sm">Model Name</Label>
          <Input
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            placeholder="my-detector"
            className="mt-1"
            disabled={isTraining}
          />
        </div>
        <div>
          <Label className="text-sm">Base Model</Label>
          <Select value={baseModel} onValueChange={setBaseModel} disabled={isTraining}>
            <SelectTrigger className="mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="yolov8n">YOLOv8 Nano (fastest)</SelectItem>
              <SelectItem value="yolov8s">YOLOv8 Small</SelectItem>
              <SelectItem value="yolov8m">YOLOv8 Medium</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-sm">Task</Label>
          <Select value={task} onValueChange={setTask} disabled={isTraining}>
            <SelectTrigger className="mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TASK_OPTIONS.map(opt => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div>
        <Label className="text-sm">Dataset Path</Label>
        <Input
          value={dataset}
          onChange={e => setDataset(e.target.value)}
          placeholder="/path/to/dataset or dataset name"
          className="mt-1"
          disabled={isTraining}
        />
        <div className="mt-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1.5">
            <span>Try sample data:</span>
            <a
              href={SAMPLE_REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              vision-sample-data <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <div className="flex flex-wrap gap-1.5 items-center">
            {SAMPLE_CATEGORIES.map(cat => (
              <button
                key={cat.id}
                type="button"
                onClick={() => handleSampleClick(cat)}
                disabled={isTraining || cloneMutation.isPending}
                className={cn(
                  'px-2.5 py-1 rounded-full border text-xs transition-colors',
                  dataset.endsWith(`/${cat.id}`)
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border bg-background text-muted-foreground hover:border-primary/50 hover:text-foreground'
                )}
              >
                {cat.label}
              </button>
            ))}
            {cloneMutation.isPending && (
              <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                <Loader2 className="w-3 h-3 animate-spin" /> Downloading sample data…
              </span>
            )}
          </div>
          {cloneMutation.isError && (
            <p className="text-xs text-destructive mt-1">
              Failed to download sample data. Check your network connection and try again.
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <Label className="text-sm">Epochs</Label>
          <Input
            type="number"
            value={epochs}
            onChange={e => setEpochs(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            className="mt-1"
            disabled={isTraining}
          />
        </div>
        <div>
          <Label className="text-sm">Batch Size</Label>
          <Input
            type="number"
            value={batchSize}
            onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            className="mt-1"
            disabled={isTraining}
          />
        </div>
        <div>
          <Label className="text-sm">Learning Rate</Label>
          <Input
            type="number"
            value={learningRate}
            onChange={e => setLearningRate(parseFloat(e.target.value) || 0.001)}
            step={0.0001}
            min={0}
            className="mt-1"
            disabled={isTraining}
          />
        </div>
      </div>

      <Button
        onClick={handleTrain}
        disabled={!modelName.trim() || !dataset.trim() || isTraining || startTraining.isPending}
        className="w-full"
      >
        {startTraining.isPending ? 'Starting...' : isTraining ? 'Training in progress...' : 'Start Training'}
      </Button>

      {startTraining.isError && (
        <p className="text-sm text-destructive">Failed to start training. Check that the dataset path is valid and the runtime is running.</p>
      )}
    </div>
  )
}
