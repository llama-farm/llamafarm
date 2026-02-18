import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { useStartTraining, useTrainingJobStatus } from '../../hooks/useVision'

const TASK_OPTIONS = [
  { value: 'detection', label: 'Object Detection' },
  { value: 'classification', label: 'Image Classification' },
  { value: 'segmentation', label: 'Instance Segmentation' },
]

export function TrainingPanel() {
  const [modelName, setModelName] = useState('')
  const [task, setTask] = useState('detection')
  const [dataset, setDataset] = useState('')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState(0.001)
  const [jobId, setJobId] = useState<string | null>(null)

  const startTraining = useStartTraining()
  const { data: jobStatus } = useTrainingJobStatus(jobId)

  const handleTrain = () => {
    if (!modelName.trim() || !dataset.trim()) return
    startTraining.mutate(
      {
        model: modelName.trim(),
        dataset: dataset.trim(),
        task,
        config: { epochs, batch_size: batchSize, learning_rate: learningRate },
      },
      { onSuccess: data => setJobId(data.job_id) }
    )
  }

  const isTraining = jobStatus?.status === 'running' || jobStatus?.status === 'pending'
  const isComplete = jobStatus?.status === 'completed'
  const isFailed = jobStatus?.status === 'failed'

  return (
    <div className="max-w-2xl flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">
        Train a custom vision model when the built-in models aren't accurate enough for your specific use case. Start with the Analyze tab to test zero-shot capabilities first — you may not need custom training.
      </p>

      {/* Dataset guidance */}
      <div className="rounded-lg border border-border bg-muted/20 p-3">
        <p className="text-sm font-medium mb-2">Dataset requirements</p>
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

      <div className="grid grid-cols-2 gap-4">
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
        <p className="text-sm text-destructive">{(startTraining.error as Error).message}</p>
      )}

      {/* Training progress */}
      {jobStatus && (
        <div className="rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Training Job: {jobStatus.job_id}</span>
            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
              isComplete ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
              isFailed ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
              'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
            }`}>
              {jobStatus.status}
            </span>
          </div>

          {jobStatus.progress !== undefined && (
            <div className="mb-2">
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
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${jobStatus.progress * 100}%` }}
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
        </div>
      )}
    </div>
  )
}
