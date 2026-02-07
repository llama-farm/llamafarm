import { useState, useEffect, useCallback } from 'react'
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

interface TrainingJob {
  job_id: string
  model_id: string
  task: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_epoch?: number
  total_epochs?: number
  metrics?: Record<string, number>
  error?: string
  created_at?: string
  started_at?: string
  completed_at?: string
}

interface AutoTrainStatus {
  enabled: boolean
  threshold: number
  buffer_size: number
  training_eligible: boolean
  last_training_at?: string
  next_eligible_at?: string
  training_in_progress: boolean
  current_job_id?: string
  total_trainings: number
  total_samples_trained: number
}

interface ReplayBufferStats {
  size: number
  max_size: number
  by_source: Record<string, number>
  avg_priority: number
}

export function TrainingPanel() {
  const { toast } = useToast()
  
  // Training jobs
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [isLoadingJobs, setIsLoadingJobs] = useState(true)
  
  // Auto-train status
  const [autoTrainStatus, setAutoTrainStatus] = useState<AutoTrainStatus | null>(null)
  
  // Replay buffer
  const [replayStats, setReplayStats] = useState<ReplayBufferStats | null>(null)
  
  // New training config
  const [newJobModel, setNewJobModel] = useState('yolov8n')
  const [newJobTask, setNewJobTask] = useState('detection')
  const [newJobEpochs, setNewJobEpochs] = useState(10)
  const [newJobBatchSize, setNewJobBatchSize] = useState(16)
  const [useEwc, setUseEwc] = useState(true)
  const [useReplay, setUseReplay] = useState(true)
  const [isStartingJob, setIsStartingJob] = useState(false)
  
  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/train`)
      if (response.ok) {
        const data = await response.json()
        setJobs(data.jobs || [])
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setIsLoadingJobs(false)
    }
  }, [])
  
  // Fetch auto-train status
  const fetchAutoTrainStatus = useCallback(async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/auto-train/status`)
      if (response.ok) {
        const data = await response.json()
        setAutoTrainStatus(data)
      }
    } catch (error) {
      console.error('Failed to fetch auto-train status:', error)
    }
  }, [])
  
  // Fetch replay buffer
  const fetchReplayStats = useCallback(async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/replay-buffer`)
      if (response.ok) {
        const data = await response.json()
        setReplayStats(data.stats)
      }
    } catch (error) {
      console.error('Failed to fetch replay stats:', error)
    }
  }, [])
  
  // Initial fetch and polling
  useEffect(() => {
    fetchJobs()
    fetchAutoTrainStatus()
    fetchReplayStats()
    
    const interval = setInterval(() => {
      fetchJobs()
      fetchAutoTrainStatus()
      fetchReplayStats()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [fetchJobs, fetchAutoTrainStatus, fetchReplayStats])
  
  // Start training
  const startTraining = async () => {
    setIsStartingJob(true)
    
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: newJobModel,
          dataset: 'replay_buffer', // Use replay buffer as dataset
          task: newJobTask,
          config: {
            epochs: newJobEpochs,
            batch_size: newJobBatchSize,
            use_ewc: useEwc,
            use_replay: useReplay,
          },
        }),
      })
      
      if (!response.ok) throw new Error('Failed to start training')
      
      const data = await response.json()
      toast({ message: `Training job ${data.job_id} started` })
      fetchJobs()
    } catch (error) {
      toast({ message: 'Failed to start training', variant: 'destructive' })
    } finally {
      setIsStartingJob(false)
    }
  }
  
  // Trigger auto-training
  const triggerAutoTrain = async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/auto-train/trigger`, {
        method: 'POST',
      })
      
      if (!response.ok) throw new Error('Failed to trigger')
      
      const data = await response.json()
      if (data.triggered) {
        toast({ message: 'Auto-training triggered!' })
      } else {
        toast({ message: 'Not eligible for training yet' })
      }
      fetchAutoTrainStatus()
    } catch (error) {
      toast({ message: 'Failed to trigger auto-training', variant: 'destructive' })
    }
  }
  
  // Cancel job
  const cancelJob = async (jobId: string) => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/train/${jobId}`, {
        method: 'DELETE',
      })
      
      if (response.ok) {
        toast({ message: 'Job cancelled' })
        fetchJobs()
      }
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }
  
  // Clear replay buffer
  const clearReplayBuffer = async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/replay-buffer/clear`, {
        method: 'POST',
      })
      
      if (response.ok) {
        toast({ message: 'Replay buffer cleared' })
        fetchReplayStats()
      }
    } catch (error) {
      toast({ message: 'Failed to clear buffer', variant: 'destructive' })
    }
  }
  
  const runningJobs = jobs.filter(j => j.status === 'running' || j.status === 'queued')
  const completedJobs = jobs.filter(j => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled')
  
  return (
    <div className="space-y-6">
      {/* Auto-Training Status */}
      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <FontIcon type="tools-alt" className="w-5 h-5" />
            Auto-Training
            {autoTrainStatus?.training_in_progress && (
              <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-500 text-xs rounded-full animate-pulse">
                Training...
              </span>
            )}
          </h3>
          <Button 
            size="sm" 
            onClick={triggerAutoTrain}
            disabled={!autoTrainStatus?.training_eligible || autoTrainStatus?.training_in_progress}
          >
            Trigger Now
          </Button>
        </div>
        
        <div className="grid grid-cols-4 gap-4 text-center mb-4">
          <div>
            <div className="text-2xl font-bold">{autoTrainStatus?.buffer_size || 0}</div>
            <div className="text-xs text-muted-foreground">Buffer Size</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{autoTrainStatus?.threshold || 50}</div>
            <div className="text-xs text-muted-foreground">Threshold</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{autoTrainStatus?.total_trainings || 0}</div>
            <div className="text-xs text-muted-foreground">Total Trainings</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{autoTrainStatus?.total_samples_trained || 0}</div>
            <div className="text-xs text-muted-foreground">Samples Trained</div>
          </div>
        </div>
        
        {autoTrainStatus?.training_eligible && !autoTrainStatus?.training_in_progress && (
          <div className="p-3 bg-green-500/10 border border-green-500/20 rounded">
            <div className="flex items-center gap-2 text-green-500">
              <FontIcon type="checkmark-filled" className="w-4 h-4" />
              <span className="font-medium">Ready for training!</span>
              <span className="text-sm text-muted-foreground ml-2">
                Buffer has {autoTrainStatus.buffer_size} samples
              </span>
            </div>
          </div>
        )}
        
        {autoTrainStatus?.last_training_at && (
          <div className="text-sm text-muted-foreground mt-2">
            Last training: {new Date(autoTrainStatus.last_training_at).toLocaleString()}
          </div>
        )}
      </div>
      
      {/* Replay Buffer */}
      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <FontIcon type="data" className="w-5 h-5" />
            Training Data (Replay Buffer)
            <span className="px-2 py-0.5 bg-muted text-xs rounded-full">
              {replayStats?.size || 0} / {replayStats?.max_size || 1000}
            </span>
          </h3>
          <Button size="sm" variant="ghost" onClick={clearReplayBuffer}>
            Clear Buffer
          </Button>
        </div>
        
        <div className="h-3 bg-muted rounded-full overflow-hidden mb-4">
          <div 
            className="h-full bg-primary transition-all"
            style={{ width: `${((replayStats?.size || 0) / (replayStats?.max_size || 1000)) * 100}%` }}
          />
        </div>
        
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-blue-500">{replayStats?.by_source?.escalation || 0}</div>
            <div className="text-xs text-muted-foreground">From Cascade</div>
          </div>
          <div>
            <div className="text-lg font-bold text-green-500">{replayStats?.by_source?.correction || 0}</div>
            <div className="text-xs text-muted-foreground">From Corrections</div>
          </div>
          <div>
            <div className="text-lg font-bold text-yellow-500">{replayStats?.by_source?.review || 0}</div>
            <div className="text-xs text-muted-foreground">From Review</div>
          </div>
          <div>
            <div className="text-lg font-bold">{replayStats?.avg_priority?.toFixed(2) || '0.00'}</div>
            <div className="text-xs text-muted-foreground">Avg Priority</div>
          </div>
        </div>
      </div>
      
      {/* Manual Training */}
      <div className="bg-card border rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <FontIcon type="arrow-right" className="w-5 h-5" />
          Start Training Job
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Base Model</Label>
              <Select value={newJobModel} onValueChange={setNewJobModel}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="yolov8n">YOLOv8 Nano</SelectItem>
                  <SelectItem value="yolov8s">YOLOv8 Small</SelectItem>
                  <SelectItem value="yolov8m">YOLOv8 Medium</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Task</Label>
              <Select value={newJobTask} onValueChange={setNewJobTask}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="detection">Object Detection</SelectItem>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="segmentation">Segmentation</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Epochs</Label>
                <span className="text-sm text-muted-foreground">{newJobEpochs}</span>
              </div>
              <Slider value={[newJobEpochs]} onValueChange={([v]) => setNewJobEpochs(v)} min={1} max={50} />
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Batch Size</Label>
                <span className="text-sm text-muted-foreground">{newJobBatchSize}</span>
              </div>
              <Slider value={[newJobBatchSize]} onValueChange={([v]) => setNewJobBatchSize(v)} min={4} max={64} step={4} />
            </div>
          </div>
        </div>
        
        <div className="flex gap-4 mt-4 pt-4 border-t">
          <div className="flex items-center gap-2">
            <Switch checked={useEwc} onCheckedChange={setUseEwc} />
            <Label className="text-sm">EWC (prevent forgetting)</Label>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={useReplay} onCheckedChange={setUseReplay} />
            <Label className="text-sm">Experience Replay</Label>
          </div>
          <div className="flex-1" />
          <Button onClick={startTraining} disabled={isStartingJob || (replayStats?.size || 0) < 1}>
            {isStartingJob ? <><Loader className="w-4 h-4 mr-2" />Starting...</> : 'Start Training'}
          </Button>
        </div>
      </div>
      
      {/* Running Jobs */}
      {runningJobs.length > 0 && (
        <div className="bg-card border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Running Jobs</h3>
          <div className="space-y-3">
            {runningJobs.map(job => (
              <div key={job.job_id} className="flex items-center gap-4 p-3 bg-muted rounded">
                <Loader className="w-5 h-5" />
                <div className="flex-1">
                  <div className="font-medium">{job.model_id}</div>
                  <div className="text-sm text-muted-foreground">
                    {job.task} • Epoch {job.current_epoch || 0}/{job.total_epochs || '?'}
                  </div>
                </div>
                <div className="w-32">
                  <div className="h-2 bg-background rounded-full overflow-hidden">
                    <div className="h-full bg-primary transition-all" style={{ width: `${job.progress * 100}%` }} />
                  </div>
                  <div className="text-xs text-muted-foreground text-right mt-1">
                    {(job.progress * 100).toFixed(0)}%
                  </div>
                </div>
                <Button size="sm" variant="ghost" onClick={() => cancelJob(job.job_id)}>
                  Cancel
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Job History */}
      <div className="bg-card border rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">Training History</h3>
        
        {isLoadingJobs ? (
          <div className="flex items-center justify-center py-8">
            <Loader className="w-6 h-6" />
          </div>
        ) : completedJobs.length === 0 ? (
          <p className="text-center text-muted-foreground py-8">No training history yet</p>
        ) : (
          <div className="space-y-2">
            {completedJobs.slice(0, 10).map(job => (
              <div key={job.job_id} className="flex items-center gap-4 p-3 bg-muted rounded">
                <FontIcon 
                  type={job.status === 'completed' ? 'checkmark-filled' : 'close'} 
                  className={`w-5 h-5 ${job.status === 'completed' ? 'text-green-500' : 'text-red-500'}`} 
                />
                <div className="flex-1">
                  <div className="font-medium">{job.model_id}</div>
                  <div className="text-sm text-muted-foreground">
                    {job.task} • {job.status}
                    {job.completed_at && ` • ${new Date(job.completed_at).toLocaleDateString()}`}
                  </div>
                </div>
                {job.metrics && (
                  <div className="text-sm">
                    {Object.entries(job.metrics).slice(0, 2).map(([k, v]) => (
                      <span key={k} className="mr-3">{k}: {typeof v === 'number' ? v.toFixed(3) : v}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
