import { useState, useEffect, useCallback } from 'react'
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
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'

const RUNTIME_URL = ''

interface SavedModel {
  name: string
  task: string
  path: string
  base_model?: string
  classes?: string[]
  num_classes?: number
  size_mb?: number
  modified_at?: number
  created_at?: string
}

export function ModelsPanel() {
  const { toast } = useToast()
  
  const [models, setModels] = useState<SavedModel[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [taskFilter, setTaskFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  
  // Load model dialog
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  
  // Fetch models
  const fetchModels = useCallback(async () => {
    try {
      const url = taskFilter === 'all' 
        ? `${RUNTIME_URL}/v1/vision/models`
        : `${RUNTIME_URL}/v1/vision/models?task=${taskFilter}`
      
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setModels(data.models || [])
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setIsLoading(false)
    }
  }, [taskFilter])
  
  useEffect(() => {
    fetchModels()
  }, [fetchModels])
  
  // Load model
  const loadModel = async (model: SavedModel) => {
    setLoadingModel(model.name)
    
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: model.name,
          task: model.task,
        }),
      })
      
      if (!response.ok) throw new Error('Failed to load model')
      
      toast({ message: `Model "${model.name}" loaded for inference` })
    } catch (error) {
      toast({ message: 'Failed to load model', variant: 'destructive' })
    } finally {
      setLoadingModel(null)
    }
  }
  
  // Delete model
  const deleteModel = async (model: SavedModel) => {
    if (!confirm(`Delete model "${model.name}"? This cannot be undone.`)) return
    
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/models/${model.task}/${model.name}`, {
        method: 'DELETE',
      })
      
      if (!response.ok) throw new Error('Failed to delete')
      
      toast({ message: `Model "${model.name}" deleted` })
      fetchModels()
    } catch (error) {
      toast({ message: 'Failed to delete model', variant: 'destructive' })
    }
  }
  
  // Filter models
  const filteredModels = models.filter(m => {
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      if (!m.name.toLowerCase().includes(q) && !m.task.toLowerCase().includes(q)) {
        return false
      }
    }
    return true
  })
  
  // Group by task
  const groupedModels: Record<string, SavedModel[]> = {}
  filteredModels.forEach(m => {
    if (!groupedModels[m.task]) groupedModels[m.task] = []
    groupedModels[m.task].push(m)
  })
  
  const taskColors: Record<string, string> = {
    detection: 'bg-blue-500',
    classification: 'bg-green-500',
    segmentation: 'bg-purple-500',
  }
  
  return (
    <div className="space-y-6">
      {/* Header & Filters */}
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <Input
            placeholder="Search models..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="max-w-xs"
          />
        </div>
        
        <Select value={taskFilter} onValueChange={setTaskFilter}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by task" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Tasks</SelectItem>
            <SelectItem value="detection">Detection</SelectItem>
            <SelectItem value="classification">Classification</SelectItem>
            <SelectItem value="segmentation">Segmentation</SelectItem>
          </SelectContent>
        </Select>
        
        <Button variant="outline" onClick={fetchModels}>
          <FontIcon type="recently-viewed" className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>
      
      {/* Models List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader className="w-8 h-8" />
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="bg-card border rounded-lg p-8 text-center">
          <FontIcon type="model" className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-lg font-semibold mb-2">No Models Found</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery 
              ? 'No models match your search.' 
              : 'Train a model to see it here. Models are automatically saved after training.'}
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedModels).map(([task, taskModels]) => (
            <div key={task} className="space-y-3">
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${taskColors[task] || 'bg-gray-500'}`} />
                {task} ({taskModels.length})
              </h3>
              
              <div className="grid gap-3">
                {taskModels.map(model => (
                  <div 
                    key={`${model.task}-${model.name}`} 
                    className="bg-card border rounded-lg p-4 flex items-center gap-4 group"
                  >
                    {/* Icon */}
                    <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${taskColors[model.task] || 'bg-gray-500'} bg-opacity-20`}>
                      <FontIcon type="model" className="w-6 h-6" />
                    </div>
                    
                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold truncate">{model.name}</div>
                      <div className="text-sm text-muted-foreground flex items-center gap-3">
                        {model.base_model && <span>Base: {model.base_model}</span>}
                        {model.num_classes && <span>{model.num_classes} classes</span>}
                        {model.size_mb && <span>{model.size_mb} MB</span>}
                      </div>
                      {model.classes && model.classes.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {model.classes.slice(0, 5).map(cls => (
                            <span key={cls} className="px-1.5 py-0.5 bg-muted text-xs rounded">
                              {cls}
                            </span>
                          ))}
                          {model.classes.length > 5 && (
                            <span className="px-1.5 py-0.5 bg-muted text-xs rounded">
                              +{model.classes.length - 5} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    
                    {/* Modified date */}
                    {model.modified_at && (
                      <div className="text-sm text-muted-foreground">
                        {new Date(model.modified_at * 1000).toLocaleDateString()}
                      </div>
                    )}
                    
                    {/* Actions */}
                    <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button 
                        size="sm" 
                        onClick={() => loadModel(model)}
                        disabled={loadingModel === model.name}
                      >
                        {loadingModel === model.name ? (
                          <Loader className="w-4 h-4" />
                        ) : (
                          <>
                            <FontIcon type="arrow-right" className="w-4 h-4 mr-1" />
                            Load
                          </>
                        )}
                      </Button>
                      <Button 
                        size="sm" 
                        variant="ghost"
                        onClick={() => deleteModel(model)}
                      >
                        <FontIcon type="trashcan" className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* How Models Are Saved */}
      <div className="bg-muted/50 border border-dashed rounded-lg p-6">
        <h3 className="font-semibold mb-2">How Models Are Saved</h3>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• <strong>Auto-save:</strong> Models are automatically saved after training completes</li>
          <li>• <strong>Incremental:</strong> Fine-tuned models build on base models (YOLOv8, CLIP)</li>
          <li>• <strong>EWC:</strong> Elastic Weight Consolidation prevents forgetting previous knowledge</li>
          <li>• <strong>Classes:</strong> Custom class names are stored with the model</li>
        </ul>
        <div className="mt-4 p-3 bg-background rounded text-sm">
          <div className="font-mono text-xs text-muted-foreground mb-1">Example: Using a saved model</div>
          <code className="text-xs">
            POST /v1/vision/detect {`{ "model": "my-custom-detector", "image": "..." }`}
          </code>
        </div>
      </div>
    </div>
  )
}
