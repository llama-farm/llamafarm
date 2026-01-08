import { useState, useEffect, useCallback, useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { Select } from '../ui/select'
import { useToast } from '../ui/toast'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '../ui/collapsible'
import FontIcon from '../../common/FontIcon'
import TrainingLoadingOverlay from './TrainingLoadingOverlay'
import {
  useListRouterModels,
  useTrainRouter,
  useRouteQuery,
  useDeleteRouterModel,
  useGenerateRouterData,
} from '../../hooks/useMLModels'
import { useCachedModels } from '../../hooks/useModels'
import { useProjectModels } from '../../hooks/useProjectModels'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import {
  ROUTER_EMBEDDER_OPTIONS,
  DEFAULT_GENERATION_MODEL,
  type RouterModelInfo,
  type RouterRoute,
} from '../../types/ml'

type TrainingState = 'idle' | 'training' | 'success' | 'error'

// Default embedder model
const DEFAULT_EMBEDDER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

// Route configuration for the form
interface RouteFormData {
  id: string
  name: string
  targetModel: string
  description: string
  utterances: string[]
}

// Test result type
interface RouterTestResult {
  id: string
  query: string
  routeName: string | null
  targetModel: string
  similarityScore: number
  matchedUtterance: string | null
  timestamp: string
}

// Generate unique ID for routes
function generateRouteId(): string {
  return `route_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// Create empty route
function createEmptyRoute(): RouteFormData {
  return {
    id: generateRouteId(),
    name: '',
    targetModel: '',
    description: '',
    utterances: [],
  }
}

function RouterModel() {
  const navigate = useNavigate()
  const { id: routerId } = useParams<{ id: string }>()
  const { toast } = useToast()
  const isNewRouter = routerId === 'new' || !routerId

  // Router configuration state
  const [routerName, setRouterName] = useState('')
  const [embedderModel, setEmbedderModel] = useState(DEFAULT_EMBEDDER_MODEL)
  const [defaultModel, setDefaultModel] = useState('')
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7)
  const [routes, setRoutes] = useState<RouteFormData[]>([createEmptyRoute()])

  // UI state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [testQuery, setTestQuery] = useState('')
  const [testResults, setTestResults] = useState<RouterTestResult[]>([])
  const [expandedRoutes, setExpandedRoutes] = useState<Set<string>>(new Set())
  const [generatingForRoute, setGeneratingForRoute] = useState<string | null>(null)

  // Generation settings state (defaults to local model - no API key needed)
  const [generationModel, setGenerationModel] = useState(DEFAULT_GENERATION_MODEL)
  const [generationCount, setGenerationCount] = useState(20)
  const [generationComplexity, setGenerationComplexity] = useState<'simple' | 'complex' | 'mixed'>('mixed')
  const [generationStyle, setGenerationStyle] = useState('')

  // Get current project for project models
  const activeProject = useActiveProject()
  const namespace = activeProject?.namespace
  const projectId = activeProject?.project

  // Fetch project config to get router definitions
  const { data: projectDetail } = useProject(
    namespace || '',
    projectId || '',
    !!namespace && !!projectId
  )

  // API hooks
  const { data: routerModelsData, refetch: refetchRouters } = useListRouterModels()
  const trainRouterMutation = useTrainRouter()
  const routeQueryMutation = useRouteQuery()
  const deleteRouterMutation = useDeleteRouterModel()
  const generateDataMutation = useGenerateRouterData()
  const updateProjectMutation = useUpdateProject()

  // Fetch available models
  const { data: cachedModelsData } = useCachedModels()
  const { data: projectModelsData } = useProjectModels(namespace, projectId)

  // Extract routers from project config (llamafarm.yaml)
  const configRouters = useMemo(() => {
    const runtimeCfg = (projectDetail as any)?.project?.config?.runtime || {}
    const models = runtimeCfg?.models || []
    return models.filter((m: any) => m.provider === 'router')
  }, [projectDetail])

  // Get list of available models for generation (from disk/Universal Runtime)
  const availableGenerationModels = useMemo(() => {
    const models = cachedModelsData?.data || []
    // Add the default model if not already present
    const modelIds = models.map(m => m.id)
    if (!modelIds.includes(DEFAULT_GENERATION_MODEL)) {
      return [{ id: DEFAULT_GENERATION_MODEL, name: 'Qwen3-1.7B (Default)' }, ...models]
    }
    return models
  }, [cachedModelsData])

  // Get list of available target models (from project config)
  const availableTargetModels = useMemo(() => {
    const models = projectModelsData?.models || []
    return models.map(m => ({
      name: (m as any).name || m.model,
      model: m.model,
    }))
  }, [projectModelsData])

  // Get existing router names for validation
  const existingRouterNames = useMemo(() => {
    return new Set(routerModelsData?.data?.map((r: RouterModelInfo) => r.name) || [])
  }, [routerModelsData])

  // Load existing router if editing - check both trained models and config
  useEffect(() => {
    if (!isNewRouter && routerId) {
      // First, check if it's a trained router from the API
      const trainedRouter = routerModelsData?.data?.find(
        (r: RouterModelInfo) => r.name === routerId
      )

      // Then check if it's defined in the config
      const configRouter = configRouters.find((r: any) => r.name === routerId)

      if (configRouter) {
        // Load from config (source of truth for project-specific routers)
        setRouterName(configRouter.name)
        setEmbedderModel(configRouter.embedder_model || DEFAULT_EMBEDDER_MODEL)
        setDefaultModel(configRouter.default_model || '')
        setSimilarityThreshold(configRouter.similarity_threshold || 0.7)

        // Load routes with full data from config
        if (configRouter.routes && configRouter.routes.length > 0) {
          setRoutes(
            configRouter.routes.map((route: any) => ({
              id: generateRouteId(),
              name: route.name || '',
              targetModel: route.target_model || '',
              description: route.description || '',
              utterances: route.utterances || [],
            }))
          )
        }
      } else if (trainedRouter) {
        // Fallback to trained router data (for routers not in config)
        setRouterName(trainedRouter.name)
        setEmbedderModel(trainedRouter.embedder_model || DEFAULT_EMBEDDER_MODEL)
        setDefaultModel(trainedRouter.default_model || '')
        setSimilarityThreshold(trainedRouter.similarity_threshold || 0.7)
        // Check if we have full route data (from project-specific routers)
        if (trainedRouter.routeData && trainedRouter.routeData.length > 0) {
          setRoutes(
            trainedRouter.routeData.map((route: any) => ({
              id: generateRouteId(),
              name: route.name || '',
              targetModel: route.target_model || '',
              description: route.description || '',
              utterances: route.utterances || [],
            }))
          )
        } else if (trainedRouter.routes && trainedRouter.routes.length > 0) {
          // For global/legacy routers, we only have route names
          setRoutes(
            trainedRouter.routes.map((routeName: string) => ({
              id: generateRouteId(),
              name: routeName,
              targetModel: '',
              description: '',
              utterances: [],
            }))
          )
        }
      }
    }
  }, [isNewRouter, routerId, routerModelsData, configRouters])

  // Expand first route by default
  useEffect(() => {
    if (routes.length > 0 && expandedRoutes.size === 0) {
      setExpandedRoutes(new Set([routes[0].id]))
    }
  }, [routes, expandedRoutes.size])

  // Toggle route expansion
  const toggleRouteExpanded = useCallback((routeId: string) => {
    setExpandedRoutes(prev => {
      const next = new Set(prev)
      if (next.has(routeId)) {
        next.delete(routeId)
      } else {
        next.add(routeId)
      }
      return next
    })
  }, [])

  // Add new route
  const handleAddRoute = useCallback(() => {
    const newRoute = createEmptyRoute()
    setRoutes(prev => [...prev, newRoute])
    setExpandedRoutes(prev => new Set([...prev, newRoute.id]))
  }, [])

  // Remove route
  const handleRemoveRoute = useCallback((routeId: string) => {
    setRoutes(prev => prev.filter(r => r.id !== routeId))
    setExpandedRoutes(prev => {
      const next = new Set(prev)
      next.delete(routeId)
      return next
    })
  }, [])

  // Update route field
  const handleRouteChange = useCallback(
    (routeId: string, field: keyof RouteFormData, value: string | string[]) => {
      setRoutes(prev =>
        prev.map(route =>
          route.id === routeId ? { ...route, [field]: value } : route
        )
      )
    },
    []
  )

  // Parse utterances from textarea
  const parseUtterances = (text: string): string[] => {
    return text
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
  }

  // Format utterances for textarea
  const formatUtterances = (utterances: string[]): string => {
    return utterances.join('\n')
  }

  // Generate synthetic data for a route using the selected generation model
  const handleGenerateData = useCallback(
    async (routeId: string) => {
      const route = routes.find(r => r.id === routeId)
      if (!route || !route.description) {
        toast({
          message: 'Please add a description for the route first',
          variant: 'destructive',
        })
        return
      }

      setGeneratingForRoute(routeId)
      try {
        const result = await generateDataMutation.mutateAsync({
          route_description: route.description,
          count: generationCount,
          complexity: generationComplexity,
          style: generationStyle || undefined,
          model: generationModel,
        })

        if ('utterances' in result && result.utterances) {
          // Merge with existing utterances, avoiding duplicates
          const existingSet = new Set(route.utterances)
          const newUtterances = result.utterances.filter(
            (u: string) => !existingSet.has(u)
          )
          handleRouteChange(routeId, 'utterances', [
            ...route.utterances,
            ...newUtterances,
          ])
          toast({
            message: `Generated ${newUtterances.length} new ${generationComplexity} utterances`,
            icon: 'checkmark-filled',
          })
        }
      } catch (error) {
        console.error('Failed to generate data:', error)
        toast({
          message: 'Failed to generate data. Make sure the local model is running.',
          variant: 'destructive',
        })
      } finally {
        setGeneratingForRoute(null)
      }
    },
    [routes, generateDataMutation, handleRouteChange, toast, generationModel, generationCount, generationComplexity, generationStyle]
  )

  // Validate router configuration
  const validateRouter = useCallback((): string | null => {
    if (!routerName.trim()) {
      return 'Router name is required'
    }
    if (isNewRouter && existingRouterNames.has(routerName.trim())) {
      return 'A router with this name already exists'
    }
    if (!defaultModel.trim()) {
      return 'Default model is required'
    }
    if (routes.length === 0) {
      return 'At least one route is required'
    }
    for (const route of routes) {
      if (!route.name.trim()) {
        return 'All routes must have a name'
      }
      if (!route.targetModel.trim()) {
        return `Route "${route.name}" needs a target model`
      }
      if (route.utterances.length === 0) {
        return `Route "${route.name}" needs at least one utterance`
      }
    }
    return null
  }, [routerName, defaultModel, routes, isNewRouter, existingRouterNames])

  // Train router
  const handleTrain = useCallback(async () => {
    const error = validateRouter()
    if (error) {
      toast({ message: error, variant: 'destructive' })
      return
    }

    setTrainingState('training')
    try {
      // Convert routes to API format
      const routesData: RouterRoute[] = routes.map(route => ({
        name: route.name.trim(),
        target_model: route.targetModel.trim(),
        description: route.description.trim() || undefined,
        utterances: route.utterances,
      }))

      await trainRouterMutation.mutateAsync({
        model: routerName.trim(),
        embedder_model: embedderModel,
        default_model: defaultModel.trim(),
        similarity_threshold: similarityThreshold,
        routes: routesData,
        namespace: namespace || 'default',
        project_id: projectId || 'default',
      })

      // Add router to project config for governance
      if (namespace && projectId && projectDetail) {
        try {
          const currentConfig = (projectDetail as any)?.project?.config || {}
          const currentModels = currentConfig?.runtime?.models || []

          // Build router model config
          const routerModelConfig = {
            name: routerName.trim(),
            provider: 'router',
            description: `Semantic router trained via UI`,
            embedder_model: embedderModel,
            default_model: defaultModel.trim(),
            similarity_threshold: similarityThreshold,
            routes: routesData.map(r => ({
              name: r.name,
              target_model: r.target_model,
              description: r.description,
              utterances: r.utterances,
            })),
          }

          // Check if router already exists in config
          const existingIndex = currentModels.findIndex(
            (m: any) => m.name === routerName.trim() && m.provider === 'router'
          )

          let updatedModels: any[]
          if (existingIndex >= 0) {
            // Update existing router
            updatedModels = [...currentModels]
            updatedModels[existingIndex] = routerModelConfig
          } else {
            // Add new router
            updatedModels = [...currentModels, routerModelConfig]
          }

          // Update project config
          await updateProjectMutation.mutateAsync({
            namespace,
            projectId,
            request: {
              config: {
                ...currentConfig,
                runtime: {
                  ...currentConfig.runtime,
                  models: updatedModels,
                },
              },
            },
          })
        } catch (configError) {
          console.warn('Failed to update project config:', configError)
          // Don't fail the whole operation - router was trained successfully
        }
      }

      setTrainingState('success')
      toast({
        message: `Router "${routerName}" trained successfully!`,
        icon: 'checkmark-filled',
      })

      // Refetch router list
      await refetchRouters()

      // Navigate to the router after a short delay
      setTimeout(() => {
        navigate(`/chat/models/train/router/${routerName.trim()}`)
      }, 1500)
    } catch (error) {
      console.error('Failed to train router:', error)
      setTrainingState('error')
      toast({
        message: 'Failed to train router. Please try again.',
        variant: 'destructive',
      })
    }
  }, [
    validateRouter,
    routerName,
    embedderModel,
    defaultModel,
    similarityThreshold,
    routes,
    trainRouterMutation,
    refetchRouters,
    navigate,
    toast,
    namespace,
    projectId,
    projectDetail,
    updateProjectMutation,
  ])

  // Test routing
  const handleTestRoute = useCallback(async () => {
    if (!testQuery.trim()) {
      toast({ message: 'Enter a query to test', variant: 'destructive' })
      return
    }

    if (!routerName.trim()) {
      toast({ message: 'Train the router first', variant: 'destructive' })
      return
    }

    try {
      const result = await routeQueryMutation.mutateAsync({
        model: routerName.trim(),
        query: testQuery.trim(),
        namespace: namespace || 'default',
        project_id: projectId || 'default',
      })

      const testResult: RouterTestResult = {
        id: `test_${Date.now()}`,
        query: testQuery.trim(),
        routeName: result.route_name,
        targetModel: result.target_model,
        similarityScore: result.similarity_score,
        matchedUtterance: result.matched_utterance,
        timestamp: new Date().toISOString(),
      }

      setTestResults(prev => [testResult, ...prev].slice(0, 10))
      setTestQuery('')
    } catch (error) {
      console.error('Failed to test route:', error)
      toast({
        message: 'Failed to test route. Make sure the router is trained.',
        variant: 'destructive',
      })
    }
  }, [testQuery, routerName, routeQueryMutation, toast])

  // Delete router
  const handleDelete = useCallback(async () => {
    if (!routerName || isNewRouter) return

    if (!window.confirm(`Delete router "${routerName}"? This cannot be undone.`)) {
      return
    }

    try {
      await deleteRouterMutation.mutateAsync(routerName)
      toast({
        message: `Router "${routerName}" deleted`,
        icon: 'checkmark-filled',
      })
      navigate('/chat/models?tab=trained')
    } catch (error) {
      console.error('Failed to delete router:', error)
      toast({
        message: 'Failed to delete router',
        variant: 'destructive',
      })
    }
  }, [routerName, isNewRouter, deleteRouterMutation, navigate, toast])

  // Check if form has valid data for training
  const canTrain = useMemo(() => {
    return (
      routerName.trim() &&
      defaultModel.trim() &&
      routes.length > 0 &&
      routes.every(
        r => r.name.trim() && r.targetModel.trim() && r.utterances.length > 0
      )
    )
  }, [routerName, defaultModel, routes])

  return (
    <div className="flex flex-col gap-6 max-w-4xl">
      {/* Training overlay */}
      {trainingState === 'training' && (
        <TrainingLoadingOverlay message="Training router..." />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/chat/models?tab=trained')}
          >
            <FontIcon type="chevron-down" className="w-4 h-4 rotate-90" />
          </Button>
          <div>
            <h2 className="text-lg font-medium">
              {isNewRouter ? 'Create semantic router' : `Edit router: ${routerName}`}
            </h2>
            <p className="text-sm text-muted-foreground">
              Route queries to the right LLM based on semantic similarity
            </p>
          </div>
        </div>
        {!isNewRouter && (
          <Button variant="destructive" size="sm" onClick={handleDelete}>
            Delete
          </Button>
        )}
      </div>

      {/* Router Configuration */}
      <div className="rounded-lg border border-border bg-card p-6 flex flex-col gap-4">
        <h3 className="font-medium">Router Configuration</h3>

        {/* Router Name */}
        <div className="flex flex-col gap-2">
          <Label htmlFor="router-name">Router name</Label>
          <Input
            id="router-name"
            value={routerName}
            onChange={e => setRouterName(e.target.value)}
            placeholder="my-customer-router"
            disabled={!isNewRouter}
          />
          <p className="text-xs text-muted-foreground">
            Unique identifier for this router
          </p>
        </div>

        {/* Embedder Model */}
        <div className="flex flex-col gap-2">
          <Label htmlFor="embedder-model">Embedder model</Label>
          <Select
            id="embedder-model"
            value={embedderModel}
            onChange={e => setEmbedderModel(e.target.value)}
          >
            {ROUTER_EMBEDDER_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
          <p className="text-xs text-muted-foreground">
            Model used to generate embeddings for semantic matching
          </p>
        </div>

        {/* Default Model */}
        <div className="flex flex-col gap-2">
          <Label htmlFor="default-model">Default model (fallback)</Label>
          {availableTargetModels.length > 0 ? (
            <Select
              id="default-model"
              value={defaultModel}
              onChange={e => setDefaultModel(e.target.value)}
            >
              <option value="">Select a model...</option>
              {availableTargetModels.map(m => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.model})
                </option>
              ))}
            </Select>
          ) : (
            <Input
              id="default-model"
              value={defaultModel}
              onChange={e => setDefaultModel(e.target.value)}
              placeholder="general-assistant"
            />
          )}
          <p className="text-xs text-muted-foreground">
            Model to use when no route matches the query
          </p>
        </div>

        {/* Similarity Threshold */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="similarity-threshold">Similarity threshold</Label>
            <span className="text-sm font-mono">{similarityThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            id="similarity-threshold"
            min={0}
            max={1}
            step={0.05}
            value={similarityThreshold}
            onChange={e => setSimilarityThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-muted-foreground">
            Minimum similarity score to match a route (0-1). Higher = stricter matching.
          </p>
        </div>
      </div>

      {/* Data Generation Settings */}
      <div className="rounded-lg border border-border bg-card p-6 flex flex-col gap-4">
        <h3 className="font-medium">Data Generation Settings</h3>
        <p className="text-sm text-muted-foreground -mt-2">
          Configure how synthetic training utterances are generated for each route
        </p>

        {/* Generation Model */}
        <div className="flex flex-col gap-2">
          <Label htmlFor="generation-model">Generation model</Label>
          <Select
            id="generation-model"
            value={generationModel}
            onChange={e => setGenerationModel(e.target.value)}
          >
            {availableGenerationModels.map(model => (
              <option key={model.id} value={model.id}>
                {model.name || model.id}
              </option>
            ))}
          </Select>
          <p className="text-xs text-muted-foreground">
            Local model used to generate utterances. No API key required.
          </p>
        </div>

        {/* Count and Complexity Row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="generation-count">Utterances per generation</Label>
            <Input
              id="generation-count"
              type="number"
              min={1}
              max={100}
              value={generationCount}
              onChange={e => setGenerationCount(Math.min(100, Math.max(1, parseInt(e.target.value) || 20)))}
            />
            <p className="text-xs text-muted-foreground">
              1-100 utterances (default: 20)
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="generation-complexity">Complexity</Label>
            <Select
              id="generation-complexity"
              value={generationComplexity}
              onChange={e => setGenerationComplexity(e.target.value as 'simple' | 'complex' | 'mixed')}
            >
              <option value="mixed">Mixed (recommended)</option>
              <option value="simple">Simple (5-10 words)</option>
              <option value="complex">Complex (15-30 words)</option>
            </Select>
            <p className="text-xs text-muted-foreground">
              Variety of generated utterances
            </p>
          </div>
        </div>

        {/* Example Style */}
        <div className="flex flex-col gap-2">
          <Label htmlFor="generation-style">Style hint (optional)</Label>
          <Input
            id="generation-style"
            value={generationStyle}
            onChange={e => setGenerationStyle(e.target.value)}
            placeholder="e.g., formal business language, casual chat, technical jargon"
          />
          <p className="text-xs text-muted-foreground">
            Optional hint for tone/style of generated utterances
          </p>
        </div>
      </div>

      {/* Routes Section */}
      <div className="rounded-lg border border-border bg-card p-6 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">Routes</h3>
            <p className="text-sm text-muted-foreground">
              Define routes that map queries to target models
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={handleAddRoute}>
            <FontIcon type="add" className="w-4 h-4 mr-1" />
            Add route
          </Button>
        </div>

        {routes.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <p className="text-sm text-muted-foreground">
              No routes yet. Add a route to get started.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {routes.map((route, index) => (
              <Collapsible
                key={route.id}
                open={expandedRoutes.has(route.id)}
                onOpenChange={() => toggleRouteExpanded(route.id)}
              >
                <div className="border rounded-lg">
                  <CollapsibleTrigger className="w-full p-3 flex items-center justify-between hover:bg-secondary/50">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="font-mono">
                        {index + 1}
                      </Badge>
                      <span className="font-medium">
                        {route.name || 'Unnamed route'}
                      </span>
                      {route.utterances.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {route.utterances.length} utterance
                          {route.utterances.length !== 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                    <FontIcon
                      type="chevron-down"
                      className={`w-4 h-4 transition-transform ${
                        expandedRoutes.has(route.id) ? '' : '-rotate-90'
                      }`}
                    />
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="p-4 pt-2 flex flex-col gap-4 border-t">
                      {/* Route Name and Target Model */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="flex flex-col gap-2">
                          <Label>Route name</Label>
                          <Input
                            value={route.name}
                            onChange={e =>
                              handleRouteChange(route.id, 'name', e.target.value)
                            }
                            placeholder="billing"
                          />
                        </div>
                        <div className="flex flex-col gap-2">
                          <Label>Target model</Label>
                          {availableTargetModels.length > 0 ? (
                            <Select
                              value={route.targetModel}
                              onChange={e =>
                                handleRouteChange(route.id, 'targetModel', e.target.value)
                              }
                            >
                              <option value="">Select a model...</option>
                              {availableTargetModels.map(m => (
                                <option key={m.name} value={m.name}>
                                  {m.name} ({m.model})
                                </option>
                              ))}
                            </Select>
                          ) : (
                            <Input
                              value={route.targetModel}
                              onChange={e =>
                                handleRouteChange(route.id, 'targetModel', e.target.value)
                              }
                              placeholder="billing-specialist"
                            />
                          )}
                        </div>
                      </div>

                      {/* Route Description */}
                      <div className="flex flex-col gap-2">
                        <Label>Description (for data generation)</Label>
                        <Input
                          value={route.description}
                          onChange={e =>
                            handleRouteChange(route.id, 'description', e.target.value)
                          }
                          placeholder="Billing and payment inquiries - questions about bills, invoices, account balances"
                        />
                      </div>

                      {/* Utterances */}
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <Label>Example utterances (one per line)</Label>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleGenerateData(route.id)}
                            disabled={
                              generatingForRoute === route.id ||
                              !route.description.trim()
                            }
                          >
                            {generatingForRoute === route.id ? (
                              'Generating...'
                            ) : (
                              <>
                                <FontIcon type="fade" className="w-4 h-4 mr-1" />
                                Generate examples
                              </>
                            )}
                          </Button>
                        </div>
                        <Textarea
                          value={formatUtterances(route.utterances)}
                          onChange={e =>
                            handleRouteChange(
                              route.id,
                              'utterances',
                              parseUtterances(e.target.value)
                            )
                          }
                          placeholder={"what is my bill\nhow much do I owe\npayment options\ninvoice question"}
                          rows={6}
                          className="font-mono text-sm"
                        />
                        <p className="text-xs text-muted-foreground">
                          Add example queries that should route to this model.
                          More examples = better matching.
                        </p>
                      </div>

                      {/* Remove Route Button */}
                      {routes.length > 1 && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-fit text-destructive"
                          onClick={() => handleRemoveRoute(route.id)}
                        >
                          <FontIcon type="trashcan" className="w-4 h-4 mr-1" />
                          Remove route
                        </Button>
                      )}
                    </div>
                  </CollapsibleContent>
                </div>
              </Collapsible>
            ))}
          </div>
        )}
      </div>

      {/* Train Button */}
      <div className="flex items-center gap-3">
        <Button
          onClick={handleTrain}
          disabled={!canTrain || trainingState === 'training'}
        >
          {trainingState === 'training' ? 'Training...' : 'Train Router'}
        </Button>
        {trainingState === 'success' && (
          <Badge className="bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300">
            Training successful!
          </Badge>
        )}
        {trainingState === 'error' && (
          <Badge className="bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300">
            Training failed
          </Badge>
        )}
      </div>

      {/* Test Section */}
      {!isNewRouter && (
        <div className="rounded-lg border border-border bg-card p-6 flex flex-col gap-4">
          <h3 className="font-medium">Test Router</h3>

          <div className="flex gap-2">
            <Input
              value={testQuery}
              onChange={e => setTestQuery(e.target.value)}
              placeholder="Enter a query to test routing..."
              onKeyDown={e => {
                if (e.key === 'Enter') handleTestRoute()
              }}
            />
            <Button
              onClick={handleTestRoute}
              disabled={routeQueryMutation.isPending}
            >
              {routeQueryMutation.isPending ? 'Testing...' : 'Test'}
            </Button>
          </div>

          {testResults.length > 0 && (
            <div className="flex flex-col gap-2">
              <Label>Results</Label>
              <div className="rounded-lg border border-border divide-y divide-border">
                {testResults.map(result => (
                  <div
                    key={result.id}
                    className="p-3 flex flex-col gap-1 text-sm"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">"{result.query}"</span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <span>Route:</span>
                      <Badge variant="outline">
                        {result.routeName || '(default)'}
                      </Badge>
                      <span className="mx-1">→</span>
                      <Badge>{result.targetModel}</Badge>
                      <span
                        className={`ml-auto font-mono text-xs px-2 py-0.5 rounded ${
                          result.similarityScore >= 0.8
                            ? 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300'
                            : result.similarityScore >= 0.6
                              ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-300'
                              : 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300'
                        }`}
                      >
                        {(result.similarityScore * 100).toFixed(1)}% match
                      </span>
                    </div>
                    {result.matchedUtterance && (
                      <div className="text-xs text-muted-foreground">
                        Matched: "{result.matchedUtterance}"
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Bottom spacer */}
      <div className="h-16" />
    </div>
  )
}

export default RouterModel
