import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import PageActions from '../common/PageActions'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { getStoredArray, setStoredArray } from '../../utils/storage'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject } from '../../hooks/useProjects'
import { useListDatasets } from '../../hooks/useDatasets'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'
import { apiClient } from '../../api/client'
import DatabaseModal from './DatabaseModal'
import {
  useDatabaseManager,
  type Database as DatabaseType,
} from '../../hooks/useDatabaseManager'
import { Settings, Star, Trash2 } from 'lucide-react'

type Database = {
  name: string
  type?: string
  config?: Record<string, any>
}

function Databases() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [mode, setMode] = useModeWithReset('designer')
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

  const { data: datasetsResp } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject }
  )

  const [reembedOpen, setReembedOpen] = useState(false)

  // Database modal state
  const [databaseModalOpen, setDatabaseModalOpen] = useState(false)
  const [databaseModalMode, setDatabaseModalMode] = useState<'create' | 'edit'>(
    'create'
  )
  const [editingDatabase, setEditingDatabase] = useState<Database | null>(null)
  const [databaseError, setDatabaseError] = useState<string | null>(null)

  // Database mutations
  const databaseManager = useDatabaseManager(
    activeProject?.namespace || '',
    activeProject?.project || ''
  )

  // Database management -------------------------------------------------
  const databases = useMemo((): Database[] => {
    // Prefer databases from live project config
    const cfgDbs = (projectResp as any)?.project?.config?.rag?.databases
    if (Array.isArray(cfgDbs) && cfgDbs.length > 0) {
      return cfgDbs.map((db: any) => ({
        name: db?.name || 'unnamed',
        type: db?.type,
        config: db?.config,
      }))
    }

    // Fallbacks (legacy/local dev)
    try {
      const stored = localStorage.getItem('lf_databases')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed
        }
      }
    } catch {}

    return []
  }, [projectResp])

  // Active database state (namespaced per project)
  const projectKey = useMemo(() => {
    const ns = activeProject?.namespace || 'global'
    const proj = activeProject?.project || 'global'
    return `${ns}__${proj}`
  }, [activeProject?.namespace, activeProject?.project])

  const ACTIVE_DB_KEY = useMemo(
    () => `lf_ui_${projectKey}_active_database`,
    [projectKey]
  )

  const [activeDatabase, setActiveDatabase] = useState<string>(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DB_KEY)
      return stored || 'main_database'
    } catch {
      return 'main_database'
    }
  })

  // Persist active database selection
  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_DB_KEY, activeDatabase)
    } catch {}
  }, [activeDatabase, ACTIVE_DB_KEY])

  // Reload selection when project changes (validate against available list)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DB_KEY)
      if (stored) {
        // ensure it's in the current databases list
        const exists = databases.some(d => d.name === stored)
        setActiveDatabase(exists ? stored : activeDatabase)
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ACTIVE_DB_KEY, databases])

  // Ensure active database exists in the list
  useEffect(() => {
    if (
      databases.length > 0 &&
      !databases.find(db => db.name === activeDatabase)
    ) {
      setActiveDatabase(databases[0].name)
    }
  }, [databases, activeDatabase])

  // Connected datasets for the active database
  const connectedDatasets = useMemo(() => {
    if (!datasetsResp?.datasets) return []
    return datasetsResp.datasets.filter(
      (d: any) => d.database === activeDatabase
    )
  }, [datasetsResp, activeDatabase])

  // Database-level configs (Embeddings / Retrievals) -------------------------
  type EmbeddingItem = {
    id: string
    name: string
    isDefault: boolean
    enabled: boolean
  }
  type RetrievalItem = {
    id: string
    name: string
    isDefault: boolean
    enabled: boolean
  }

  // Database-scoped storage keys (UI-only fallbacks for retrieval strategies; namespaced per project)
  const RET_LIST_KEY = `lf_ui_${projectKey}_db_${activeDatabase}_retrievals`

  // Note: Embedding strategies are now managed server-side only, no localStorage fallbacks

  const getRetrievals = (): RetrievalItem[] => {
    const arr = getStoredArray(RET_LIST_KEY)
    return arr
      .filter(
        (e: any) => e && typeof e.id === 'string' && typeof e.name === 'string'
      )
      .map((e: any) => ({
        id: e.id,
        name: e.name,
        isDefault: Boolean(e.isDefault),
        enabled: typeof e.enabled === 'boolean' ? e.enabled : true,
      })) as RetrievalItem[]
  }
  const saveRetrievals = (list: RetrievalItem[]) =>
    setStoredArray(RET_LIST_KEY, list)

  // Live RAG databases (includes both retrieval and embedding strategies) -------
  type RagDatabasesResponse = {
    databases: {
      name: string
      type?: string
      is_default?: boolean
      embedding_strategies?: {
        name: string
        type?: string
        priority?: number
        is_default?: boolean
        config?: Record<string, any>
      }[]
      retrieval_strategies?: {
        name: string
        type?: string
        is_default?: boolean
      }[]
    }[]
    default_database?: string | null
  }
  const [ragDatabases, setRagDatabases] = useState<RagDatabasesResponse | null>(
    null
  )
  useEffect(() => {
    const ns = activeProject?.namespace
    const proj = activeProject?.project
    if (!ns || !proj) return
    let cancelled = false
    ;(async () => {
      try {
        const resp = await apiClient.get<RagDatabasesResponse>(
          `/projects/${encodeURIComponent(ns)}/${encodeURIComponent(
            proj
          )}/rag/databases`
        )
        if (!cancelled) setRagDatabases(resp.data)
      } catch {
        if (!cancelled) setRagDatabases(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeProject?.namespace, activeProject?.project])

  // Embeddings from server (no localStorage fallback)
  const serverEmbeddings: EmbeddingItem[] | null = useMemo(() => {
    if (!ragDatabases) return null
    const db = ragDatabases.databases?.find(d => d.name === activeDatabase)
    const list = db?.embedding_strategies || []
    return list.map(s => ({
      id: s.name,
      name: s.name,
      isDefault: Boolean(s.is_default),
      enabled: true,
    }))
  }, [ragDatabases, activeDatabase])

  // Retrievals from server (no localStorage fallback)
  const serverRetrievals: RetrievalItem[] | null = useMemo(() => {
    if (!ragDatabases) return null
    const db = ragDatabases.databases?.find(d => d.name === activeDatabase)
    const list = db?.retrieval_strategies || []
    return list.map(s => ({
      id: s.name,
      name: s.name,
      isDefault: Boolean(s.is_default),
      enabled: true,
    }))
  }, [ragDatabases, activeDatabase])

  const usingServerEmbeddings = Boolean(
    serverEmbeddings && serverEmbeddings.length > 0
  )
  const usingServerRetrievals = Boolean(
    serverRetrievals && serverRetrievals.length > 0
  )

  // Get embedding strategy details from server data
  const getEmbeddingStrategy = (strategyName: string) => {
    const db = ragDatabases?.databases?.find(d => d.name === activeDatabase)
    return db?.embedding_strategies?.find(s => s.name === strategyName)
  }

  const getEmbeddingSummary = (strategyName: string): string => {
    const strategy = getEmbeddingStrategy(strategyName)
    if (!strategy?.config) return 'Not configured'
    // Extract model name from config
    const model = strategy.config.model || strategy.config.modelId || strategy.config.model_name
    return model ? String(model) : 'Not set'
  }

  const getEmbeddingProvider = (strategyName: string): string | null => {
    const strategy = getEmbeddingStrategy(strategyName)
    if (!strategy || !strategy.type) return null
    // Map type to provider label
    const typeToProvider: Record<string, string> = {
      'OllamaEmbedder': 'Ollama',
      'OpenAIEmbedder': 'OpenAI',
      'HuggingFaceEmbedder': 'HuggingFace',
      'SentenceTransformerEmbedder': 'Sentence Transformers'
    }
    return typeToProvider[strategy.type] || strategy.type
  }

  const getEmbeddingDimension = (strategyName: string): number | null => {
    const strategy = getEmbeddingStrategy(strategyName)
    if (!strategy?.config) return null
    return strategy.config.dimension || strategy.config.vector_size || null
  }

  const getEmbeddingLocation = (strategyName: string): string | null => {
    const strategy = getEmbeddingStrategy(strategyName)
    if (!strategy?.config) return null
    
    const baseUrl = strategy.config.base_url || strategy.config.baseUrl
    if (baseUrl) {
      try {
        const u = new URL(baseUrl)
        return `${u.hostname}${u.port ? `:${u.port}` : ''}`
      } catch {
        return baseUrl
      }
    }
    
    if (strategy.config.endpoint) {
      try {
        const u = new URL(strategy.config.endpoint)
        return `${u.hostname}${u.port ? `:${u.port}` : ''}`
      } catch {
        return String(strategy.config.endpoint)
      }
    }
    
    if (strategy.config.region) return String(strategy.config.region)
    
    // Default for local Ollama
    if (strategy.type === 'OllamaEmbedder') {
      return 'localhost:11434'
    }
    
    return null
  }

  const getEmbeddingRuntime = (strategyName: string): 'Local' | 'Cloud' | null => {
    const strategy = getEmbeddingStrategy(strategyName)
    if (!strategy) return null
    // Ollama is local, others are typically cloud
    return strategy.type === 'OllamaEmbedder' ? 'Local' : 'Cloud'
  }
  // (removed) summary helper was unused
  const getRetrievalDescription = (_id: string): string => {
    return 'Vector search with configurable extraction pipeline'
  }
  const getRetrievalMeta = (rid: string): string => {
    if (rid.includes('filtered')) return 'MetadataFilteredStrategy'
    return 'BasicSimilarityStrategy'
  }

  // Processing strategy helpers ----------------------------------------------
  // Deprecated: old usage metric helper replaced by per-strategy dataset badges

  // Create handlers for new cards --------------------------------------------
  // const createEmbedding = () => {
  //   const name = prompt('Enter embedding strategy name')?.trim()
  //   if (!name) return
  //   const slug = name
  //     .toLowerCase()
  //     .replace(/[^a-z0-9]+/g, '-')
  //     .replace(/^-+|-+$/g, '')
  //   const id = `emb-${slug || Date.now()}`
  //   const list = getEmbeddings()
  //   list.push({ id, name, isDefault: list.length === 0, enabled: true })
  //   saveEmbeddings(list)
  //   setMetaTick(t => t + 1)
  //   navigate(`/chat/databases/${id}/change-embedding`)
  // }
  // const createRetrieval = () => {
  //   const name = prompt('Enter retrieval strategy name')?.trim()
  //   if (!name) return
  //   const slug = name
  //     .toLowerCase()
  //     .replace(/[^a-z0-9]+/g, '-')
  //     .replace(/^-+|-+$/g, '')
  //   const id = `ret-${slug || Date.now()}`
  //   const list = getRetrievals()
  //   list.push({ id, name, isDefault: list.length === 0, enabled: true })
  //   saveRetrievals(list)
  //   setMetaTick(t => t + 1)
  //   navigate(`/chat/databases/${id}/retrieval`)
  // }

  // Retrieval strategy handlers (still using localStorage for now)
  const setDefaultRetrieval = (id: string) => {
    const list = getRetrievals().map(r => ({ ...r, isDefault: r.id === id }))
    saveRetrievals(list)
  }
  const toggleRetrievalEnabled = (id: string) => {
    const list = getRetrievals().map(r =>
      r.id === id ? { ...r, enabled: !r.enabled } : r
    )
    saveRetrievals(list)
  }
  const duplicateRetrieval = (id: string) => {
    const list = getRetrievals()
    const found = list.find(r => r.id === id)
    if (!found) return
    const base = `${found.name} (copy)`
    const slug = base
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
    const newId = `ret-${slug}-${Date.now()}`
    list.push({ ...found, id: newId, name: base, isDefault: false })
    saveRetrievals(list)
  }

  const sortedEmbeddings = useMemo(() => {
    const list = serverEmbeddings || []
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [serverEmbeddings])
  
  const sortedRetrievals = useMemo(() => {
    const list = serverRetrievals || []
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [serverRetrievals])
  const embeddingCount = sortedEmbeddings.length
  const retrievalCount = sortedRetrievals.length

  // Database modal handlers
  const openCreateDatabaseModal = () => {
    setDatabaseModalMode('create')
    setEditingDatabase(null)
    setDatabaseError(null)
    setDatabaseModalOpen(true)
  }

  const openEditDatabaseModal = (dbName: string) => {
    const db = databases.find(d => d.name === dbName)
    if (!db) return
    setDatabaseModalMode('edit')
    setEditingDatabase(db)
    setDatabaseError(null)
    setDatabaseModalOpen(true)
  }

  const handleCreateDatabase = async (database: DatabaseType) => {
    try {
      setDatabaseError(null)
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      await databaseManager.createDatabase.mutateAsync({
        database,
        projectConfig,
      })

      toast({
        message: `Database "${database.name}" created successfully`,
        variant: 'default',
      })

      setActiveDatabase(database.name)
      setDatabaseModalOpen(false)
    } catch (error: any) {
      console.error('Failed to create database:', error)
      setDatabaseError(error.message || 'Failed to create database')
      throw error
    }
  }

  const handleUpdateDatabase = async (
    oldName: string,
    updates: Partial<DatabaseType>
  ) => {
    try {
      setDatabaseError(null)
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      // If renaming, update datasets that reference this database
      let datasetUpdates: Array<{ name: string; database: string }> = []
      if (updates.name && updates.name !== oldName) {
        const affectedDatasets =
          datasetsResp?.datasets?.filter((d: any) => d.database === oldName) ||
          []
        datasetUpdates = affectedDatasets.map((d: any) => ({
          name: d.name,
          database: updates.name!,
        }))
      }

      await databaseManager.updateDatabase.mutateAsync({
        oldName,
        updates,
        projectConfig,
        datasetUpdates: datasetUpdates.length > 0 ? datasetUpdates : undefined,
      })

      toast({
        message: `Database updated successfully`,
        variant: 'default',
      })

      // Update active database if it was renamed
      if (
        updates.name &&
        updates.name !== oldName &&
        activeDatabase === oldName
      ) {
        setActiveDatabase(updates.name)
      }

      setDatabaseModalOpen(false)
    } catch (error: any) {
      console.error('Failed to update database:', error)
      setDatabaseError(error.message || 'Failed to update database')
      throw error
    }
  }

  const handleDeleteDatabase = async (
    databaseName: string,
    reassignTo?: string
  ) => {
    try {
      setDatabaseError(null)
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      await databaseManager.deleteDatabase.mutateAsync({
        databaseName,
        projectConfig,
        reassignTo,
      })

      toast({
        message: `Database "${databaseName}" deleted successfully`,
        variant: 'default',
      })

      // Switch to first remaining database if the deleted one was active
      if (activeDatabase === databaseName && databases.length > 1) {
        const remaining = databases.filter(db => db.name !== databaseName)
        if (remaining.length > 0) {
          setActiveDatabase(remaining[0].name)
        }
      }

      setDatabaseModalOpen(false)
    } catch (error: any) {
      console.error('Failed to delete database:', error)
      setDatabaseError(error.message || 'Failed to delete database')
      throw error
    }
  }

  // Get affected datasets for database deletion
  const affectedDatasets = useMemo(() => {
    if (!editingDatabase || databaseModalMode !== 'edit') return []
    return (
      datasetsResp?.datasets?.filter(
        (d: any) => d.database === editingDatabase.name
      ) || []
    )
  }, [editingDatabase, databaseModalMode, datasetsResp])

  // Embedding strategy handlers
  const handleEditEmbedding = (embedding: EmbeddingItem) => {
    const strategy = getEmbeddingStrategy(embedding.name)
    if (!strategy) {
      toast({
        message: 'Strategy not found',
        variant: 'destructive',
      })
      return
    }

    navigate(`/chat/change-embedding-model`, {
      state: {
        database: activeDatabase,
        strategyName: embedding.name,
        strategyType: strategy.type || 'OllamaEmbedder',
        currentConfig: strategy.config || {},
        isDefault: embedding.isDefault,
        priority: strategy.priority || 0,
      },
    })
  }

  const handleSetDefaultEmbedding = async (strategyName: string) => {
    try {
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      const currentDb = projectConfig.rag?.databases?.find(
        (db: any) => db.name === activeDatabase
      )

      if (!currentDb) {
        throw new Error(`Database ${activeDatabase} not found`)
      }

      await databaseManager.updateDatabase.mutateAsync({
        oldName: activeDatabase,
        updates: {
          default_embedding_strategy: strategyName,
        },
        projectConfig,
      })

      toast({
        message: `"${strategyName}" set as default embedding strategy`,
        variant: 'default',
      })
    } catch (error: any) {
      console.error('Failed to set default embedding:', error)
      toast({
        message: error.message || 'Failed to set default strategy',
        variant: 'destructive',
      })
    }
  }

  const handleDeleteEmbedding = async (strategyName: string) => {
    if (
      !window.confirm(
        `Are you sure you want to delete the embedding strategy "${strategyName}"? This action cannot be undone.`
      )
    ) {
      return
    }

    try {
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      const currentDb = projectConfig.rag?.databases?.find(
        (db: any) => db.name === activeDatabase
      )

      if (!currentDb) {
        throw new Error(`Database ${activeDatabase} not found`)
      }

      const updatedStrategies =
        currentDb.embedding_strategies?.filter(
          (s: any) => s.name !== strategyName
        ) || []

      if (updatedStrategies.length === 0) {
        throw new Error(
          'Cannot delete the last embedding strategy. At least one strategy is required.'
        )
      }

      // If deleting the default, set the first remaining as default
      let updatedDefaultStrategy = currentDb.default_embedding_strategy
      if (currentDb.default_embedding_strategy === strategyName) {
        updatedDefaultStrategy = updatedStrategies[0]?.name || ''
      }

      await databaseManager.updateDatabase.mutateAsync({
        oldName: activeDatabase,
        updates: {
          embedding_strategies: updatedStrategies,
          default_embedding_strategy: updatedDefaultStrategy,
        },
        projectConfig,
      })

      toast({
        message: `Embedding strategy "${strategyName}" deleted`,
        variant: 'default',
      })
    } catch (error: any) {
      console.error('Failed to delete embedding strategy:', error)
      toast({
        message: error.message || 'Failed to delete strategy',
        variant: 'destructive',
      })
    }
  }

  return (
    <>
      <div
        className={`w-full h-full flex flex-col ${mode === 'designer' ? 'gap-4 pb-32' : ''}`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h2 className="text-2xl">
              {mode === 'designer' ? 'Databases' : 'Config editor'}
            </h2>
            {mode === 'designer' && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button className="text-muted-foreground hover:text-foreground transition-colors">
                      <FontIcon type="info" className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">
                      Databases store and organize your embedded data for AI
                      search. Create multiple databases to handle different
                      content types with specialized embedding and retrieval
                      strategies.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
          <PageActions mode={mode} onModeChange={setMode} />
        </div>
        {mode === 'designer' && databases.length > 0 && (
          <div className="mb-2 border-b border-border">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-1">
                {databases.map(db => (
                  <button
                    key={db.name}
                    onClick={() => setActiveDatabase(db.name)}
                    className={`px-4 py-2 font-medium transition-colors relative ${
                      activeDatabase === db.name
                        ? 'text-primary'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {db.name}
                    {activeDatabase === db.name && (
                      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                    )}
                  </button>
                ))}
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={openCreateDatabaseModal}
                className="mb-0.5"
              >
                Add database
                <FontIcon type="add" className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </div>
        )}
        {mode !== 'designer' ? (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" />
          </div>
        ) : (
          <>
            {/* Database name header */}
            <div className="flex items-center gap-2 mb-4 mt-2">
              <div className="text-xl font-semibold">{activeDatabase}</div>
              <button
                className="rounded-sm hover:opacity-80"
                onClick={() => openEditDatabaseModal(activeDatabase)}
                title="Edit database"
              >
                <FontIcon type="edit" className="w-5 h-5 text-primary" />
              </button>
            </div>

            {/* Embedding and Retrieval strategies - title outside card */}
            <div className="text-sm font-medium mb-1">
              Project Embedding and retrieval strategies
            </div>
            {/* Embeddings card */}
            <div className="rounded-lg border border-border bg-card p-4 pt-3">
              {/* Embeddings block header with Add new inline */}
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-foreground font-medium">
                  Embedding strategies ({embeddingCount})
                </div>
                {usingServerEmbeddings ? (
                  <span className="text-xs text-muted-foreground">
                    Managed by server
                  </span>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      navigate(
                        `/chat/add-embedding-strategy?database=${activeDatabase}`
                      )
                    }
                  >
                    Add new
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedEmbeddings.length === 0 && (
                  <div className="col-span-2 text-center p-6 text-sm text-muted-foreground">
                    No embedding strategies configured for this database.
                    {!usingServerEmbeddings && (
                      <div className="mt-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            navigate(
                              `/chat/add-embedding-strategy?database=${activeDatabase}`
                            )
                          }
                        >
                          Add first strategy
                        </Button>
                      </div>
                    )}
                  </div>
                )}
                {sortedEmbeddings.map(ei => (
                  <div
                    key={ei.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 transition-colors ${ei.enabled ? '' : 'opacity-70'} ${embeddingCount === 1 ? 'md:col-span-2' : ''}`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="text-base font-semibold truncate">
                          {ei.name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {getEmbeddingProvider(ei.name) || 'Provider'}
                        </div>
                        <div className="text-xs font-mono text-foreground mt-1">
                          {getEmbeddingSummary(ei.name)}
                        </div>
                        <div className="text-xs text-muted-foreground w-full truncate mt-0.5">
                          {(() => {
                            const loc = getEmbeddingLocation(ei.name)
                            return loc ? loc : ''
                          })()}
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0"
                                onClick={() => handleEditEmbedding(ei)}
                                title="Edit configuration"
                              >
                                <Settings className="h-4 w-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Edit configuration</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        {!ei.isDefault && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                  onClick={() => handleSetDefaultEmbedding(ei.name)}
                                  title="Set as default"
                                >
                                  <Star className="h-4 w-4" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Set as default</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                                onClick={() => handleDeleteEmbedding(ei.name)}
                                disabled={ei.isDefault}
                                title={
                                  ei.isDefault
                                    ? 'Cannot delete default strategy'
                                    : 'Delete strategy'
                                }
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                {ei.isDefault
                                  ? 'Cannot delete default strategy'
                                  : 'Delete strategy'}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap pt-2">
                      {(() => {
                        const dim = getEmbeddingDimension(ei.name)
                        return dim ? (
                          <Badge
                            variant="secondary"
                            size="sm"
                            className="rounded-xl"
                          >
                            {dim}-d
                          </Badge>
                        ) : null
                      })()}
                      {(() => {
                        const runtime = getEmbeddingRuntime(ei.name)
                        return runtime ? (
                          <Badge
                            variant="secondary"
                            size="sm"
                            className="rounded-xl"
                          >
                            {runtime}
                          </Badge>
                        ) : null
                      })()}
                      {ei.isDefault && (
                        <Badge variant="default" size="sm" className="rounded-xl">
                          Default
                        </Badge>
                      )}
                      {usingServerEmbeddings && (
                        <Badge variant="outline" size="sm" className="rounded-xl text-xs">
                          Server managed
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Retrievals card */}
            <div className="rounded-lg border border-border bg-card p-4 pt-3 mt-3">
              <div className="flex items-center justify-between mb-3">
                <div className="text-sm text-foreground font-medium">
                  Retrieval strategies ({retrievalCount})
                </div>
                {usingServerRetrievals ? (
                  <span className="text-xs text-muted-foreground">
                    Managed by server
                  </span>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      navigate(
                        `/chat/databases/add-retrieval?database=${activeDatabase}`
                      )
                    }
                  >
                    Add new
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedRetrievals.map(ri => (
                  <div
                    key={ri.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ri.enabled ? '' : 'opacity-70'} ${retrievalCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() =>
                      navigate(`/chat/databases/${ri.id}/retrieval`)
                    }
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/databases/${ri.id}/retrieval`)
                      }
                    }}
                  >
                    {!usingServerRetrievals && (
                      <div className="absolute top-2 right-2">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button
                              className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                              onClick={e => e.stopPropagation()}
                            >
                              <FontIcon type="overflow" className="w-4 h-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent
                            align="end"
                            className="min-w-[12rem] w-[12rem]"
                          >
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                const name = prompt(
                                  'Rename retrieval strategy',
                                  ri.name
                                )?.trim()
                                if (!name) return
                                const list = getRetrievals().map(x =>
                                  x.id === ri.id ? { ...x, name } : x
                                )
                                saveRetrievals(list)
                              }}
                            >
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                duplicateRetrieval(ri.id)
                              }}
                            >
                              Duplicate
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                setDefaultRetrieval(ri.id)
                              }}
                            >
                              Set as default
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                toggleRetrievalEnabled(ri.id)
                              }}
                            >
                              {ri.enabled ? 'Disable' : 'Enable'}
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-destructive focus:text-destructive"
                              onClick={e => {
                                e.stopPropagation()
                                const ok = confirm(
                                  'Delete this retrieval strategy?'
                                )
                                if (!ok) return
                                let list = getRetrievals().filter(
                                  x => x.id !== ri.id
                                )
                                if (
                                  list.length > 0 &&
                                  !list.some(x => x.isDefault)
                                ) {
                                  list[0].isDefault = true
                                }
                                saveRetrievals(list)
                              }}
                            >
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}

                    <div className="text-sm font-medium">{ri.name}</div>
                    <div className="text-xs text-primary text-left w-full truncate">
                      {getRetrievalDescription(ri.id)}
                    </div>
                    <div className="text-xs text-muted-foreground w-full truncate">
                      {getRetrievalMeta(ri.id)}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap justify-between pt-0.5">
                      <div className="flex items-center gap-2 flex-wrap">
                        {ri.isDefault ? (
                          <Badge
                            variant="default"
                            size="sm"
                            className="rounded-xl"
                          >
                            Default
                          </Badge>
                        ) : null}
                      </div>
                      <div className="ml-auto">
                        <Button
                          variant="outline"
                          size="sm"
                          className="px-3 h-7"
                          onClick={e => {
                            e.stopPropagation()
                            navigate(`/chat/databases/${ri.id}/retrieval`)
                          }}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Connected datasets section */}
            <div className="text-sm font-medium mb-1 mt-6">
              Connected datasets
            </div>
            <div className="rounded-lg border border-border bg-card p-4">
              {connectedDatasets.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-4">
                  No datasets connected to this database yet
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 px-2 font-medium">
                          Name
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          Files
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          Processing Strategy
                        </th>
                        <th className="w-20"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {connectedDatasets.map((dataset: any) => (
                        <tr
                          key={dataset.name}
                          className="border-b border-border last:border-0 hover:bg-accent/20"
                        >
                          <td className="py-2 px-2">{dataset.name}</td>
                          <td className="py-2 px-2">
                            {Array.isArray(dataset.files)
                              ? dataset.files.length
                              : 0}
                          </td>
                          <td className="py-2 px-2">
                            <Badge
                              variant="default"
                              size="sm"
                              className="rounded-xl"
                            >
                              {dataset.data_processing_strategy || 'default'}
                            </Badge>
                          </td>
                          <td className="py-2 px-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 px-3"
                              onClick={e => {
                                e.stopPropagation()
                                navigate(`/chat/data/${dataset.name}`)
                              }}
                            >
                              View
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        <div className="h-24 shrink-0" aria-hidden />
      </div>

      {/* Re-embed confirmation modal (after setting default embedding) */}
      <Dialog open={reembedOpen} onOpenChange={setReembedOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Re-embed project data?
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            To keep your project running smoothly, this change requires
            re-embedding project data. Would you like to proceed now?
          </div>
          <DialogFooter>
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                setReembedOpen(false)
                toast({
                  message: 'Default strategy updated',
                  variant: 'default',
                })
              }}
              type="button"
            >
              I'll do it later
            </button>
            <button
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
              onClick={() => {
                setReembedOpen(false)
                toast({
                  message: 'Default strategy updated',
                  variant: 'default',
                })
              }}
              type="button"
            >
              Yes, proceed with re-embed
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Database management modal */}
      <DatabaseModal
        isOpen={databaseModalOpen}
        mode={databaseModalMode}
        initialDatabase={editingDatabase as DatabaseType | undefined}
        existingDatabases={databases.map(db => ({
          name: db.name,
          type: (db.type || 'ChromaStore') as 'ChromaStore' | 'QdrantStore',
          config: db.config,
          default_embedding_strategy: (
            projectResp as any
          )?.project?.config?.rag?.databases?.find(
            (d: any) => d.name === db.name
          )?.default_embedding_strategy,
          default_retrieval_strategy: (
            projectResp as any
          )?.project?.config?.rag?.databases?.find(
            (d: any) => d.name === db.name
          )?.default_retrieval_strategy,
          embedding_strategies:
            (projectResp as any)?.project?.config?.rag?.databases?.find(
              (d: any) => d.name === db.name
            )?.embedding_strategies || [],
          retrieval_strategies:
            (projectResp as any)?.project?.config?.rag?.databases?.find(
              (d: any) => d.name === db.name
            )?.retrieval_strategies || [],
        }))}
        onClose={() => {
          setDatabaseModalOpen(false)
          setDatabaseError(null)
        }}
        onCreate={handleCreateDatabase}
        onUpdate={handleUpdateDatabase}
        onDelete={handleDeleteDatabase}
        isLoading={databaseManager.isUpdating}
        error={databaseError}
        affectedDatasets={affectedDatasets}
      />
    </>
  )
}

export default Databases
