import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'

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

  const [metaTick, setMetaTick] = useState(0)
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

  const projectConfig = (projectResp as any)?.project?.config as ProjectConfig | undefined
  const getDatabaseLocation = useCallback(() => {
    if (activeDatabase) {
      return {
        type: 'rag.database' as const,
        databaseName: activeDatabase,
      }
    }
    return { type: 'rag.databases' as const }
  }, [activeDatabase])
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getDatabaseLocation,
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

  // Database-scoped storage keys (UI-only fallbacks; namespaced per project)
  const EMB_LIST_KEY = `lf_ui_${projectKey}_db_${activeDatabase}_embeddings`
  const RET_LIST_KEY = `lf_ui_${projectKey}_db_${activeDatabase}_retrievals`

  const getEmbeddings = (): EmbeddingItem[] => {
    const arr = getStoredArray(EMB_LIST_KEY)
    return arr
      .filter(
        (e: any) => e && typeof e.id === 'string' && typeof e.name === 'string'
      )
      .map((e: any) => ({
        id: e.id,
        name: e.name,
        isDefault: Boolean(e.isDefault),
        enabled: typeof e.enabled === 'boolean' ? e.enabled : true,
      })) as EmbeddingItem[]
  }
  const saveEmbeddings = (list: EmbeddingItem[]) =>
    setStoredArray(EMB_LIST_KEY, list)

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

  // Live RAG databases (for retrieval strategies defaults) -------------------
  type RagDatabasesResponse = {
    databases: {
      name: string
      type?: string
      is_default?: boolean
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

  // Current config database (for embedding strategies)
  const currentConfigDb = useMemo(() => {
    const cfgDbs = (projectResp as any)?.project?.config?.rag?.databases
    if (!Array.isArray(cfgDbs)) return null
    return cfgDbs.find((d: any) => d?.name === activeDatabase) || null
  }, [projectResp, activeDatabase])

  // Embeddings from config (fallback to local only if config missing)
  const configEmbeddings: EmbeddingItem[] | null = useMemo(() => {
    if (!currentConfigDb) return null
    const list = Array.isArray(currentConfigDb.embedding_strategies)
      ? (currentConfigDb.embedding_strategies as any[])
      : []
    const def = currentConfigDb.default_embedding_strategy
    return list.map((e: any) => ({
      id: String(e?.name ?? 'embedding'),
      name: String(e?.name ?? 'embedding'),
      isDefault: def ? String(def) === String(e?.name) : false,
      enabled: true,
    }))
  }, [currentConfigDb])

  // Retrievals from server (fallback to local only if missing)
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

  const usingConfigEmbeddings = Boolean(
    configEmbeddings && configEmbeddings.length > 0
  )
  const usingServerRetrievals = Boolean(
    serverRetrievals && serverRetrievals.length > 0
  )

  // Seed defaults once
  useEffect(() => {
    // Skip local seeding when live config/server data are present
    if (usingConfigEmbeddings || usingServerRetrievals) return
    try {
      if (getEmbeddings().length === 0) {
        saveEmbeddings([
          {
            id: 'default_embeddings',
            name: 'default_embeddings',
            isDefault: true,
            enabled: true,
          },
        ])
      }
      if (getRetrievals().length === 0) {
        saveRetrievals([
          {
            id: 'basic_search',
            name: 'basic_search',
            isDefault: true,
            enabled: true,
          },
        ])
      }
      // Ensure each embedding strategy has a default config for display
      const embeddings = getEmbeddings()
      embeddings.forEach(e => {
        const key = `lf_db_${activeDatabase}_embedding_config_${e.id}`
        const raw = localStorage.getItem(key)
        if (!raw) {
          const payload = {
            runtime: 'local',
            provider: 'Ollama (remote)',
            modelId: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
            batchSize: 16,
            dimension: 768,
            maxInputTokens: 8192,
            similarity: 'cosine',
            timeout: 60,
          }
          try {
            localStorage.setItem(key, JSON.stringify(payload))
          } catch {}
        }
      })
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDatabase, usingConfigEmbeddings, usingServerRetrievals])

  const getEmbeddingSummary = (id: string): string => {
    try {
      const storedCfg = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      const storedModel = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_model_${id}`
      )
      if (storedModel) return storedModel
      if (storedCfg) {
        const parsed = JSON.parse(storedCfg)
        if (parsed?.modelId) return parsed.modelId
      }
    } catch {}
    return 'Not set'
  }
  const getEmbeddingProvider = (id: string): string | null => {
    try {
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      if (!raw) return null
      const cfg = JSON.parse(raw)
      const p = cfg?.provider || cfg?.runtime || null
      if (!p) return null
      if (typeof p === 'string') {
        if (p.includes('Ollama')) return 'Ollama'
        if (p.includes('OpenAI')) return 'OpenAI'
        if (p.includes('Cohere')) return 'Cohere'
        if (p.includes('Google')) return 'Google'
        if (p.includes('Azure')) return 'Azure OpenAI'
        if (p.includes('Bedrock')) return 'AWS Bedrock'
      }
      return String(p)
    } catch {
      return null
    }
  }
  // kept for future use if needed; currently not used after card redesign
  const getEmbeddingDimension = (id: string): number | null => {
    try {
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      if (!raw) return null
      const cfg = JSON.parse(raw)
      return typeof cfg?.dimension === 'number' ? cfg.dimension : null
    } catch {
      return null
    }
  }
  const getEmbeddingLocation = (id: string): string | null => {
    try {
      let raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      if (!raw) {
        raw = localStorage.getItem('lf_last_embedding_provider_config')
      }
      if (!raw) return null
      const cfg = JSON.parse(raw)
      const baseUrl = cfg?.baseUrl || cfg?.base_url
      if (typeof baseUrl === 'string' && baseUrl.trim().length > 0) {
        try {
          const u = new URL(baseUrl)
          return `${u.hostname}${u.port ? `:${u.port}` : ''}`
        } catch {
          return baseUrl
        }
      }
      if (cfg?.endpoint) {
        try {
          const u = new URL(cfg.endpoint)
          return `${u.hostname}${u.port ? `:${u.port}` : ''}`
        } catch {
          return String(cfg.endpoint)
        }
      }
      if (cfg?.region) return String(cfg.region)
      if (cfg?.deployment) return String(cfg.deployment)
      if (
        (cfg?.provider &&
          String(cfg.provider).toLowerCase().includes('ollama')) ||
        cfg?.runtime === 'local'
      ) {
        return 'localhost:11434'
      }
      return null
    } catch {
      return null
    }
  }
  const getEmbeddingRuntime = (id: string): 'Local' | 'Cloud' | null => {
    try {
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      if (!raw) return null
      const cfg = JSON.parse(raw)
      if (cfg?.runtime === 'local') return 'Local'
      if (cfg?.runtime === 'cloud') return 'Cloud'
      const p = String(cfg?.provider || '').toLowerCase()
      if (p.includes('ollama')) return 'Local'
      return 'Cloud'
    } catch {
      return null
    }
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

  const setDefaultEmbedding = (id: string) => {
    const list = getEmbeddings().map(e => ({ ...e, isDefault: e.id === id }))
    saveEmbeddings(list)
    setMetaTick(t => t + 1)
  }
  const setDefaultRetrieval = (id: string) => {
    const list = getRetrievals().map(r => ({ ...r, isDefault: r.id === id }))
    saveRetrievals(list)
    setMetaTick(t => t + 1)
  }
  const toggleRetrievalEnabled = (id: string) => {
    const list = getRetrievals().map(r =>
      r.id === id ? { ...r, enabled: !r.enabled } : r
    )
    saveRetrievals(list)
    setMetaTick(t => t + 1)
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
    setMetaTick(t => t + 1)
  }

  const sortedEmbeddings = useMemo(() => {
    const list = configEmbeddings ?? getEmbeddings()
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [metaTick, activeDatabase, configEmbeddings])
  const sortedRetrievals = useMemo(() => {
    const list = serverRetrievals ?? getRetrievals()
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [metaTick, activeDatabase, serverRetrievals])
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

  return (
    <>
      <div className="w-full h-full flex flex-col">
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
          <PageActions mode={mode} onModeChange={handleModeChange} />
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
            <ConfigEditor
              className="h-full"
              initialPointer={configPointer}
            />
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-auto pb-20">
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
                {!usingConfigEmbeddings && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      navigate(
                        `/chat/databases/add-embedding?database=${activeDatabase}`
                      )
                    }
                  >
                    Add new
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedEmbeddings.map(ei => (
                  <div
                    key={ei.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ei.enabled ? '' : 'opacity-70'} ${embeddingCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() =>
                      navigate(`/chat/databases/${ei.id}/change-embedding`)
                    }
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/databases/${ei.id}/change-embedding`)
                      }
                    }}
                  >
                    {!usingConfigEmbeddings && (
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
                                navigate(
                                  `/chat/databases/${ei.id}/change-embedding`
                                )
                              }}
                            >
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                setDefaultEmbedding(ei.id)
                                setReembedOpen(true)
                              }}
                            >
                              Make default
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-destructive focus:text-destructive"
                              onClick={e => {
                                e.stopPropagation()
                                const ok = confirm(
                                  'Remove this embedding strategy?'
                                )
                                if (!ok) return
                                // hard delete config and list entry
                                try {
                                  localStorage.removeItem(
                                    `lf_db_${activeDatabase}_embedding_config_${ei.id}`
                                  )
                                  localStorage.removeItem(
                                    `lf_db_${activeDatabase}_embedding_model_${ei.id}`
                                  )
                                } catch {}
                                let list = getEmbeddings().filter(
                                  x => x.id !== ei.id
                                )
                                if (
                                  list.length > 0 &&
                                  !list.some(x => x.isDefault)
                                ) {
                                  list[0].isDefault = true
                                }
                                saveEmbeddings(list)
                                setMetaTick(t => t + 1)
                              }}
                            >
                              Remove
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}

                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">
                          {getEmbeddingProvider(ei.id) || 'Provider'}
                        </div>
                        <div className="text-base font-semibold font-mono truncate">
                          {getEmbeddingSummary(ei.id)}
                        </div>
                        <div className="text-xs text-muted-foreground w-full truncate">
                          {(() => {
                            const loc = getEmbeddingLocation(ei.id)
                            return loc ? loc : ''
                          })()}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-wrap" />
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {ei.name}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap justify-between pt-0.5">
                      <div className="flex items-center gap-2 flex-wrap">
                        {(() => {
                          const dim = getEmbeddingDimension(ei.id)
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
                          const runtime = getEmbeddingRuntime(ei.id)
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
                        {ei.isDefault ? (
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
                            navigate(
                              `/chat/databases/${ei.id}/change-embedding`
                            )
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
                                setMetaTick(t => t + 1)
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
                                setMetaTick(t => t + 1)
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
          </div>
        )}

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
