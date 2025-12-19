import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { useToast } from '../ui/toast'
import type { Database } from '../../hooks/useDatabaseManager'

export type DatabaseModalMode = 'create' | 'edit'

interface DatabaseModalProps {
  isOpen: boolean
  mode: DatabaseModalMode
  initialDatabase?: Database
  existingDatabases: Database[]
  onClose: () => void
  onCreate: (database: Database) => Promise<void>
  onUpdate: (oldName: string, updates: Partial<Database>) => Promise<void>
  onDelete: (databaseName: string, reassignTo?: string) => Promise<void>
  isLoading?: boolean
  error?: string | null
  affectedDatasets?: Array<{ name: string; database: string }>
}

const DatabaseModal: React.FC<DatabaseModalProps> = ({
  isOpen,
  mode,
  initialDatabase,
  existingDatabases,
  onClose,
  onCreate,
  onUpdate,
  onDelete,
  isLoading = false,
  error = null,
  affectedDatasets = [],
}) => {
  const { toast } = useToast()
  const [name, setName] = useState('')
  const [type, setType] = useState<'ChromaStore' | 'QdrantStore'>('ChromaStore')
  const [copyFromDb, setCopyFromDb] = useState('none')
  const [defaultEmbedding, setDefaultEmbedding] = useState('')
  const [defaultRetrieval, setDefaultRetrieval] = useState('')
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [reassignToDb, setReassignToDb] = useState('')

  useEffect(() => {
    if (isOpen) {
      setName(initialDatabase?.name || '')
      setType(initialDatabase?.type || 'ChromaStore')
      setCopyFromDb('none')
      if (mode === 'create') {
        // Set default values for new databases
        setDefaultEmbedding('semantic_embeddings')
        setDefaultRetrieval('comprehensive_search')
      } else {
        setDefaultEmbedding(initialDatabase?.default_embedding_strategy || '')
        setDefaultRetrieval(initialDatabase?.default_retrieval_strategy || '')
      }
      setConfirmingDelete(false)
      const otherDbs = existingDatabases.filter(
        db => db.name !== initialDatabase?.name
      )
      setReassignToDb(otherDbs[0]?.name || '')
    }
  }, [
    isOpen,
    initialDatabase,
    mode,
    affectedDatasets.length,
    existingDatabases,
  ])

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isLoading) onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, isLoading, onClose])

  const title =
    mode === 'create' ? 'Create new database' : `Edit ${initialDatabase?.name}`
  const cta = mode === 'create' ? 'Create' : 'Save'

  const nameValidationError = (() => {
    const trimmedName = name.trim()
    if (trimmedName.length === 0) return null

    // Check for invalid characters first
    if (!/^[a-zA-Z0-9_-]+$/.test(trimmedName)) {
      return 'Database name can only contain letters, numbers, hyphens (-), and underscores (_)'
    }

    // Check for at least one alphanumeric character
    if (!/[a-zA-Z0-9]/.test(trimmedName)) {
      return 'Database name must contain at least one letter or number'
    }

    return null
  })()

  const isValid = name.trim().length > 0 && !nameValidationError && !isLoading

  const handleSave = async () => {
    if (!isValid) return

    try {
      if (mode === 'create') {
        const snakeCaseName = name
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '_')
          .replace(/^_+|_+$/g, '')

        // Validate that sanitization didn't result in an empty string
        if (!snakeCaseName) {
          toast({
            message: 'Database name must contain at least one letter or number',
            variant: 'destructive',
          })
          return
        }

        const sourceDb =
          copyFromDb !== 'none'
            ? existingDatabases.find(db => db.name === copyFromDb)
            : undefined

        // Create default strategies if not copying from another database
        let embeddingStrategies: Database['embedding_strategies'] = []
        let retrievalStrategies: Database['retrieval_strategies'] = []
        let finalDefaultEmbedding = defaultEmbedding
        let finalDefaultRetrieval = defaultRetrieval

        if (sourceDb) {
          // Copy strategies from source database
          embeddingStrategies = JSON.parse(
            JSON.stringify(sourceDb.embedding_strategies || [])
          )
          retrievalStrategies = JSON.parse(
            JSON.stringify(sourceDb.retrieval_strategies || [])
          )
          finalDefaultEmbedding =
            defaultEmbedding || sourceDb.default_embedding_strategy || ''
          finalDefaultRetrieval =
            defaultRetrieval || sourceDb.default_retrieval_strategy || ''
        } else {
          // Not copying from a database - create or copy strategies based on selection
          // Find selected strategies from other databases if they exist
          const selectedEmbeddingStrategy = existingDatabases
            .flatMap(db => db.embedding_strategies || [])
            .find(emb => emb.name === defaultEmbedding)

          const selectedRetrievalStrategy = existingDatabases
            .flatMap(db => db.retrieval_strategies || [])
            .find(ret => ret.name === defaultRetrieval)

          // Handle embedding strategy
          if (defaultEmbedding && defaultEmbedding !== '') {
            if (defaultEmbedding === 'semantic_embeddings') {
              // Create the default embedding strategy
              embeddingStrategies.push({
                name: 'semantic_embeddings',
                type: 'UniversalEmbedder',
                priority: 0,
                config: {
                  model: 'sentence-transformers/all-MiniLM-L6-v2',
                  dimension: 384,
                  batch_size: 16,
                  timeout: 60,
                },
              })
            } else if (selectedEmbeddingStrategy) {
              // Copy the selected strategy from another database
              embeddingStrategies.push(
                JSON.parse(JSON.stringify(selectedEmbeddingStrategy))
              )
            }
          }

          // Handle retrieval strategy
          if (defaultRetrieval && defaultRetrieval !== '') {
            if (defaultRetrieval === 'comprehensive_search') {
              // Create the default retrieval strategy
              retrievalStrategies.push({
                name: 'comprehensive_search',
                type: 'BasicSimilarityStrategy',
                default: true,
                config: {
                  distance_metric: 'cosine',
                  top_k: 10,
                },
              })
            } else if (selectedRetrievalStrategy) {
              // Copy the selected strategy from another database
              retrievalStrategies.push(
                JSON.parse(JSON.stringify(selectedRetrievalStrategy))
              )
            }
          }

          // Set defaults (empty string if "None" was selected)
          finalDefaultEmbedding = defaultEmbedding || ''
          finalDefaultRetrieval = defaultRetrieval || ''
        }

        const newDatabase: Database = {
          name: snakeCaseName,
          type,
          config: {
            persist_directory: `./data/${type === 'ChromaStore' ? 'chroma_db' : 'qdrant_db'}`,
            distance_function: 'cosine',
            collection_name: snakeCaseName,
          },
          default_embedding_strategy: finalDefaultEmbedding,
          default_retrieval_strategy: finalDefaultRetrieval,
          embedding_strategies: embeddingStrategies,
          retrieval_strategies: retrievalStrategies,
        }

        await onCreate(newDatabase)
      } else {
        await onUpdate(initialDatabase?.name || '', {
          name: name.trim(),
          type,
        })
      }
      onClose()
    } catch (e) {
      console.error('Failed to save database:', e)
    }
  }

  const otherDatabases = existingDatabases.filter(
    db => db.name !== initialDatabase?.name
  )

  const copySourceDb =
    copyFromDb !== 'none'
      ? existingDatabases.find(db => db.name === copyFromDb)
      : undefined
  const availableEmbeddings = copySourceDb?.embedding_strategies || []
  const availableRetrievals = copySourceDb?.retrieval_strategies || []

  // When not copying, collect strategies from all existing databases
  const allEmbeddingStrategies =
    copyFromDb === 'none'
      ? existingDatabases.flatMap(db => db.embedding_strategies || [])
      : []
  const allRetrievalStrategies =
    copyFromDb === 'none'
      ? existingDatabases.flatMap(db => db.retrieval_strategies || [])
      : []

  const selectStyle = {
    backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
    backgroundPosition: 'right 0.75rem center',
    backgroundRepeat: 'no-repeat',
    backgroundSize: '1.5em 1.5em',
  }

  return (
    <Dialog open={isOpen} onOpenChange={open => !open && onClose()}>
      <DialogContent
        className="sm:max-w-xl"
        onEscapeKeyDown={e => {
          e.preventDefault()
          if (!isLoading) onClose()
        }}
        onPointerDownOutside={e => isLoading && e.preventDefault()}
        onInteractOutside={e => isLoading && e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">{title}</DialogTitle>
        </DialogHeader>

        {!confirmingDelete ? (
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Database name
              </label>
              <input
                className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border text-foreground ${
                  error || nameValidationError
                    ? 'border-destructive'
                    : 'border-input'
                }`}
                placeholder="Enter database name"
                value={name}
                onChange={e => setName(e.target.value)}
                disabled={isLoading}
              />
              {nameValidationError && (
                <p className="text-xs text-destructive mt-1">
                  {nameValidationError}
                </p>
              )}
              {error && !nameValidationError && (
                <p className="text-xs text-destructive mt-1">{error}</p>
              )}
            </div>

            <div>
              <label className="text-xs text-muted-foreground">Type</label>
              <select
                className="w-full mt-1 bg-transparent rounded-lg py-2 pl-3 pr-10 border border-input text-foreground appearance-none"
                style={selectStyle}
                value={type}
                onChange={e =>
                  setType(e.target.value as 'ChromaStore' | 'QdrantStore')
                }
                disabled={isLoading}
              >
                <option value="ChromaStore">ChromaStore</option>
                <option value="QdrantStore">QdrantStore</option>
              </select>
            </div>

            {mode === 'create' && (
              <>
                <div>
                  <label className="text-xs text-muted-foreground">
                    Copy strategies from
                  </label>
                  <select
                    className="w-full mt-1 bg-transparent rounded-lg py-2 pl-3 pr-10 border border-input text-foreground appearance-none"
                    style={selectStyle}
                    value={copyFromDb}
                    onChange={e => {
                      const newCopyFromDb = e.target.value
                      setCopyFromDb(newCopyFromDb)

                      // If switching to a database and current selections are "None",
                      // auto-select the default strategies from that database
                      if (newCopyFromDb !== 'none') {
                        const selectedDb = existingDatabases.find(
                          db => db.name === newCopyFromDb
                        )
                        if (selectedDb) {
                          // Only reset if current selection is "None" (empty string)
                          if (!defaultEmbedding || defaultEmbedding === '') {
                            setDefaultEmbedding(
                              selectedDb.default_embedding_strategy || ''
                            )
                          }
                          if (!defaultRetrieval || defaultRetrieval === '') {
                            setDefaultRetrieval(
                              selectedDb.default_retrieval_strategy || ''
                            )
                          }
                        }
                      } else {
                        // Switching back to "none" - reset to defaults if they were empty
                        if (!defaultEmbedding || defaultEmbedding === '') {
                          setDefaultEmbedding('semantic_embeddings')
                        }
                        if (!defaultRetrieval || defaultRetrieval === '') {
                          setDefaultRetrieval('comprehensive_search')
                        }
                      }
                    }}
                    disabled={isLoading}
                  >
                    <option value="none">None</option>
                    {existingDatabases.map(db => (
                      <option key={db.name} value={db.name}>
                        {db.name}
                      </option>
                    ))}
                  </select>
                  {copyFromDb !== 'none' ? (
                    <p className="text-xs text-muted-foreground mt-1">
                      This will copy all embedding and retrieval strategies from{' '}
                      {copyFromDb}
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground mt-1">
                      Default strategies will be created automatically if
                      selected
                    </p>
                  )}
                </div>

                <div>
                  <label className="text-xs text-muted-foreground">
                    Default embedding strategy
                  </label>
                  <select
                    className="w-full mt-1 bg-transparent rounded-lg py-2 pl-3 pr-10 border border-input text-foreground appearance-none"
                    style={selectStyle}
                    value={defaultEmbedding}
                    onChange={e => setDefaultEmbedding(e.target.value)}
                    disabled={isLoading}
                  >
                    <option value="">None</option>
                    {copyFromDb === 'none' ? (
                      <>
                        <option value="semantic_embeddings">
                          semantic_embeddings
                        </option>
                        {allEmbeddingStrategies.map(emb => (
                          <option key={emb.name} value={emb.name}>
                            {emb.name}
                          </option>
                        ))}
                      </>
                    ) : (
                      availableEmbeddings.map(emb => (
                        <option key={emb.name} value={emb.name}>
                          {emb.name}
                        </option>
                      ))
                    )}
                  </select>
                </div>

                <div>
                  <label className="text-xs text-muted-foreground">
                    Default retrieval strategy
                  </label>
                  <select
                    className="w-full mt-1 bg-transparent rounded-lg py-2 pl-3 pr-10 border border-input text-foreground appearance-none"
                    style={selectStyle}
                    value={defaultRetrieval}
                    onChange={e => setDefaultRetrieval(e.target.value)}
                    disabled={isLoading}
                  >
                    <option value="">None</option>
                    {copyFromDb === 'none' ? (
                      <>
                        <option value="comprehensive_search">
                          comprehensive_search
                        </option>
                        {allRetrievalStrategies.map(ret => (
                          <option key={ret.name} value={ret.name}>
                            {ret.name}
                          </option>
                        ))}
                      </>
                    ) : (
                      availableRetrievals.map(ret => (
                        <option key={ret.name} value={ret.name}>
                          {ret.name}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-3 pt-1">
            {affectedDatasets.length > 0 ? (
              <>
                <p className="text-sm text-foreground">
                  Deleting this database will leave {affectedDatasets.length}{' '}
                  dataset{affectedDatasets.length > 1 ? 's' : ''} unassigned.
                </p>
                <div>
                  <label className="text-xs text-muted-foreground">
                    Assign these datasets to:
                  </label>
                  <select
                    className="w-full mt-1 bg-transparent rounded-lg py-2 pl-3 pr-10 border border-input text-foreground appearance-none"
                    style={selectStyle}
                    value={reassignToDb}
                    onChange={e => setReassignToDb(e.target.value)}
                    disabled={isLoading || otherDatabases.length === 0}
                  >
                    {otherDatabases.map(db => (
                      <option key={db.name} value={db.name}>
                        {db.name}
                      </option>
                    ))}
                  </select>
                </div>
                <p className="text-xs text-muted-foreground">
                  Affected datasets:{' '}
                  {affectedDatasets.map(d => d.name).join(', ')}
                </p>
              </>
            ) : (
              <p className="text-sm text-foreground">
                Are you sure you want to delete this database?
              </p>
            )}
          </div>
        )}

        <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
          {mode === 'edit' && !confirmingDelete ? (
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
              onClick={() => setConfirmingDelete(true)}
              disabled={isLoading}
              type="button"
            >
              Delete
            </button>
          ) : confirmingDelete ? (
            <div className="flex items-center gap-2">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
                onClick={() => setConfirmingDelete(false)}
                disabled={isLoading}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
                onClick={async () => {
                  if (!initialDatabase) return
                  try {
                    await onDelete(
                      initialDatabase.name,
                      affectedDatasets.length > 0 ? reassignToDb : undefined
                    )
                    onClose()
                  } catch (e) {
                    console.error('Failed to delete database:', e)
                  }
                }}
                disabled={
                  isLoading || (affectedDatasets.length > 0 && !reassignToDb)
                }
                type="button"
              >
                Confirm delete
              </button>
            </div>
          ) : (
            <div />
          )}
          {!confirmingDelete && (
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
                onClick={e => {
                  e.preventDefault()
                  e.stopPropagation()
                  onClose()
                }}
                disabled={isLoading}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${
                  isValid
                    ? 'bg-primary text-primary-foreground hover:opacity-90'
                    : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
                }`}
                onClick={handleSave}
                disabled={!isValid}
                type="button"
              >
                {isLoading
                  ? mode === 'create'
                    ? 'Creating...'
                    : 'Saving...'
                  : cta}
              </button>
            </div>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default DatabaseModal
