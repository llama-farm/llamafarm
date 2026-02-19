import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useToast } from '@/components/ui/toast'
import { Selector } from '@/components/ui/selector'
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

            <Selector
              label="Type"
              value={type}
              onChange={v => setType(v as 'ChromaStore' | 'QdrantStore')}
              options={[
                { value: 'ChromaStore', label: 'ChromaStore' },
                { value: 'QdrantStore', label: 'QdrantStore' },
              ]}
              disabled={isLoading}
              className="mt-0"
            />

            {mode === 'create' && (
              <>
                <div>
                  <Selector
                    label="Copy strategies from"
                    value={copyFromDb}
                    onChange={newCopyFromDb => {
                      setCopyFromDb(newCopyFromDb)

                      // If switching to a database and current selections are "None",
                      // auto-select the default strategies from that database
                      if (newCopyFromDb !== 'none') {
                        const selectedDb = existingDatabases.find(
                          db => db.name === newCopyFromDb
                        )
                        if (selectedDb) {
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
                        if (!defaultEmbedding || defaultEmbedding === '') {
                          setDefaultEmbedding('semantic_embeddings')
                        }
                        if (!defaultRetrieval || defaultRetrieval === '') {
                          setDefaultRetrieval('comprehensive_search')
                        }
                      }
                    }}
                    options={[
                      { value: 'none', label: 'None' },
                      ...existingDatabases.map(db => ({
                        value: db.name,
                        label: db.name,
                      })),
                    ]}
                    disabled={isLoading}
                  />
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

                <Selector
                  label="Default embedding strategy"
                  value={defaultEmbedding}
                  onChange={v => setDefaultEmbedding(v)}
                  options={[
                    { value: '', label: 'None' },
                    ...(copyFromDb === 'none'
                      ? [
                          { value: 'semantic_embeddings', label: 'semantic_embeddings' },
                          ...allEmbeddingStrategies.map(emb => ({
                            value: emb.name,
                            label: emb.name,
                          })),
                        ]
                      : availableEmbeddings.map(emb => ({
                          value: emb.name,
                          label: emb.name,
                        }))),
                  ]}
                  disabled={isLoading}
                />

                <Selector
                  label="Default retrieval strategy"
                  value={defaultRetrieval}
                  onChange={v => setDefaultRetrieval(v)}
                  options={[
                    { value: '', label: 'None' },
                    ...(copyFromDb === 'none'
                      ? [
                          { value: 'comprehensive_search', label: 'comprehensive_search' },
                          ...allRetrievalStrategies.map(ret => ({
                            value: ret.name,
                            label: ret.name,
                          })),
                        ]
                      : availableRetrievals.map(ret => ({
                          value: ret.name,
                          label: ret.name,
                        }))),
                  ]}
                  disabled={isLoading}
                />
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
                <Selector
                  label="Assign these datasets to:"
                  value={reassignToDb}
                  onChange={v => setReassignToDb(v)}
                  options={otherDatabases.map(db => ({
                    value: db.name,
                    label: db.name,
                  }))}
                  disabled={isLoading || otherDatabases.length === 0}
                />
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
