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
      setDefaultEmbedding(initialDatabase?.default_embedding_strategy || '')
      setDefaultRetrieval(initialDatabase?.default_retrieval_strategy || '')
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

        const newDatabase: Database = {
          name: snakeCaseName,
          type,
          config: {
            persist_directory: `./data/${type === 'ChromaStore' ? 'chroma_db' : 'qdrant_db'}`,
            distance_function: 'cosine',
            collection_name: snakeCaseName,
          },
          default_embedding_strategy:
            defaultEmbedding || sourceDb?.default_embedding_strategy || '',
          default_retrieval_strategy:
            defaultRetrieval || sourceDb?.default_retrieval_strategy || '',
          embedding_strategies: sourceDb
            ? JSON.parse(JSON.stringify(sourceDb.embedding_strategies || []))
            : [],
          retrieval_strategies: sourceDb
            ? JSON.parse(JSON.stringify(sourceDb.retrieval_strategies || []))
            : [],
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
                      setCopyFromDb(e.target.value)
                      if (e.target.value === 'none') {
                        setDefaultEmbedding('')
                        setDefaultRetrieval('')
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
                  {copyFromDb !== 'none' && (
                    <p className="text-xs text-muted-foreground mt-1">
                      This will copy all embedding and retrieval strategies from{' '}
                      {copyFromDb}
                    </p>
                  )}
                </div>

                {copyFromDb !== 'none' && availableEmbeddings.length > 0 && (
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
                      <option value="">Select a strategy</option>
                      {availableEmbeddings.map(emb => (
                        <option key={emb.name} value={emb.name}>
                          {emb.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {copyFromDb !== 'none' && availableRetrievals.length > 0 && (
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
                      <option value="">Select a strategy</option>
                      {availableRetrievals.map(ret => (
                        <option key={ret.name} value={ret.name}>
                          {ret.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
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
