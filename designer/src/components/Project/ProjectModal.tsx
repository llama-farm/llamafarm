import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

export type ProjectModalMode = 'create' | 'edit'

interface ProjectModalProps {
  isOpen: boolean
  mode: ProjectModalMode
  initialName?: string
  initialBrief?: { what?: string }
  onClose: () => void
  onSave: (
    name: string,
    details?: { brief?: { what?: string } }
  ) => void
  onOpenDelete?: () => void
  onCopy?: () => void
  isLoading?: boolean
  projectError?: string | null
  onNameChange?: (name: string) => void
}

const ProjectModal: React.FC<ProjectModalProps> = ({
  isOpen,
  mode,
  initialName = '',
  initialBrief = {},
  onClose,
  onSave,
  onOpenDelete,
  onCopy,
  isLoading = false,
  projectError = null,
  onNameChange,
}) => {
  const [name, setName] = useState(initialName)
  const [what, setWhat] = useState(initialBrief.what || '')

  useEffect(() => {
    if (isOpen) {
      setName(initialName)
      setWhat(initialBrief.what || '')
    }
  }, [
    isOpen,
    initialName,
    initialBrief?.what,
  ])

  // Handle Escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isLoading) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, isLoading, onClose])

  const title = mode === 'create' ? 'Create new project' : 'Edit project'
  const cta = mode === 'create' ? 'Create' : 'Save'
  const isValid = name.trim().length > 0 && !isLoading

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      onClose()
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sm:max-w-xl"
        onEscapeKeyDown={e => {
          e.preventDefault()
          if (!isLoading) onClose()
        }}
        onPointerDownOutside={e => {
          // allow clicking outside to close modal when not loading
          if (!isLoading) return
          e.preventDefault()
        }}
        onInteractOutside={e => {
          if (!isLoading) return
          e.preventDefault()
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">{title}</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-3 pt-1">
          <div>
            <label className="text-xs text-muted-foreground">
              Project name
            </label>
            <input
              className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border text-foreground ${
                projectError ? 'border-destructive' : 'border-input'
              }`}
              placeholder="Enter name"
              value={name}
              onChange={e => {
                const newName = e.target.value
                setName(newName)
                onNameChange?.(newName)
              }}
            />
            {projectError && (
              <p className="text-xs text-destructive mt-1">{projectError}</p>
            )}
          </div>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs text-muted-foreground">
                What are you building? (optional)
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground min-h-[44px]"
                placeholder="What are you building?"
                value={what}
                onChange={e => setWhat(e.target.value)}
              />
            </div>
          </div>
        </div>

        <DialogFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          {mode === 'edit' ? (
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
              onClick={e => {
                e.preventDefault()
                e.stopPropagation()
                if (onOpenDelete) onOpenDelete()
              }}
              disabled={isLoading}
              type="button"
            >
              Delete
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-2 ml-auto">
            <button
              className="px-3 py-2 rounded-md border border-input text-foreground hover:bg-accent/20 text-sm disabled:opacity-50"
              onClick={e => {
                e.preventDefault()
                e.stopPropagation()
                if (onCopy) onCopy()
              }}
              disabled={isLoading || mode !== 'edit'}
              type="button"
            >
              Copy to new project
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${
                isValid
                  ? 'bg-primary text-primary-foreground hover:opacity-90'
                  : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
              }`}
              onClick={e => {
                e.preventDefault()
                if (isValid) {
                  const details = { brief: { what: what.trim() } }
                  onSave(name.trim(), details)
                }
              }}
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default ProjectModal
