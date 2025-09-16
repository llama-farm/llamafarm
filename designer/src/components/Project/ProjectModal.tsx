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
  initialDescription?: string
  onClose: () => void
  onSave: (name: string) => void
  onDelete?: () => void
  isLoading?: boolean
  projectError?: string | null
}

const ProjectModal: React.FC<ProjectModalProps> = ({
  isOpen,
  mode,
  initialName = '',
  initialDescription = '',
  onClose,
  onSave,
  onDelete,
  isLoading = false,
  projectError = null,
}) => {
  const [name, setName] = useState(initialName)
  const [desc, setDesc] = useState(initialDescription)

  useEffect(() => {
    if (isOpen) {
      setName(initialName)
      setDesc(initialDescription)
    }
  }, [isOpen, initialName, initialDescription])

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

  const handleDelete = () => {
    if (!onDelete) return
    const ok = confirm('Are you sure you want to delete this project?')
    if (ok) onDelete()
  }

  const handleCancel = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onClose()
  }

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
              onChange={e => setName(e.target.value)}
            />
            {projectError && (
              <p className="text-xs text-destructive mt-1">{projectError}</p>
            )}
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Description</label>
            <textarea
              rows={4}
              className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
              placeholder="Add a brief description"
              value={desc}
              onChange={e => setDesc(e.target.value)}
            />
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between gap-2">
          {mode === 'edit' ? (
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
              onClick={handleDelete}
              disabled={isLoading}
              type="button"
            >
              {isLoading ? 'Deleting...' : 'Delete'}
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-2 ml-auto">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
              onClick={handleCancel}
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
              onClick={e => {
                e.preventDefault()
                if (isValid) {
                  onSave(name.trim())
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
