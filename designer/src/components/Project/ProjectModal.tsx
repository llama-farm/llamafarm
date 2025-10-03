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
  initialBrief?: { what: string; goals: string; audience: string }
  onClose: () => void
  onSave: (
    name: string,
    details: { brief: { what: string; goals: string; audience: string } }
  ) => void
  onDelete?: () => void
  isLoading?: boolean
  projectError?: string | null
}

const ProjectModal: React.FC<ProjectModalProps> = ({
  isOpen,
  mode,
  initialName = '',
  initialBrief = { what: '', goals: '', audience: '' },
  onClose,
  onSave,
  onDelete,
  isLoading = false,
  projectError = null,
}) => {
  const [name, setName] = useState(initialName)
  const [what, setWhat] = useState(initialBrief.what || '')
  const [goals, setGoals] = useState(initialBrief.goals || '')
  const [audience, setAudience] = useState(initialBrief.audience || '')
  const [confirmingDelete, setConfirmingDelete] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setName(initialName)
      setWhat(initialBrief.what || '')
      setGoals(initialBrief.goals || '')
      setAudience(initialBrief.audience || '')
      setConfirmingDelete(false)
    }
  }, [
    isOpen,
    initialName,
    initialBrief?.what,
    initialBrief?.goals,
    initialBrief?.audience,
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

  const handleDelete = () => {
    if (!onDelete) return
    setConfirmingDelete(true)
  }

  const handleConfirmDelete = () => {
    if (!onDelete) return
    onDelete()
  }

  const handleCancel = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onClose()
  }

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setConfirmingDelete(false)
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
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs text-muted-foreground">What are you building?</label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground min-h-[44px]"
                placeholder="What are you building?"
                value={what}
                onChange={e => setWhat(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">What do you hope to achieve?</label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground min-h-[44px]"
                placeholder="What do you hope to achieve?"
                value={goals}
                onChange={e => setGoals(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Who will use this?</label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Who will use this?"
                value={audience}
                onChange={e => setAudience(e.target.value)}
              />
            </div>
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between sm:justify-between gap-2">
          {mode === 'edit' ? (
            confirmingDelete ? (
              <div className="flex items-center gap-2">
                <span className="text-sm">Delete this project?</span>
                <button
                  className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
                  onClick={() => setConfirmingDelete(false)}
                  disabled={isLoading}
                  type="button"
                >
                  Keep
                </button>
                <button
                  className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
                  onClick={handleConfirmDelete}
                  disabled={isLoading}
                  type="button"
                >
                  Confirm delete
                </button>
              </div>
            ) : (
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
                onClick={handleDelete}
                disabled={isLoading}
                type="button"
              >
                Delete
              </button>
            )
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
                  onSave(name.trim(), { brief: { what, goals, audience } })
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
