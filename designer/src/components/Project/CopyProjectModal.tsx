import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { Label } from '../ui/label'
import { Input } from '../ui/input'

interface CopyProjectModalProps {
  isOpen: boolean
  sourceProjectName: string
  availableProjects: string[]
  onClose: () => void
  onCopy: (newName: string, sourceProject: string) => void
  isLoading?: boolean
  projectError?: string | null
  onNameChange?: (name: string) => void
}

const CopyProjectModal: React.FC<CopyProjectModalProps> = ({
  isOpen,
  sourceProjectName,
  availableProjects,
  onClose,
  onCopy,
  isLoading = false,
  projectError = null,
  onNameChange,
}) => {
  const [name, setName] = useState(`${sourceProjectName}-copy`)
  const [selectedSource, setSelectedSource] = useState(sourceProjectName)

  useEffect(() => {
    if (isOpen) {
      const initialName = `${sourceProjectName}-copy`
      setName(initialName)
      setSelectedSource(sourceProjectName)
      // Trigger validation on the next tick after state updates
      setTimeout(() => onNameChange?.(initialName), 0)
    }
  }, [isOpen, sourceProjectName])

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

  const isValid = name.trim().length > 0 && !isLoading

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

  const handleCreate = (e: React.MouseEvent) => {
    e.preventDefault()
    if (isValid) {
      onCopy(name.trim(), selectedSource)
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
          if (!isLoading) return
          e.preventDefault()
        }}
        onInteractOutside={e => {
          if (!isLoading) return
          e.preventDefault()
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">
            Create new project with existing configuration
          </DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-4 pt-1">
          <div className="grid gap-2.5">
            <Label htmlFor="copyProjectName">New project name</Label>
            <Input
              id="copyProjectName"
              value={name}
              onChange={e => {
                const newName = e.target.value
                setName(newName)
                onNameChange?.(newName)
              }}
              placeholder="Enter name"
              disabled={isLoading}
              className={projectError ? 'border-destructive' : ''}
            />
            {projectError && (
              <p className="text-xs text-destructive">{projectError}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Only letters, numbers, underscores (_), and hyphens (-) allowed.
              No spaces.
            </p>
          </div>

          <div className="grid gap-2.5">
            <Label htmlFor="copyFrom">Copy configuration from</Label>
            <select
              id="copyFrom"
              className="pl-3 pr-10 py-2 rounded-lg border border-input bg-card text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
              value={selectedSource}
              onChange={e => setSelectedSource(e.target.value)}
              disabled={isLoading}
            >
              {availableProjects.map(projectName => (
                <option key={projectName} value={projectName}>
                  {projectName}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Configuration will be copied from {selectedSource} (runtime, prompts, RAG settings)
            </p>
          </div>
        </div>

        <DialogFooter className="flex flex-row items-center justify-end gap-2">
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
            onClick={handleCreate}
            disabled={!isValid}
            type="button"
          >
            {isLoading ? 'Creating...' : 'Create'}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default CopyProjectModal

