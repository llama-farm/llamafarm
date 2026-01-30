import { useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

interface DeleteProjectModalProps {
  isOpen: boolean
  projectName: string
  onClose: () => void
  onConfirm: () => void
  isLoading?: boolean
}

const DeleteProjectModal: React.FC<DeleteProjectModalProps> = ({
  isOpen,
  projectName,
  onClose,
  onConfirm,
  isLoading = false,
}) => {
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

  const handleOpenChange = (open: boolean) => {
    if (!open && !isLoading) {
      onClose()
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sm:max-w-md"
        hideCloseButton
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
            Delete project?
          </DialogTitle>
        </DialogHeader>

        <div className="py-2">
          <p className="text-sm text-foreground">
            Are you sure you want to delete <strong>{projectName}</strong>? This
            action cannot be undone.
          </p>
        </div>

        <DialogFooter className="flex flex-row items-center justify-end gap-2">
          <button
            className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
            onClick={onClose}
            disabled={isLoading}
            type="button"
          >
            Cancel
          </button>
          <button
            className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm disabled:opacity-50"
            onClick={onConfirm}
            disabled={isLoading}
            type="button"
          >
            {isLoading ? 'Deleting...' : 'Delete'}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default DeleteProjectModal

