import React from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { Button } from '../ui/button'

interface UnsavedChangesModalProps {
  isOpen: boolean
  onSave: () => void | Promise<void>
  onDiscard: () => void
  onCancel: () => void
  isSaving?: boolean
  errorMessage?: string | null
  isError?: boolean
}

/**
 * Modal that prompts users when they attempt to navigate away from the config editor
 * with unsaved changes. Provides options to save, discard, or cancel navigation.
 * Can also display an error state when save fails, giving users clear options.
 */
const UnsavedChangesModal: React.FC<UnsavedChangesModalProps> = ({
  isOpen,
  onSave,
  onDiscard,
  onCancel,
  isSaving = false,
  errorMessage = null,
  isError = false,
}) => {
  const handleOpenChange = (open: boolean) => {
    // Allow closing via X button unless saving
    // In error mode, X button acts like "Stay and Fix"
    if (!open && !isSaving) {
      onCancel()
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sm:max-w-md"
        onEscapeKeyDown={e => {
          e.preventDefault()
          // Escape key acts like "Stay and Fix" in error mode
          if (!isSaving) {
            onCancel()
          }
        }}
        onPointerDownOutside={e => {
          // Only prevent closing while saving
          // In error mode, clicking outside acts like "Stay and Fix"
          if (isSaving) {
            e.preventDefault()
          }
        }}
        onInteractOutside={e => {
          // Only prevent closing while saving
          if (isSaving) {
            e.preventDefault()
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">
            {isError ? 'Configuration Error' : 'Unsaved Changes'}
          </DialogTitle>
          <DialogDescription>
            {isError ? (
              <span className="text-destructive">
                {errorMessage || 'Your configuration contains errors.'}
              </span>
            ) : (
              'You have unsaved changes. Would you like to save them before leaving?'
            )}
          </DialogDescription>
        </DialogHeader>

        <DialogFooter className="flex flex-col-reverse gap-2 sm:gap-0 sm:flex-row sm:justify-end sm:space-x-2">
          {isError ? (
            <>
              <Button
                variant="outline"
                onClick={onCancel}
              >
                Stay and Fix
              </Button>
              <Button
                variant="default"
                onClick={onDiscard}
              >
                Discard and Continue
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="outline"
                onClick={onDiscard}
                disabled={isSaving}
              >
                Discard Changes
              </Button>
              <Button
                variant="default"
                onClick={onSave}
                disabled={isSaving}
              >
                {isSaving ? 'Saving...' : 'Save'}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default UnsavedChangesModal

