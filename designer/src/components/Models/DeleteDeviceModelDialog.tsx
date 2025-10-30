import { Button } from '../ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import type { DeviceModel } from './DeviceModelsSection'

interface DeleteDeviceModelDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  model: DeviceModel | null
  deleteState: 'idle' | 'deleting' | 'success' | 'error'
  deleteError: string
  onConfirmDelete: () => void
}

export function DeleteDeviceModelDialog({
  open,
  onOpenChange,
  model,
  deleteState,
  deleteError,
  onConfirmDelete,
}: DeleteDeviceModelDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogTitle>Delete model from disk?</DialogTitle>
        <DialogDescription>
          {model && (
            <div className="mt-2 flex flex-col gap-3">
              <p className="text-sm">
                Are you sure you want to delete
                <span className="mx-1 font-medium text-foreground">
                  {model.name}
                </span>
                from disk? This will permanently remove the model files (
                {model.meta}) and cannot be undone.
              </p>

              {deleteState === 'error' && deleteError && (
                <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
                  <p className="text-sm text-destructive">{deleteError}</p>
                </div>
              )}
            </div>
          )}
        </DialogDescription>
        <DialogFooter>
          <Button
            variant="secondary"
            onClick={() => onOpenChange(false)}
            disabled={deleteState === 'deleting'}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirmDelete}
            disabled={deleteState === 'deleting'}
          >
            {deleteState === 'deleting' && (
              <span className="mr-2 inline-flex">
                <Loader size={14} className="border-destructive-foreground" />
              </span>
            )}
            {deleteState === 'success' && (
              <span className="mr-2 inline-flex">
                <FontIcon type="checkmark-filled" className="w-4 h-4" />
              </span>
            )}
            Delete from disk
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

