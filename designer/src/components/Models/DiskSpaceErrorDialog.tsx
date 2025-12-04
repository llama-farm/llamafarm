import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { Button } from '../ui/button'
import { formatBytes } from '../../utils/modelUtils'

interface DiskSpaceErrorDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  message: string
  availableBytes: number
  requiredBytes: number
}

export function DiskSpaceErrorDialog({
  open,
  onOpenChange,
  message,
  availableBytes,
  requiredBytes,
}: DiskSpaceErrorDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader className="text-left">
          <DialogTitle>Insufficient Disk Space</DialogTitle>
        </DialogHeader>
        <div className="text-sm text-muted-foreground">
          <div className="mt-2 flex flex-col gap-4">
            <p className="text-sm text-foreground">{message}</p>

            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="text-muted-foreground">Available space</div>
                <div className="font-medium">{formatBytes(availableBytes)}</div>
                {requiredBytes > 0 && (
                  <>
                    <div className="text-muted-foreground">Model size</div>
                    <div className="font-medium">
                      {formatBytes(requiredBytes)}
                    </div>
                  </>
                )}
                {requiredBytes === 0 && (
                  <>
                    <div className="text-muted-foreground">Required space</div>
                    <div className="font-medium text-muted-foreground">
                      Unknown
                    </div>
                  </>
                )}
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              Please free up disk space before downloading this model. You can
              delete unused models or other files to make room.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
