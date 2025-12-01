import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { Button } from '../ui/button'
import { formatBytes } from '../../utils/modelUtils'

interface DiskSpaceWarningDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  message: string
  availableBytes: number
  requiredBytes: number
  onContinue: () => void
  onCancel: () => void
}

export function DiskSpaceWarningDialog({
  open,
  onOpenChange,
  message,
  availableBytes,
  requiredBytes,
  onContinue,
  onCancel,
}: DiskSpaceWarningDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader className="text-left">
          <DialogTitle>Low Disk Space Warning</DialogTitle>
        </DialogHeader>
        <div className="text-sm text-muted-foreground">
          <div className="mt-2 flex flex-col gap-4">
            <p className="text-sm text-foreground">{message}</p>

            <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-3">
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
              Downloading this model may leave you with less than 10% free disk
              space, which could affect LlamaFarm's capabilities. You can
              continue anyway, but we recommend freeing up space first.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onContinue}>
            Continue anyway
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
