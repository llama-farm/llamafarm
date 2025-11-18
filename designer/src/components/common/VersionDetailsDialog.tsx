import { useMemo } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useUpgradeAvailability } from '@/hooks/useUpgradeAvailability'

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onRequestUpgrade?: () => void
}

export function VersionDetailsDialog({
  open,
  onOpenChange,
  onRequestUpgrade,
}: Props) {
  const { currentVersion, latestVersion, upgradeAvailable } =
    useUpgradeAvailability()

  const status = useMemo(() => {
    if (upgradeAvailable && latestVersion) {
      return {
        text: `New version available: v${latestVersion}`,
        highlight: true,
      }
    }
    return { text: 'You’re on the latest version', highlight: false }
  }, [upgradeAvailable, latestVersion])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>LlamaFarm version</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Current</span>
            <span className="font-mono text-foreground">
              {currentVersion ? `v${currentVersion}` : '—'}
            </span>
          </div>
          <div className="pt-1">
            {status.highlight ? (
              <span className="inline-flex items-center rounded-full border border-teal-500/40 bg-teal-500/10 text-teal-400 px-2.5 py-1 text-xs">
                {status.text}
              </span>
            ) : (
              <span className="text-foreground">{status.text}</span>
            )}
          </div>
        </div>
        <DialogFooter>
          {status.highlight ? (
            <Button onClick={onRequestUpgrade}>Upgrade now</Button>
          ) : (
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
