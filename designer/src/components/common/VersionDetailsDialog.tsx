import { useMemo, useState } from 'react'
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
}

export function VersionDetailsDialog({ open, onOpenChange }: Props) {
  const {
    currentVersion,
    latestVersion,
    upgradeAvailable,
    isLoading,
    refreshLatest,
    releasesUrl,
  } = useUpgradeAvailability()
  const [checking, setChecking] = useState(false)

  const statusText = useMemo(() => {
    if (checking || isLoading) return 'Checking for updates…'
    if (!latestVersion) return 'Unable to determine latest version'
    return upgradeAvailable
      ? `New version available: v${latestVersion}`
      : 'You’re on the latest version'
  }, [checking, isLoading, latestVersion, upgradeAvailable])

  const onCheck = async () => {
    setChecking(true)
    try {
      await refreshLatest()
    } finally {
      setChecking(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>LlamaFarm version</DialogTitle>
        </DialogHeader>
        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Current</span>
            <span className="font-mono text-foreground">v{currentVersion}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Latest</span>
            <span className="font-mono text-foreground">
              {latestVersion ? `v${latestVersion}` : '—'}
            </span>
          </div>
          <div className="pt-2 text-foreground">{statusText}</div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={onCheck}
            disabled={checking || isLoading}
          >
            {checking || isLoading ? 'Checking…' : 'Check for updates'}
          </Button>
          <a href={releasesUrl} target="_blank" rel="noreferrer">
            <Button>Open releases</Button>
          </a>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
