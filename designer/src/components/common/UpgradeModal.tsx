import { useEffect, useMemo, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useUpgradeAvailability } from '@/hooks/useUpgradeAvailability'
import { getInjectedImageTag } from '@/utils/versionUtils'

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

function detectOS(): 'windows' | 'mac_linux' {
  const ua = navigator.userAgent.toLowerCase()
  if (ua.includes('windows')) return 'windows'
  return 'mac_linux'
}

export function UpgradeModal({ open, onOpenChange }: Props) {
  const {
    currentVersion,
    latestVersion,
    refreshLatest,
    upgradeAvailable,
    releasesUrl,
  } = useUpgradeAvailability()
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const [imageTag, setImageTag] = useState<string | null>(null)

  // Prefer injected image tag if available (checks both import.meta.env and window.ENV)
  useEffect(() => {
    let alive = true
    const tag = getInjectedImageTag()
    if (alive) setImageTag(tag)
    return () => {
      alive = false
    }
  }, [])

  const commands = useMemo(() => {
    const os = detectOS()
    const tag = (() => {
      if (latestVersion && latestVersion.trim() !== '')
        return `v${latestVersion}`
      if (imageTag && imageTag.trim() !== '')
        return imageTag.startsWith('v') ? imageTag : `v${imageTag}`
      return currentVersion && currentVersion.trim() !== ''
        ? `v${currentVersion}`
        : 'v0.0.0'
    })()

    const cli: { label: string; cmd: string }[] = []
    if (os === 'mac_linux') {
      cli.push({
        label: 'One-liner (CLI + services)',
        cmd: `lf version upgrade ${tag} && docker rm -f llamafarm-server llamafarm-rag llamafarm-designer 2>/dev/null || true; LF_IMAGE_TAG=${tag} lf start; docker run -d --restart unless-stopped --name llamafarm-designer -p 7724:80 ghcr.io/llama-farm/llamafarm/designer:${tag}`,
      })
    } else {
      cli.push({
        label: 'One-liner (CLI + services)',
        cmd: `winget install LlamaFarm.CLI && set TAG=${tag} && docker rm -f llamafarm-server llamafarm-rag llamafarm-designer 2>NUL || ver>NUL && set LF_IMAGE_TAG=%TAG% && lf start && docker run -d --restart unless-stopped --name llamafarm-designer -p 7724:80 ghcr.io/llama-farm/llamafarm/designer:%TAG%`,
      })
    }
    return { os, cli }
  }, [latestVersion, imageTag, currentVersion])

  const copy = async (text: string, idx: number) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedIdx(idx)
      window.setTimeout(() => setCopiedIdx(null), 1200)
    } catch {}
  }

  // Auto-refresh latest when the modal opens, and cancel on close/unmount
  useEffect(() => {
    if (!open) return
    const abort = new AbortController()
    refreshLatest(abort.signal)
    return () => {
      abort.abort()
    }
  }, [open, refreshLatest])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Upgrade LlamaFarm</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Current</span>
              <span className="font-mono">v{imageTag || currentVersion}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Latest</span>
              <span className="font-mono">
                {latestVersion ? `v${latestVersion}` : '—'}
              </span>
            </div>
          </div>

          {!upgradeAvailable ? (
            <div className="flex items-center justify-between rounded-md border border-teal-500/40 bg-teal-500/10 text-teal-400 px-4 py-3">
              <div className="text-sm font-medium">You're up to date!</div>
              <a href={releasesUrl} target="_blank" rel="noreferrer">
                <Button>View release notes</Button>
              </a>
            </div>
          ) : (
            <>
              <div className="space-y-2">
                <div className="text-sm font-medium">Upgrade LlamaFarm</div>
                <div className="rounded-md border border-border divide-y">
                  {commands.cli.map((c, i) => (
                    <div
                      key={i}
                      className="p-3 flex items-center justify-between gap-2"
                    >
                      <div className="min-w-0">
                        <div className="text-xs text-muted-foreground">
                          {c.label}
                        </div>
                        <pre className="mt-1 text-xs font-mono whitespace-pre-wrap break-all text-foreground">
                          {c.cmd}
                        </pre>
                      </div>
                      <button
                        type="button"
                        onClick={() => copy(c.cmd, i)}
                        className={`h-8 px-2 rounded-md border text-xs hover:bg-accent/30 ${copiedIdx === i ? 'border-teal-400 text-teal-400' : 'border-input'}`}
                      >
                        {copiedIdx === i ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-end gap-2">
                <a href={releasesUrl} target="_blank" rel="noreferrer">
                  <Button>View release notes</Button>
                </a>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default UpgradeModal
