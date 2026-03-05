import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import FontIcon from '../../common/FontIcon'
import { useBundles, useDeleteBundle } from '../../hooks/useBundles'
import { useBundleModal } from '../../contexts/BundleModalContext'
import { getBundleDownloadUrl } from '../../api/bundleService'
import type { BundleSummary } from '../../api/bundleService'
import { formatBytes } from '../../utils/formatBytes'

function formatDate(iso: string): string {
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  return d.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function targetLabel(b: BundleSummary): string {
  const platformNames: Record<string, string> = {
    linux: 'Linux',
    darwin: 'Mac',
    windows: 'Windows',
  }
  const platform = platformNames[b.platform] ?? b.platform
  return `${platform} · ${b.arch} · ${b.accelerator.toUpperCase()}`
}

const DeployPage = () => {
  const navigate = useNavigate()
  const { data: bundles, isLoading } = useBundles()
  const deleteMutation = useDeleteBundle()
  const { openBundleModal } = useBundleModal()
  const [toDelete, setToDelete] = useState<BundleSummary | null>(null)

  return (
    <div className="w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/dashboard')}
          >
            Dashboard
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Deploy</span>
        </nav>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg md:text-xl font-medium">Bundles</h2>
        <Button onClick={openBundleModal}>Create Bundle</Button>
      </div>

      {/* Bundle table or empty state */}
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : !bundles || bundles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-12 h-12 mb-5 text-muted-foreground/60 mx-auto">
            <FontIcon type="rocket" />
          </div>
          <div className="text-lg font-medium text-foreground mb-2">
            No bundles yet
          </div>
          <div className="text-sm text-muted-foreground mb-6 max-w-sm leading-relaxed">
            Bundle LlamaFarm into a portable archive you can transfer to
            remote or air-gapped machines — no internet required on the target.
          </div>
          <Button onClick={openBundleModal}>Create your first bundle</Button>
        </div>
      ) : (
        <section className="rounded-md overflow-hidden border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted">
              <tr>
                <th className="text-left px-4 py-2">Bundle</th>
                <th className="text-left px-4 py-2">Target</th>
                <th className="text-left px-4 py-2">Size</th>
                <th className="text-left px-4 py-2">Date</th>
                <th className="text-right px-4 py-2 w-[1%]">Actions</th>
              </tr>
            </thead>
            <tbody>
              {bundles.map((b) => (
                <tr
                  key={b.id}
                  className="bg-card border-t border-border hover:bg-accent/20"
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-foreground">
                      {b.filename || `${b.version}-${b.platform}`}
                    </div>
                    {b.addons.length > 0 && (
                      <div className="text-xs text-muted-foreground mt-0.5">
                        + {b.addons.join(', ')}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {targetLabel(b)}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatBytes(b.size)}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">
                    {formatDate(b.created_at)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-1">
                      <button
                        type="button"
                        className="h-8 w-8 inline-flex items-center justify-center rounded-md border border-input hover:bg-accent/30 text-muted-foreground hover:text-foreground"
                        aria-label="Download bundle"
                        onClick={() =>
                          window.open(getBundleDownloadUrl(b.id), '_blank')
                        }
                      >
                        <span className="w-4 h-4"><FontIcon type="download" /></span>
                      </button>
                      <button
                        type="button"
                        className="h-8 w-8 inline-flex items-center justify-center rounded-md border border-input hover:bg-accent/30 text-muted-foreground hover:text-red-500"
                        aria-label="Delete bundle"
                        onClick={() => setToDelete(b)}
                      >
                        <span className="w-4 h-4"><FontIcon type="trashcan" /></span>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {/* Delete confirmation */}
      <Dialog
        open={!!toDelete}
        onOpenChange={(open) => !open && setToDelete(null)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete bundle</DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            {toDelete ? (
              <>
                Are you sure you want to delete{' '}
                <span className="font-mono text-foreground">
                  {toDelete.filename}
                </span>
                ? This action cannot be undone.
              </>
            ) : null}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (toDelete) {
                  deleteMutation.mutate(toDelete.id)
                  setToDelete(null)
                }
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DeployPage
