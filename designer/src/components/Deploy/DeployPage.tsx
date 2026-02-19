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
// FontIcon not used — using text labels for action buttons
import { useBundles, useDeleteBundle } from '../../hooks/useBundles'
import { useBundleModal } from '../../contexts/BundleModalContext'
import { getBundleDownloadUrl } from '../../api/bundleService'
import type { BundleSummary } from '../../api/bundleService'
import { formatBytes } from '../../utils/formatBytes'

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return iso
  }
}

function targetLabel(b: BundleSummary): string {
  const parts = []
  if (b.platform === 'linux') parts.push('Linux')
  else if (b.platform === 'darwin') parts.push('Mac')
  else if (b.platform === 'windows') parts.push('Windows')
  parts.push('·')
  parts.push(b.arch)
  parts.push('·')
  parts.push(b.accelerator.toUpperCase())
  return parts.join(' ')
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
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="text-4xl mb-4">📦</div>
          <div className="text-lg font-medium text-foreground mb-2">
            No bundles yet
          </div>
          <div className="text-sm text-muted-foreground mb-6 max-w-md">
            Bundles are distributable archives of LlamaFarm that you can
            transfer to remote or air-gapped machines.
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
                        className="h-8 w-8 inline-flex items-center justify-center rounded-md border border-input hover:bg-accent/30"
                        aria-label="Download bundle"
                        onClick={() =>
                          window.open(getBundleDownloadUrl(b.id), '_blank')
                        }
                      >
                        ⬇️
                      </button>
                      <button
                        type="button"
                        className="h-8 w-8 inline-flex items-center justify-center rounded-md border border-input hover:bg-accent/30 text-red-500"
                        aria-label="Delete bundle"
                        onClick={() => setToDelete(b)}
                      >
                        🗑️
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
