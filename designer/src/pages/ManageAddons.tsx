import { useState, useMemo, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  useListAddons,
  useInstallAddon,
  useUninstallAddon,
  addonKeys,
} from '../hooks/useAddons'
import { getTaskStatus } from '../api/addonsService'
import { useQueryClient } from '@tanstack/react-query'
import { useToast } from '../components/ui/toast'
import { SearchInput } from '../components/ui/search-input'
import { Package, Zap, HardDrive, Trash2 } from 'lucide-react'
import {
  AddonInstallSidePane,
  AddonInstallProgress,
} from '../components/Addons'
import type { AddonInfo } from '../types/addons'

export function ManageAddons() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Fetch addons
  const { data: addons, isLoading } = useListAddons()

  // Search state
  const [search, setSearch] = useState('')

  // Installation state
  const [showInstallPane, setShowInstallPane] = useState(false)
  const [selectedAddon, setSelectedAddon] = useState<AddonInfo | null>(null)
  const [addonsToInstall, setAddonsToInstall] = useState<AddonInfo[]>([])
  const [installTaskId, setInstallTaskId] = useState<string | null>(null)
  const [installingAddonName, setInstallingAddonName] = useState<string | null>(null)

  const installAbortRef = useRef<AbortController | null>(null)

  // Dependency warning state
  const [showDependencyWarning, setShowDependencyWarning] = useState(false)
  const [addonToRemove, setAddonToRemove] = useState<AddonInfo | null>(null)
  const [dependentAddons, setDependentAddons] = useState<AddonInfo[]>([])

  // Simple removal confirmation state
  const [showRemoveConfirm, setShowRemoveConfirm] = useState(false)

  const installMutation = useInstallAddon()
  const uninstallMutation = useUninstallAddon()

  // Split installed/available addons
  const { installedAddons, availableAddons } = useMemo(() => {
    if (!addons) return { installedAddons: [], availableAddons: [] }

    const filtered = search
      ? addons.filter(
          a =>
            a.name.toLowerCase().includes(search.toLowerCase()) ||
            a.display_name.toLowerCase().includes(search.toLowerCase()) ||
            a.description.toLowerCase().includes(search.toLowerCase())
        )
      : addons

    // Meta-addons (no packages, only dependencies) exist for backend dependency
    // resolution but should not appear on this page — users install individual addons directly
    const isMetaAddon = (a: AddonInfo) =>
      a.dependencies.length > 0 && (!a.packages || a.packages.length === 0)

    return {
      installedAddons: filtered.filter(a => a.installed && !isMetaAddon(a)),
      availableAddons: filtered.filter(a => !a.installed && !isMetaAddon(a)),
    }
  }, [addons, search])

  // Handlers
  const handleInstall = (addon: AddonInfo) => {
    // Gather all add-ons that will be installed (addon + its dependencies)
    const addonsToShow: AddonInfo[] = []

    // Add uninstalled dependencies first (they'll be installed first)
    if (addon.dependencies.length > 0 && addons) {
      addon.dependencies.forEach(depName => {
        const depAddon = addons.find(a => a.name === depName)
        if (depAddon && !depAddon.installed) {
          addonsToShow.push(depAddon)
        }
      })
    }

    // Add the main addon last
    addonsToShow.push(addon)

    setSelectedAddon(addon)
    setAddonsToInstall(addonsToShow)
    setShowInstallPane(true)
  }

  // Poll a background install task until it reaches a terminal state.
  // Throws on failure or timeout so the caller's catch block surfaces the error.
  const waitForTaskCompletion = useCallback(async (taskId: string, signal?: AbortSignal) => {
    const POLL_INTERVAL = 2000
    const MAX_WAIT = 10 * 60 * 1000 // 10 minutes
    const start = Date.now()
    while (Date.now() - start < MAX_WAIT) {
      if (signal?.aborted) throw new DOMException('Install cancelled', 'AbortError')
      await new Promise(r => setTimeout(r, POLL_INTERVAL))
      if (signal?.aborted) throw new DOMException('Install cancelled', 'AbortError')
      let status
      try {
        status = await getTaskStatus(taskId)
      } catch (e: unknown) {
        // Only retry true network failures (no server response).
        // Re-throw HTTP errors (4xx/5xx) since they indicate permanent problems.
        if (e && typeof e === 'object' && 'response' in e) throw e
        continue
      }
      if (status.status === 'completed') return
      if (status.status === 'failed') {
        throw new Error(status.error || `Install task ${taskId} failed`)
      }
    }
    throw new Error('Install timed out')
  }, [])

  const handleInstallConfirm = async (selectedAddons: string[]) => {
    if (selectedAddons.length === 0) return
    setShowInstallPane(false)

    // Set installing state
    setInstallingAddonName(selectedAddons[0])

    // Create an AbortController so cancel can stop the chain
    const abort = new AbortController()
    installAbortRef.current = abort

    try {
      // Install all selected addons sequentially, waiting for each to finish.
      // The server's install endpoint returns immediately (background task), so we
      // poll for completion before starting the next to avoid lock conflicts.
      for (let i = 0; i < selectedAddons.length; i++) {
        if (abort.signal.aborted) break

        const addonName = selectedAddons[i]
        const isLastAddon = i === selectedAddons.length - 1

        const response = await installMutation.mutateAsync({
          name: addonName,
          restart_service: isLastAddon, // Only restart after last addon
        })

        setInstallTaskId(response.task_id)

        // Wait for this install to finish before starting the next one
        if (!isLastAddon) {
          await waitForTaskCompletion(response.task_id, abort.signal)
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      setInstallingAddonName(null) // Clear on error
      toast({
        message: `Failed to install ${selectedAddon?.display_name}. Please try again.`,
        variant: 'destructive',
      })
    } finally {
      installAbortRef.current = null
    }
  }

  const handleInstallComplete = () => {
    setInstallTaskId(null)
    setInstallingAddonName(null) // Clear installing state
    queryClient.invalidateQueries({ queryKey: addonKeys.list() })
    toast({
      message: `${selectedAddon?.display_name} installed successfully!`,
      variant: 'default',
      icon: 'checkmark-filled',
    })
  }

  const handleUninstall = async (addon: AddonInfo) => {
    // Check if any other installed add-ons depend on this one
    const dependents =
      addons?.filter(
        a =>
          a.installed &&
          a.name !== addon.name &&
          a.dependencies.includes(addon.name)
      ) || []

    if (dependents.length > 0) {
      // Show warning modal with list of dependent add-ons
      setAddonToRemove(addon)
      setDependentAddons(dependents)
      setShowDependencyWarning(true)
      return
    }

    // No dependents, show simple confirmation modal
    setAddonToRemove(addon)
    setShowRemoveConfirm(true)
  }

  const handleConfirmRemove = async () => {
    if (!addonToRemove) return

    setShowRemoveConfirm(false)

    try {
      await uninstallMutation.mutateAsync(addonToRemove.name)
      toast({
        message: `${addonToRemove.display_name} removed`,
        variant: 'default',
      })
    } catch (error) {
      toast({
        message: 'Failed to remove add-on',
        variant: 'destructive',
      })
    } finally {
      setAddonToRemove(null)
    }
  }

  const handleConfirmRemoveWithDependents = async () => {
    if (!addonToRemove) return

    setShowDependencyWarning(false)

    try {
      await uninstallMutation.mutateAsync(addonToRemove.name)
      toast({
        message: `${addonToRemove.display_name} removed`,
        variant: 'default',
      })
    } catch (error) {
      toast({
        message: 'Failed to remove add-on',
        variant: 'destructive',
      })
    } finally {
      setAddonToRemove(null)
      setDependentAddons([])
    }
  }

  const handleCascadeRemove = async () => {
    if (!addonToRemove) return

    setShowDependencyWarning(false)

    try {
      // Remove dependents first
      for (const dep of dependentAddons) {
        await uninstallMutation.mutateAsync(dep.name)
      }
      // Then remove the main addon
      await uninstallMutation.mutateAsync(addonToRemove.name)

      toast({
        message: `Removed ${addonToRemove.display_name} and ${dependentAddons.length} dependent add-on${dependentAddons.length > 1 ? 's' : ''}`,
        variant: 'default',
      })
    } catch (error) {
      toast({
        message: 'Failed to remove add-ons',
        variant: 'destructive',
      })
    } finally {
      setAddonToRemove(null)
      setDependentAddons([])
    }
  }

  return (
    <div className="min-h-screen bg-background pt-16">
      {/* Header */}
      <div className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                <button
                  onClick={() => navigate('/')}
                  className="hover:text-foreground transition-colors"
                >
                  LlamaFarm home
                </button>
                <span>/</span>
                <span>Manage add-ons</span>
              </div>
              <h1 className="text-2xl font-semibold text-foreground">
                Manage add-ons
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Extend LlamaFarm with powerful add-ons for speech, vision, and
                more
              </p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors"
            >
              Back
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* My add-ons section */}
        <section className="mb-12">
          <h2 className="text-lg font-semibold text-foreground mb-4">
            My add-ons
          </h2>

          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map(i => (
                <div
                  key={i}
                  className="h-40 rounded-lg bg-muted animate-pulse"
                />
              ))}
            </div>
          ) : installedAddons.length === 0 ? (
            <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-border bg-muted/20">
              <div className="text-center px-6">
                <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 border border-primary/20">
                  <Package className="w-5 h-5 text-primary" />
                </div>
                <div className="text-sm font-medium text-foreground mb-1">
                  No add-ons installed yet
                </div>
                <div className="text-xs text-muted-foreground">
                  Browse available add-ons below to get started
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {installedAddons.map(addon => (
                <AddonCard
                  key={addon.name}
                  addon={addon}
                  onRemove={handleUninstall}
                  installingAddonName={installingAddonName}
                />
              ))}
            </div>
          )}
        </section>

        {/* Available add-ons section */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">
              Available add-ons
            </h2>
            <SearchInput
              placeholder="Search add-ons"
              value={search}
              onChange={e => setSearch(e.target.value)}
              onClear={() => setSearch('')}
              containerClassName="w-64"
            />
          </div>

          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map(i => (
                <div
                  key={i}
                  className="h-40 rounded-lg bg-muted animate-pulse"
                />
              ))}
            </div>
          ) : availableAddons.length === 0 ? (
            <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-border bg-muted/20">
              <div className="text-center px-6">
                <div className="text-sm font-medium text-foreground mb-1">
                  {search ? 'No add-ons found' : 'All add-ons installed'}
                </div>
                <div className="text-xs text-muted-foreground">
                  {search
                    ? 'Try a different search term'
                    : 'You have all available add-ons installed'}
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {availableAddons.map(addon => (
                <AddonCard
                  key={addon.name}
                  addon={addon}
                  onInstall={handleInstall}
                  installingAddonName={installingAddonName}
                />
              ))}
            </div>
          )}
        </section>
      </div>

      {/* Installation side pane */}
      <AddonInstallSidePane
        open={showInstallPane}
        onOpenChange={setShowInstallPane}
        addons={addonsToInstall}
        onConfirm={handleInstallConfirm}
      />

      {/* Installation progress */}
      {installTaskId && selectedAddon && (
        <AddonInstallProgress
          taskId={installTaskId}
          addonName={selectedAddon.display_name}
          onComplete={handleInstallComplete}
          onCancel={() => {
            installAbortRef.current?.abort()
            setInstallTaskId(null)
            setInstallingAddonName(null)
          }}
        />
      )}

      {/* Dependency warning modal */}
      {showDependencyWarning && addonToRemove && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-[60] bg-black/70 backdrop-blur-sm"
            onClick={() => setShowDependencyWarning(false)}
            aria-hidden="true"
          />

          {/* Modal */}
          <div className="fixed left-1/2 top-1/2 z-[70] w-full max-w-md -translate-x-1/2 -translate-y-1/2">
            <div className="rounded-lg border border-border bg-card p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  ⚠️ Dependency Warning
                </h3>
                <p className="text-sm text-muted-foreground">
                  Removing <strong>{addonToRemove.display_name}</strong> will
                  break the following add-ons:
                </p>
              </div>

              <div className="mb-6 space-y-2">
                {dependentAddons.map(dep => (
                  <div
                    key={dep.name}
                    className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2"
                  >
                    <Package className="w-4 h-4 text-destructive" />
                    <div className="text-sm font-medium text-foreground">
                      {dep.display_name}
                    </div>
                  </div>
                ))}
              </div>

              <p className="text-xs text-muted-foreground mb-6">
                These add-ons require {addonToRemove.display_name} to function
                properly. They may stop working if you remove it.
              </p>

              <div className="space-y-3">
                <button
                  onClick={handleCascadeRemove}
                  className="w-full px-4 py-2 rounded-lg bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors text-sm font-medium"
                >
                  Remove all ({dependentAddons.length + 1} add-ons)
                </button>
                <button
                  onClick={handleConfirmRemoveWithDependents}
                  className="w-full px-4 py-2 rounded-lg border border-destructive/50 text-destructive hover:bg-destructive/10 transition-colors text-sm font-medium"
                >
                  Remove only {addonToRemove.display_name}
                </button>
                <button
                  onClick={() => setShowDependencyWarning(false)}
                  className="w-full px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors text-sm font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Simple removal confirmation modal */}
      {showRemoveConfirm && addonToRemove && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-[60] bg-black/70 backdrop-blur-sm"
            onClick={() => setShowRemoveConfirm(false)}
            aria-hidden="true"
          />

          {/* Modal */}
          <div className="fixed left-1/2 top-1/2 z-[70] w-full max-w-md -translate-x-1/2 -translate-y-1/2">
            <div className="rounded-lg border border-border bg-card p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  Remove Add-on
                </h3>
                <p className="text-sm text-muted-foreground">
                  Are you sure you want to remove{' '}
                  <strong>{addonToRemove.display_name}</strong>?
                </p>
              </div>

              <p className="text-xs text-muted-foreground mb-6">
                This action cannot be undone. You can reinstall it later if
                needed.
              </p>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowRemoveConfirm(false)}
                  className="flex-1 px-4 py-2 rounded-lg border border-border hover:bg-accent transition-colors text-sm font-medium"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmRemove}
                  className="flex-1 px-4 py-2 rounded-lg bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors text-sm font-medium"
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// Addon Card Component
function AddonCard({
  addon,
  onInstall,
  onRemove,
  installingAddonName,
}: {
  addon: AddonInfo
  onInstall?: (addon: AddonInfo) => void
  onRemove?: (addon: AddonInfo) => void
  installingAddonName?: string | null
}) {
  const isInstalled = addon.installed
  const isInstalling = installingAddonName === addon.name

  return (
    <div className="group rounded-lg p-4 bg-card border border-border hover:border-primary/30 transition-all">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <Package className="w-5 h-5 text-primary" />
          <div className="text-base font-semibold text-foreground">
            {addon.display_name}
          </div>
        </div>
        <div className="flex gap-2">
          {isInstalled && (
            <span className="px-2 py-0.5 rounded text-xs bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20">
              Installed
            </span>
          )}
          {isInstalling && (
            <span className="px-2 py-0.5 rounded text-xs bg-primary/10 text-primary border border-primary/20 animate-pulse">
              Installing...
            </span>
          )}
        </div>
      </div>

      <p className="text-xs text-muted-foreground mb-3 line-clamp-2">
        {addon.description}
      </p>

      {/* TODO: Backend should expose package count and calculate actual size */}
      {/* TODO: Backend should expose hardware_notes (CUDA/CPU/Metal support) */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground mb-4">
        <div className="flex items-center gap-1">
          <HardDrive className="w-3 h-3" />
          <span>~200MB</span>
        </div>
        {addon.dependencies.length > 0 && (
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span>
              {addon.dependencies.length} dep
              {addon.dependencies.length > 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>

      {isInstalled ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            {addon.installed_at && (
              <span>
                Added {new Date(addon.installed_at).toLocaleDateString()}
              </span>
            )}
            <span className="text-muted-foreground/60">v{addon.version}</span>
          </div>
          <button
            onClick={() => onRemove?.(addon)}
            disabled={isInstalling}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg border border-border hover:bg-destructive/10 hover:border-destructive/30 hover:text-destructive transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-3 h-3" />
            <span className="text-sm font-medium">Remove</span>
          </button>
        </div>
      ) : (
        <button
          onClick={() => onInstall?.(addon)}
          disabled={isInstalling}
          className="w-full py-2 px-4 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isInstalling ? 'Installing...' : 'ADD'}
        </button>
      )}
    </div>
  )
}
