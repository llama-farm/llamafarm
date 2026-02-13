import { X, Package, Zap, HardDrive, Cpu } from 'lucide-react'
import type { AddonInfo } from '../../types/addons'
import { useEffect, useState } from 'react'
import { Checkbox } from '../ui/checkbox'

interface AddonInstallSidePaneProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  addons: AddonInfo[]
  onConfirm: (selectedAddons: string[]) => void
}

/**
 * Reusable side panel component for add-on installation confirmation
 *
 * Shows addon details, dependencies, size estimates, and confirmation actions.
 * Slides in from the right with overlay backdrop.
 * Users can select which addons to install (STT, TTS, or both).
 */
export function AddonInstallSidePane({
  open,
  onOpenChange,
  addons,
  onConfirm,
}: AddonInstallSidePaneProps) {
  // Track which addons are selected (default: primary addons checked, dependencies auto-included)
  const [selectedAddons, setSelectedAddons] = useState<Set<string>>(() => {
    const selected = new Set<string>()
    // Auto-select primary addons (those that aren't dependencies of others)
    addons.forEach(addon => {
      const isDependency = addons.some(other =>
        other.name !== addon.name && other.dependencies.includes(addon.name)
      )
      if (!isDependency) {
        selected.add(addon.name)
      }
    })
    return selected
  })

  // Reset selection when panel opens
  useEffect(() => {
    if (open) {
      const selected = new Set<string>()
      // Auto-select primary addons (those that aren't dependencies of others)
      addons.forEach(addon => {
        const isDependency = addons.some(other =>
          other.name !== addon.name && other.dependencies.includes(addon.name)
        )
        if (!isDependency) {
          selected.add(addon.name)
        }
      })
      setSelectedAddons(selected)
    }
  }, [open, addons])

  // Separate addons into selectable (primary) and dependencies
  // A primary addon is one that isn't a dependency of any other addon in the list
  const selectableAddons = addons.filter(addon => {
    const isDependency = addons.some(other =>
      other.name !== addon.name && other.dependencies.includes(addon.name)
    )
    return !isDependency
  })

  const dependencyAddons = addons.filter(addon => {
    const isDependency = addons.some(other =>
      other.name !== addon.name && other.dependencies.includes(addon.name)
    )
    return isDependency
  })

  // Get full dependency closure for selected addons (includes transitive deps)
  const selectedDependencies = (() => {
    const needed = new Set<string>()
    const collect = (addonName: string) => {
      const addon = addons.find(a => a.name === addonName)
      if (!addon) return
      for (const depName of addon.dependencies) {
        if (!needed.has(depName)) {
          needed.add(depName)
          collect(depName) // recurse for transitive deps
        }
      }
    }
    selectedAddons.forEach(name => collect(name))
    return dependencyAddons.filter(dep => needed.has(dep.name))
  })()

  // Calculate size based on selected addons + their dependencies
  const estimatedSize = (() => {
    const sizes: Record<string, number> = {
      'stt': 150,
      'tts': 200,
      'onnxruntime': 100,
    }

    let total = 0
    // Add selected primary addons
    selectedAddons.forEach(name => {
      total += sizes[name] || 50
    })
    // Add dependencies that will be auto-installed (only for selected addons)
    selectedDependencies.forEach(dep => {
      total += sizes[dep.name] || 50
    })
    return total
  })()

  // Toggle addon selection
  const toggleAddon = (addonName: string) => {
    const newSelected = new Set(selectedAddons)
    if (newSelected.has(addonName)) {
      newSelected.delete(addonName)
    } else {
      newSelected.add(addonName)
    }
    setSelectedAddons(newSelected)
  }

  // Determine which addons to install (selected + their dependencies)
  const getAddonsToInstall = (): string[] => {
    const toInstall: string[] = []
    // Add selected primary addons
    selectedAddons.forEach(name => toInstall.push(name))
    // Add dependencies for selected addons only
    selectedDependencies.forEach(dep => toInstall.push(dep.name))
    return toInstall
  }

  // Handle escape key to close
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        onOpenChange(false)
      }
    }

    if (open) {
      document.addEventListener('keydown', handleEscape)
      // Prevent body scroll when panel open
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [open, onOpenChange])

  if (!open) return null

  return (
    <>
      {/* Overlay backdrop */}
      <div
        className="fixed inset-0 z-[60] bg-black/70 backdrop-blur-sm animate-in fade-in-0 duration-300"
        onClick={() => onOpenChange(false)}
        aria-hidden="true"
      />

      {/* Side Panel */}
      <div
        className={`fixed right-0 top-0 bottom-0 w-[440px] z-[70] bg-background border-l border-border/50 shadow-2xl
          transform transition-transform duration-300 ease-out translate-x-0`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="addon-panel-title"
      >
        {/* Header */}
        <div className="p-6 border-b border-border/50 bg-gradient-to-b from-muted/30 to-background">
          <div className="flex items-center gap-3 mb-2">
            <h2 id="addon-panel-title" className="text-xl font-semibold text-foreground">
              Install Speech Add-ons
            </h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Add powerful speech capabilities to your project
          </p>
          <button
            onClick={() => onOpenChange(false)}
            className="absolute top-5 right-5 rounded-lg p-1.5 opacity-70 hover:opacity-100 hover:bg-accent transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            aria-label="Close panel"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 240px)' }}>
          {/* Addons List */}
          <div className="space-y-4">
            {/* Selectable Add-ons */}
            {selectableAddons.length > 0 && (
              <div className="space-y-2.5">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Select Features
                </div>
                {selectableAddons.map(addon => (
                  <div
                    key={addon.name}
                    className={`group relative flex items-start gap-3 p-4 rounded-lg border transition-all cursor-pointer ${
                      selectedAddons.has(addon.name)
                        ? 'border-primary/50 bg-primary/5 shadow-sm'
                        : 'border-border bg-card hover:border-border hover:shadow-sm'
                    }`}
                    onClick={() => toggleAddon(addon.name)}
                  >
                    <Checkbox
                      checked={selectedAddons.has(addon.name)}
                      onCheckedChange={() => toggleAddon(addon.name)}
                      className="mt-0.5"
                      aria-label={`Select ${addon.display_name}`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Package className="w-4 h-4 text-primary shrink-0" />
                        <div className="font-semibold text-sm text-foreground">
                          {addon.display_name}
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground leading-relaxed">
                        {addon.description}
                      </div>
                      {addon.dependencies.length > 0 && (
                        <div className="text-xs text-muted-foreground/70 mt-2 flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          <span>Requires: {addon.dependencies.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Auto-included Dependencies */}
            {selectedDependencies.length > 0 && selectedAddons.size > 0 && (
              <div className="space-y-2.5">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Required Dependencies
                </div>
                {selectedDependencies.map(addon => (
                  <div
                    key={addon.name}
                    className="flex items-start gap-3 p-4 rounded-lg border border-border/50 bg-muted/30"
                  >
                    <Checkbox
                      checked
                      disabled
                      className="mt-0.5 opacity-40"
                      aria-label={`${addon.display_name} will be installed automatically`}
                    />
                    <div className="flex-1 min-w-0 opacity-70">
                      <div className="flex items-center gap-2 mb-1">
                        <Zap className="w-4 h-4 text-muted-foreground shrink-0" />
                        <div className="font-semibold text-sm text-foreground">
                          {addon.display_name}
                        </div>
                        <span className="text-xs text-muted-foreground/60 ml-auto">(auto-included)</span>
                      </div>
                      <div className="text-xs text-muted-foreground leading-relaxed">
                        {addon.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* System Requirements */}
          <div className="space-y-2.5 p-4 rounded-lg bg-gradient-to-br from-muted/40 to-muted/20 border border-border/50">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
              System Requirements
            </div>
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center justify-center w-8 h-8 rounded-md bg-background/60 border border-border/50">
                <HardDrive className="w-4 h-4 text-foreground" />
              </div>
              <div className="flex-1">
                <div className="text-xs text-muted-foreground">Disk Space</div>
                <div className="font-semibold text-foreground">~{estimatedSize} MB</div>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center justify-center w-8 h-8 rounded-md bg-background/60 border border-border/50">
                <Cpu className="w-4 h-4 text-foreground" />
              </div>
              <div className="flex-1">
                <div className="text-xs text-muted-foreground">Memory</div>
                <div className="font-semibold text-foreground">2GB RAM</div>
              </div>
            </div>
          </div>

          {/* Info Note */}
          <div className="p-4 rounded-lg bg-muted/30 border border-border/50 text-sm">
            <p className="text-foreground/90 leading-relaxed">
              <span className="font-semibold">Note:</span> Installation takes 3-5 minutes. Services will restart automatically (~30-60s). You can minimize this panel and continue working.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-border/50 bg-gradient-to-t from-muted/20 to-background">
          <div className="flex flex-col gap-2.5">
            <button
              onClick={() => onConfirm(getAddonsToInstall())}
              disabled={selectedAddons.size === 0}
              className="w-full py-3 px-4 bg-primary text-primary-foreground rounded-lg font-semibold shadow-sm hover:shadow-md hover:bg-primary/90 transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
            >
              Install {selectedAddons.size > 0 ? `${selectedAddons.size} add-on${selectedAddons.size > 1 ? 's' : ''}` : 'add-ons'}
            </button>
            <button
              onClick={() => onOpenChange(false)}
              className="w-full py-2.5 px-4 border border-border/50 rounded-lg font-medium hover:bg-accent/50 transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              Cancel
            </button>
            <a
              href="https://docs.llamafarm.ai/addons"
              target="_blank"
              rel="noopener noreferrer"
              className="text-center text-xs text-muted-foreground hover:text-foreground transition-colors hover:underline focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 rounded py-1"
            >
              Learn more about add-ons →
            </a>
          </div>
        </div>
      </div>
    </>
  )
}
