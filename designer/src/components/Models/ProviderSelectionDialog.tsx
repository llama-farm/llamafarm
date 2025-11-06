import { useState } from 'react'
import { Button } from '../ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import { Label } from '../ui/label'
import FontIcon from '../../common/FontIcon'
import { type ProviderInfo } from '../../utils/modelCatalog'

interface ProviderSelectionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  variantName: string
  parameters: string
  downloadSize: string
  providers: ProviderInfo[]
  onSelectProvider: (provider: ProviderInfo) => void
}

export function ProviderSelectionDialog({
  open,
  onOpenChange,
  variantName,
  parameters,
  downloadSize,
  providers,
  onSelectProvider,
}: ProviderSelectionDialogProps) {
  const [selectedProvider, setSelectedProvider] = useState<ProviderInfo | null>(
    null
  )
  const [copiedProvider, setCopiedProvider] = useState<string | null>(null)

  const handleCopyCommand = (provider: ProviderInfo) => {
    if (provider.downloadCommand) {
      navigator.clipboard.writeText(provider.downloadCommand)
      setCopiedProvider(provider.provider)
      setTimeout(() => setCopiedProvider(null), 2000)
    }
  }

  const handleSelect = () => {
    if (selectedProvider) {
      onSelectProvider(selectedProvider)
      onOpenChange(false)
      setSelectedProvider(null)
    }
  }

  const formatBadgeText = (runtime: string): string => {
    return runtime.charAt(0).toUpperCase() + runtime.slice(1)
  }

  const formatFormatBadge = (format: string): string => {
    return format.toUpperCase()
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogTitle>Select provider for {variantName}</DialogTitle>
        <DialogDescription>
          <div className="mt-2 flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="text-muted-foreground">Model</div>
              <div>{variantName}</div>
              <div className="text-muted-foreground">Parameter size</div>
              <div>{parameters}</div>
              <div className="text-muted-foreground">Download size</div>
              <div>{downloadSize}</div>
            </div>

            <div>
              <Label className="text-sm font-medium mb-2 block">
                Choose runtime provider
              </Label>
              <div className="flex flex-col gap-2">
                {providers.map(provider => (
                  <div
                    key={provider.provider}
                    className={`rounded-lg border-2 p-4 cursor-pointer transition-colors ${
                      selectedProvider?.provider === provider.provider
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50 hover:bg-accent/30'
                    }`}
                    onClick={() => setSelectedProvider(provider)}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="font-medium">
                            {formatBadgeText(provider.runtime)}
                          </div>
                          <div className="text-xs px-2 py-0.5 rounded-full bg-primary/15 text-primary border border-primary/30">
                            {formatFormatBadge(provider.format)}
                          </div>
                        </div>

                        {provider.notes && (
                          <div className="text-xs text-muted-foreground mb-2">
                            {provider.notes}
                          </div>
                        )}

                        {provider.downloadCommand && (
                          <div className="mt-2">
                            <div className="text-xs text-muted-foreground mb-1">
                              Download command:
                            </div>
                            <div className="flex items-center gap-2">
                              <code className="flex-1 text-xs bg-secondary px-2 py-1.5 rounded font-mono overflow-x-auto">
                                {provider.downloadCommand}
                              </code>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-7 px-2 text-xs"
                                onClick={e => {
                                  e.stopPropagation()
                                  handleCopyCommand(provider)
                                }}
                              >
                                {copiedProvider === provider.provider ? (
                                  <>
                                    <FontIcon
                                      type="checkmark-filled"
                                      className="w-3 h-3 mr-1"
                                    />
                                    Copied
                                  </>
                                ) : (
                                  'Copy'
                                )}
                              </Button>
                            </div>
                          </div>
                        )}

                        {!provider.downloadCommand &&
                          provider.runtime === 'universal' && (
                            <div className="text-xs text-muted-foreground mt-2">
                              ✓ Auto-downloads on first use
                            </div>
                          )}
                      </div>

                      <div className="flex-shrink-0">
                        <div
                          className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                            selectedProvider?.provider === provider.provider
                              ? 'border-primary bg-primary'
                              : 'border-border'
                          }`}
                        >
                          {selectedProvider?.provider === provider.provider && (
                            <div className="w-2 h-2 rounded-full bg-primary-foreground" />
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {providers.length === 0 && (
              <div className="text-sm text-muted-foreground text-center py-4">
                No providers available for this model.
              </div>
            )}
          </div>
        </DialogDescription>
        <DialogFooter>
          <Button
            variant="secondary"
            onClick={() => {
              onOpenChange(false)
              setSelectedProvider(null)
            }}
          >
            Cancel
          </Button>
          <Button disabled={!selectedProvider} onClick={handleSelect}>
            Continue with{' '}
            {selectedProvider
              ? formatBadgeText(selectedProvider.runtime)
              : 'provider'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
