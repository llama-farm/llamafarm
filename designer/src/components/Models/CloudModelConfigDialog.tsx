import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import FontIcon from '../../common/FontIcon'

interface CloudModelConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  modelName: string
  provider: string
  defaultBaseUrl?: string
  onConfigure: (config: { apiKey: string; baseUrl: string }) => void
  onCancel: () => void
}

export function CloudModelConfigDialog({
  open,
  onOpenChange,
  modelName,
  provider,
  defaultBaseUrl = '',
  onConfigure,
  onCancel,
}: CloudModelConfigDialogProps) {
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl)
  const [showApiKey, setShowApiKey] = useState(false)

  // Sync baseUrl when dialog opens or defaultBaseUrl changes
  useEffect(() => {
    if (open) {
      setBaseUrl(defaultBaseUrl)
    }
  }, [open, defaultBaseUrl])

  const handleConfigure = () => {
    if (!apiKey.trim()) {
      return
    }
    onConfigure({
      apiKey: apiKey.trim(),
      baseUrl: baseUrl.trim() || defaultBaseUrl,
    })
    // Reset form
    setApiKey('')
    setBaseUrl(defaultBaseUrl)
    setShowApiKey(false)
  }

  const getProviderName = () => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return 'OpenAI'
      case 'xai':
        return 'xAI (Grok)'
      case 'togetherai':
        return 'Together AI'
      default:
        return provider
    }
  }

  const getApiKeyPlaceholder = () => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return 'sk-...'
      case 'xai':
        return 'xai-...'
      case 'togetherai':
        return 'together-...'
      default:
        return 'Enter your API key'
    }
  }

  const getApiKeyEnvVar = () => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return 'OPENAI_API_KEY'
      case 'xai':
        return 'XAI_API_KEY'
      case 'togetherai':
        return 'TOGETHER_API_KEY'
      default:
        return 'API_KEY'
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogTitle>Configure Cloud Model</DialogTitle>
        <DialogDescription>
          <div className="mt-2 flex flex-col gap-4">
            {/* Info section */}
            <div className="rounded-lg bg-secondary/50 p-3">
              <div className="flex items-start gap-3">
                <FontIcon
                  type="integration"
                  className="w-4 h-4 text-primary flex-shrink-0 mt-0.5"
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground mb-1">
                    Cloud API Configuration
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Configure API access for <strong>{modelName}</strong> via{' '}
                    <strong>{getProviderName()}</strong>
                  </div>
                </div>
              </div>
            </div>

            {/* API Key field */}
            <div className="space-y-2">
              <Label htmlFor="api-key" className="text-sm font-medium">
                API Key
              </Label>
              <div className="relative">
                <Input
                  id="api-key"
                  type={showApiKey ? 'text' : 'password'}
                  placeholder={getApiKeyPlaceholder()}
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  className="pr-10"
                  autoComplete="off"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-secondary rounded"
                >
                  <FontIcon
                    type={showApiKey ? 'eye-off' : 'eye'}
                    className="w-4 h-4 text-muted-foreground"
                  />
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                You can also set this via the <code>{getApiKeyEnvVar()}</code>{' '}
                environment variable
              </p>
            </div>

            {/* Base URL field */}
            <div className="space-y-2">
              <Label htmlFor="base-url" className="text-sm font-medium">
                Base URL
              </Label>
              <Input
                id="base-url"
                type="text"
                placeholder={defaultBaseUrl}
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Default: <code>{defaultBaseUrl}</code>
              </p>
            </div>

            {/* Warning if no API key */}
            {!apiKey.trim() && (
              <div className="rounded-lg bg-orange-500/10 border border-orange-500/30 p-3">
                <div className="flex items-start gap-2">
                  <FontIcon
                    type="alert-triangle"
                    className="w-4 h-4 text-orange-500 flex-shrink-0 mt-0.5"
                  />
                  <div className="text-xs text-orange-600 dark:text-orange-400">
                    An API key is required to use this model
                  </div>
                </div>
              </div>
            )}
          </div>
        </DialogDescription>

        {/* Actions */}
        <DialogFooter className="gap-2 mt-4">
          <Button
            variant="secondary"
            onClick={() => {
              onCancel()
              setApiKey('')
              setBaseUrl(defaultBaseUrl)
              setShowApiKey(false)
            }}
          >
            Cancel
          </Button>
          <Button onClick={handleConfigure} disabled={!apiKey.trim()}>
            Configure
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
