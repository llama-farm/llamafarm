import { useEffect, useState } from 'react'
import { Button } from '../ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import modelService from '../../api/modelService'

interface GGUFOption {
  filename: string
  quantization: string | null
  size_bytes: number
  size_human: string
}

interface ModelDetailsModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  modelName: string
  baseModelId: string
  defaultQuantization: string
  onSelect: (modelIdentifier: string) => void
}

export function ModelDetailsModal({
  open,
  onOpenChange,
  modelName = '',
  baseModelId = '',
  defaultQuantization = '',
  onSelect,
}: ModelDetailsModalProps) {
  const [options, setOptions] = useState<GGUFOption[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedQuantization, setSelectedQuantization] = useState<
    string | null
  >(null)

  useEffect(() => {
    // Create AbortController for this effect run
    const abortController = new AbortController()
    const signal = abortController.signal

    // Capture the current baseModelId to check against in callbacks
    const currentBaseModelId = baseModelId

    if (open && currentBaseModelId && currentBaseModelId.trim()) {
      setIsLoading(true)
      setError(null)
      setOptions([])
      setSelectedQuantization(defaultQuantization || null)

      modelService
        .getGGUFOptions(currentBaseModelId, signal)
        .then(data => {
          // Guard: Check if request was aborted or baseModelId changed
          if (signal.aborted || baseModelId !== currentBaseModelId) {
            return
          }

          if (data && data.options) {
            setOptions(data.options)
            // Set default selection if available
            if (data.options.length > 0) {
              const defaultOption = data.options.find(
                opt => opt.quantization === defaultQuantization
              )
              if (defaultOption) {
                setSelectedQuantization(defaultQuantization)
              } else {
                // If default not found, select first option
                setSelectedQuantization(data.options[0].quantization)
              }
            }
          } else {
            setError('No options returned from server')
          }
        })
        .catch(err => {
          // Ignore abort errors - they're expected when cleaning up
          if (
            signal.aborted ||
            (err as any)?.name === 'AbortError' ||
            (err as any)?.code === 'ERR_CANCELED'
          ) {
            return
          }

          // Guard: Check if baseModelId changed
          if (baseModelId !== currentBaseModelId) {
            return
          }

          console.error('Error loading GGUF options:', err)
          setError(
            err?.message ||
              err?.response?.data?.detail ||
              'Failed to load download options. Please try again.'
          )
        })
        .finally(() => {
          // Guard: Only update loading state if this is still the current request
          if (!signal.aborted && baseModelId === currentBaseModelId) {
            setIsLoading(false)
          }
        })
    } else if (!open) {
      // Reset state when modal closes
      setOptions([])
      setError(null)
      setSelectedQuantization(null)
    }

    // Cleanup: Cancel all in-flight requests when effect re-runs or unmounts
    return () => {
      abortController.abort()
    }
  }, [open, baseModelId, defaultQuantization])

  // Early return if required props are missing (after all hooks)
  if (!baseModelId || !onSelect) {
    return null
  }

  const handleSelect = () => {
    if (!selectedQuantization) return

    const selectedOption = options.find(
      opt => opt.quantization === selectedQuantization
    )
    if (!selectedOption) return

    // Construct full model identifier with quantization
    const modelIdentifier = `${baseModelId}:${selectedQuantization}`
    onSelect(modelIdentifier)
    onOpenChange(false)
  }

  if (!baseModelId) {
    return null
  }

  // Sort options by quantization preference (smaller/faster first, then by size)
  const sortedOptions = [...options].sort((a, b) => {
    // If quantization is null, put at end
    if (!a.quantization) return 1
    if (!b.quantization) return -1
    // Sort by size bytes (smaller first)
    return a.size_bytes - b.size_bytes
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
        <DialogTitle>Choose download option</DialogTitle>
        <DialogDescription>
          <div className="mt-2 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <div className="text-sm text-muted-foreground">
                {options.length > 0
                  ? `${options.length} download options available`
                  : 'Loading download options...'}
              </div>
              {options.length > 0 && (
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  title="Information about download options"
                >
                  <FontIcon type="info" className="w-4 h-4" />
                </button>
              )}
            </div>

            {isLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader
                  size={24}
                  className="border-blue-400 dark:border-blue-100"
                />
              </div>
            )}

            {error && (
              <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            {!isLoading && !error && sortedOptions.length > 0 && (
              <div className="flex flex-col gap-2 max-h-[400px] overflow-y-auto border border-border rounded-lg p-2">
                {sortedOptions
                  .filter(opt => opt.quantization) // Filter out null quantization options
                  .map((option, index) => {
                    const isSelected =
                      option.quantization === selectedQuantization
                    return (
                      <button
                        key={`${option.quantization}-${index}`}
                        type="button"
                        onClick={() =>
                          setSelectedQuantization(option.quantization)
                        }
                        className={`flex items-center gap-3 p-3 rounded-lg border transition-colors text-left ${
                          isSelected
                            ? 'bg-accent/80 border-primary'
                            : 'border-border hover:bg-accent/50'
                        }`}
                      >
                        <div className="flex-shrink-0">
                          {isSelected ? (
                            <FontIcon
                              type="checkmark-filled"
                              className="w-5 h-5 text-primary"
                            />
                          ) : (
                            <div className="w-5 h-5 rounded-full border-2 border-muted-foreground" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0 flex items-center gap-3">
                          <span className="text-sm font-medium px-3 py-1 rounded-md bg-primary/10 text-primary border border-primary/20">
                            {option.quantization || 'Unknown'}
                          </span>
                          <span className="text-sm text-muted-foreground truncate">
                            {modelName}
                          </span>
                        </div>
                        <div className="flex-shrink-0 text-sm font-medium text-foreground">
                          {option.size_human}
                        </div>
                      </button>
                    )
                  })}
              </div>
            )}

            {!isLoading && !error && options.length === 0 && (
              <div className="p-6 text-center text-sm text-muted-foreground">
                No download options available for this model.
              </div>
            )}
          </div>
        </DialogDescription>
        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSelect}
            disabled={!selectedQuantization || isLoading}
          >
            Download and add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
