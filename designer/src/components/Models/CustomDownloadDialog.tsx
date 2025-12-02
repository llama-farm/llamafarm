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
import { PromptSetSelector } from './PromptSetSelector'
import { sanitizeModelName, formatBytes } from '../../utils/modelUtils'

interface CustomDownloadDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  promptSetNames: string[]
  customModelInput: string
  setCustomModelInput: (value: string) => void
  customModelName: string
  setCustomModelName: (value: string) => void
  customModelDescription: string
  setCustomModelDescription: (value: string) => void
  customSelectedPromptSets: string[]
  setCustomSelectedPromptSets: (sets: string[]) => void
  customDownloadState: 'idle' | 'downloading' | 'success' | 'error'
  customDownloadProgress: number
  customDownloadError: string
  customModelNameError?: string
  onClearModelNameError?: () => void
  downloadedBytes: number
  totalBytes: number
  estimatedTimeRemaining: string
  onDownload: () => void
  onMoveToBackground: () => void
}

export function CustomDownloadDialog({
  open,
  onOpenChange,
  promptSetNames,
  customModelInput,
  setCustomModelInput,
  customModelName,
  setCustomModelName,
  customModelDescription,
  setCustomModelDescription,
  customSelectedPromptSets,
  setCustomSelectedPromptSets,
  customDownloadState,
  customDownloadProgress,
  customDownloadError,
  customModelNameError,
  onClearModelNameError,
  downloadedBytes,
  totalBytes,
  estimatedTimeRemaining,
  onDownload,
  onMoveToBackground,
}: CustomDownloadDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogTitle>Download model from HuggingFace</DialogTitle>
        <DialogDescription>
          <div className="mt-2 flex flex-col gap-3">
            <p className="text-sm">
              Enter the model name from HuggingFace to download and add it to
              your project.
            </p>

            {/* Model input field */}
            <div>
              <label
                className="text-xs text-muted-foreground"
                htmlFor="custom-model-input"
              >
                HuggingFace Model Name
              </label>
              <input
                id="custom-model-input"
                type="text"
                placeholder="e.g., meta-llama/Llama-3.2-1B"
                value={customModelInput}
                onChange={e => {
                  setCustomModelInput(e.target.value)
                  // Auto-populate name if empty - sanitize it
                  if (!customModelName) {
                    const sanitized = sanitizeModelName(e.target.value)
                    setCustomModelName(sanitized)
                    // Clear any existing name error since we're auto-filling a valid sanitized name
                    if (customModelNameError && onClearModelNameError) {
                      onClearModelNameError()
                    }
                  }
                }}
                disabled={customDownloadState === 'downloading'}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
              />
              <div className="text-xs text-muted-foreground mt-1">
                Find models at{' '}
                <a
                  href="https://huggingface.co/models"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  HuggingFace
                </a>
              </div>
            </div>

            {/* Name field */}
            <div>
              <label
                className="text-xs text-muted-foreground"
                htmlFor="custom-model-name"
              >
                Name
              </label>
              <input
                id="custom-model-name"
                type="text"
                placeholder="Enter model name"
                value={customModelName}
                onChange={e => {
                  const sanitized = sanitizeModelName(e.target.value)
                  setCustomModelName(sanitized)
                  // Clear error when user types
                  if (customModelNameError && onClearModelNameError) {
                    onClearModelNameError()
                  }
                }}
                disabled={customDownloadState === 'downloading'}
                className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border ${
                  customModelNameError ? 'border-destructive' : 'border-input'
                } text-foreground`}
              />
              {customModelNameError && (
                <div className="text-xs text-destructive mt-1">
                  {customModelNameError}
                </div>
              )}
              <div className="text-xs text-muted-foreground mt-1">
                Only letters, numbers, underscores (_), and hyphens (-) are
                allowed. No spaces.
              </div>
            </div>

            {/* Description field */}
            <div>
              <label
                className="text-xs text-muted-foreground"
                htmlFor="custom-model-description"
              >
                Description (optional)
              </label>
              <textarea
                id="custom-model-description"
                rows={2}
                placeholder="Enter model description"
                value={customModelDescription}
                onChange={e => setCustomModelDescription(e.target.value)}
                disabled={customDownloadState === 'downloading'}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
              />
            </div>

            {/* Prompt sets dropdown */}
            <PromptSetSelector
              promptSetNames={promptSetNames}
              selectedPromptSets={customSelectedPromptSets}
              onTogglePromptSet={(name, checked) => {
                if (checked) {
                  setCustomSelectedPromptSets([
                    ...customSelectedPromptSets,
                    name,
                  ])
                } else {
                  setCustomSelectedPromptSets(
                    customSelectedPromptSets.filter(s => s !== name)
                  )
                }
              }}
              onClearPromptSets={() => setCustomSelectedPromptSets([])}
              disabled={customDownloadState === 'downloading'}
              triggerId="custom-prompt-sets-trigger"
              label="Prompt sets"
            />

            {/* Metadata */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="text-muted-foreground">Provider</div>
              <div>Ollama</div>
              <div className="text-muted-foreground">Source</div>
              <div>HuggingFace</div>
            </div>

            {/* Progress bar */}
            {customDownloadState === 'downloading' && (
              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    Downloading... {formatBytes(downloadedBytes)} /{' '}
                    {formatBytes(totalBytes)}
                  </span>
                  <span className="text-muted-foreground">
                    {customDownloadProgress}%{' '}
                    {estimatedTimeRemaining && `• ${estimatedTimeRemaining}`}
                  </span>
                </div>
                <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${customDownloadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error message */}
            {customDownloadState === 'error' && customDownloadError && (
              <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
                <p className="text-sm text-destructive">
                  {customDownloadError}
                </p>
              </div>
            )}
          </div>
        </DialogDescription>
        <DialogFooter>
          {customDownloadState === 'downloading' ? (
            <Button variant="secondary" onClick={onMoveToBackground}>
              Download in background
            </Button>
          ) : (
            <Button variant="secondary" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
          )}
          <Button
            disabled={
              customDownloadState === 'downloading' ||
              !customModelInput.trim() ||
              !customModelName.trim()
            }
            onClick={onDownload}
          >
            {customDownloadState === 'downloading' && (
              <span className="mr-2 inline-flex">
                <Loader
                  size={14}
                  className="border-blue-400 dark:border-blue-100"
                />
              </span>
            )}
            {customDownloadState === 'success' && (
              <span className="mr-2 inline-flex">
                <FontIcon type="checkmark-filled" className="w-4 h-4" />
              </span>
            )}
            Download and add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
