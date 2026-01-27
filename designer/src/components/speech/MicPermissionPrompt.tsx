import { Mic, AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from '../ui/button'
import type { MicPermissionState } from '../../types/ml'

interface MicPermissionPromptProps {
  state: MicPermissionState
  onRequestPermission: () => void
  onContinueWithoutVoice?: () => void
  errorMessage?: string
  className?: string
}

export function MicPermissionPrompt({
  state,
  onRequestPermission,
  onContinueWithoutVoice,
  errorMessage,
  className = '',
}: MicPermissionPromptProps) {
  if (state === 'granted') {
    return null
  }

  if (state === 'prompt') {
    return (
      <div className={`rounded-xl border border-border bg-card/40 p-6 text-center ${className}`}>
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-500/20 border border-indigo-500/30">
          <Mic className="w-6 h-6 text-indigo-400" />
        </div>
        <h3 className="text-lg font-medium text-foreground mb-2">
          Microphone Access Required
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          LlamaFarm needs microphone access for voice input.
          <br />
          Audio is processed locally on your device.
        </p>
        <Button onClick={onRequestPermission}>
          <Mic className="w-4 h-4 mr-2" />
          Enable Microphone
        </Button>
      </div>
    )
  }

  if (state === 'denied') {
    return (
      <div className={`rounded-xl border border-amber-500/30 bg-amber-500/10 p-6 text-center ${className}`}>
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
          <AlertTriangle className="w-6 h-6 text-amber-400" />
        </div>
        <h3 className="text-lg font-medium text-foreground mb-2">
          Microphone Blocked
        </h3>
        <p className="text-sm text-muted-foreground mb-4">
          Microphone access was denied. To enable:
        </p>
        <div className="text-left bg-muted/50 rounded-lg p-3 mb-4 text-sm">
          <p className="mb-2">
            <strong>Chrome:</strong> Click the lock icon in the address bar &rarr; Site settings &rarr; Microphone &rarr; Allow
          </p>
          <p>
            <strong>Firefox:</strong> Click the permissions icon &rarr; Clear permission &rarr; Reload page
          </p>
        </div>
        <div className="flex gap-2 justify-center">
          <Button variant="outline" onClick={onRequestPermission}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Check Again
          </Button>
          {onContinueWithoutVoice && (
            <Button variant="secondary" onClick={onContinueWithoutVoice}>
              Continue without voice
            </Button>
          )}
        </div>
      </div>
    )
  }

  // error state
  return (
    <div className={`rounded-xl border border-amber-500/30 bg-amber-500/10 p-6 text-center ${className}`}>
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
        <AlertTriangle className="w-6 h-6 text-amber-400" />
      </div>
      <h3 className="text-lg font-medium text-foreground mb-2">
        Microphone Unavailable
      </h3>
      <p className="text-sm text-muted-foreground mb-2">
        Could not access microphone:
      </p>
      {errorMessage ? (
        <p className="text-sm text-amber-400 mb-4">{errorMessage}</p>
      ) : (
        <ul className="text-sm text-muted-foreground mb-4 text-left list-disc list-inside">
          <li>No microphone connected</li>
          <li>Another app may be using it</li>
        </ul>
      )}
      <div className="flex gap-2 justify-center">
        <Button variant="outline" onClick={onRequestPermission}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Try Again
        </Button>
        {onContinueWithoutVoice && (
          <Button variant="secondary" onClick={onContinueWithoutVoice}>
            Continue without voice
          </Button>
        )}
      </div>
    </div>
  )
}
