import { useState, useCallback } from 'react'
import { Plus, Play, Pause, Trash2, Mic } from 'lucide-react'
import { Button } from '../ui/button'
import { VoiceRecorder } from './VoiceRecorder'
import type { VoiceClone } from '../../types/ml'

interface VoiceCloningProps {
  voices: VoiceClone[]
  onAddVoice: (voice: VoiceClone) => void
  onDeleteVoice: (voiceId: string) => void
  onPreviewVoice: (voiceId: string) => void
  previewingVoiceId?: string | null
  className?: string
}

export function VoiceCloning({
  voices,
  onAddVoice,
  onDeleteVoice,
  onPreviewVoice,
  previewingVoiceId,
  className = '',
}: VoiceCloningProps) {
  const [isRecording, setIsRecording] = useState(false)

  const handleSaveRecording = useCallback((name: string, audioBlob: Blob, duration: number) => {
    const newVoice: VoiceClone = {
      id: `custom-${Date.now()}`,
      name,
      duration,
      createdAt: new Date().toISOString().split('T')[0],
      audioBlob,
    }
    onAddVoice(newVoice)
    setIsRecording(false)
  }, [onAddVoice])

  return (
    <div className={`rounded-xl border border-border bg-card/40 p-4 ${className}`}>
      {/* Header */}
      <div className="mb-3">
        <h3 className="text-sm font-medium">Voice Cloning</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Create custom voices from audio samples
        </p>
      </div>

      {/* Recording mode */}
      {isRecording ? (
        <VoiceRecorder
          onSave={handleSaveRecording}
          onCancel={() => setIsRecording(false)}
        />
      ) : (
        <>
          {/* Voice list */}
          {voices.length > 0 && (
            <div className="space-y-2 mb-3">
              {voices.map((voice) => (
                <VoiceCloneItem
                  key={voice.id}
                  voice={voice}
                  isPlaying={previewingVoiceId === voice.id}
                  onPlay={() => onPreviewVoice(voice.id)}
                  onDelete={() => onDeleteVoice(voice.id)}
                />
              ))}
            </div>
          )}

          {/* Empty state */}
          {voices.length === 0 && (
            <div className="py-6 text-center">
              <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-muted">
                <Mic className="w-5 h-5 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground mb-3">
                No custom voices yet
              </p>
            </div>
          )}

          {/* Add new voice button */}
          <Button
            variant="outline"
            className="w-full"
            onClick={() => setIsRecording(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            Record New Voice
          </Button>
        </>
      )}
    </div>
  )
}

// Individual voice clone item
interface VoiceCloneItemProps {
  voice: VoiceClone
  isPlaying: boolean
  onPlay: () => void
  onDelete: () => void
}

function VoiceCloneItem({ voice, isPlaying, onPlay, onDelete }: VoiceCloneItemProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  }

  return (
    <div className="flex items-center gap-3 p-2 rounded-lg border border-border/50 bg-muted/20 hover:bg-muted/40 transition-colors">
      {/* Voice icon */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
        <Mic className="w-4 h-4 text-indigo-400" />
      </div>

      {/* Voice info */}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">{voice.name}</div>
        <div className="text-xs text-muted-foreground">
          {voice.duration}s • {formatDate(voice.createdAt)}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onPlay}
          aria-label={isPlaying ? 'Stop' : 'Play'}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-destructive"
          onClick={onDelete}
          aria-label="Delete voice"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
