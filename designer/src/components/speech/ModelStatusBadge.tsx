import { Badge } from '../ui/badge'
import type { SpeechModelStatus } from '../../types/ml'

interface ModelStatusBadgeProps {
  status: SpeechModelStatus
  progress?: number
  className?: string
}

export function ModelStatusBadge({ status, progress, className = '' }: ModelStatusBadgeProps) {
  switch (status) {
    case 'ready':
      return (
        <Badge className={`bg-green-500/20 text-green-400 border border-green-500/30 ${className}`}>
          Ready
        </Badge>
      )
    case 'downloading':
      return (
        <Badge className={`bg-blue-500/20 text-blue-400 border border-blue-500/30 ${className}`}>
          {progress !== undefined ? `${progress}%` : 'Downloading...'}
        </Badge>
      )
    case 'not_downloaded':
      return (
        <Badge className={`bg-muted text-muted-foreground border border-border ${className}`}>
          Not Downloaded
        </Badge>
      )
    case 'error':
      return (
        <Badge className={`bg-destructive/20 text-destructive border border-destructive/30 ${className}`}>
          Error
        </Badge>
      )
    default:
      return null
  }
}
