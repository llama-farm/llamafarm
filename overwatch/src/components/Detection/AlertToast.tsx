import type { Alert } from '../../types'

interface AlertToastProps {
  alert: Alert
  onDismiss: () => void
  onAction?: () => void
  actionLabel?: string
}

export function AlertToast({ alert, onDismiss, onAction, actionLabel }: AlertToastProps) {
  const getAlertStyle = () => {
    switch (alert.type) {
      case 'detection':
        return {
          bg: 'bg-red-500/95',
          border: 'border-red-400',
          icon: '👤'
        }
      case 'battery':
        return {
          bg: 'bg-amber-500/95',
          border: 'border-amber-400',
          icon: '🔋'
        }
      case 'connection':
        return {
          bg: 'bg-gray-600/95',
          border: 'border-gray-500',
          icon: '📡'
        }
    }
  }

  const style = getAlertStyle()

  return (
    <div
      className={`
        animate-slide-in
        ${style.bg} ${style.border}
        border rounded-xl p-3 shadow-2xl
        flex items-start gap-3
        w-full
      `}
    >
      {/* Icon */}
      <span className="text-xl shrink-0">{style.icon}</span>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-white font-medium text-sm leading-tight">
          {alert.message}
        </p>

        {/* Actions */}
        {onAction && actionLabel && (
          <button
            onClick={onAction}
            className="mt-2 bg-white/20 active:bg-white/30 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
            style={{ minHeight: '36px' }}
          >
            {actionLabel}
          </button>
        )}
      </div>

      {/* Dismiss button */}
      <button
        onClick={onDismiss}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full active:bg-white/20 text-white/80"
        style={{ minHeight: '32px', minWidth: '32px' }}
      >
        ✕
      </button>
    </div>
  )
}

interface AlertStackProps {
  alerts: Alert[]
  onDismiss: (alertId: string) => void
  onAlertAction: (alert: Alert) => void
}

export function AlertStack({ alerts, onDismiss, onAlertAction }: AlertStackProps) {
  if (alerts.length === 0) return null

  return (
    <div className="fixed top-12 left-3 right-3 z-[3000] flex flex-col gap-2">
      {alerts.slice(0, 3).map(alert => (
        <AlertToast
          key={alert.id}
          alert={alert}
          onDismiss={() => onDismiss(alert.id)}
          onAction={() => onAlertAction(alert)}
          actionLabel={
            alert.type === 'detection' ? 'View' :
            alert.type === 'battery' ? 'Send Backup' :
            undefined
          }
        />
      ))}

      {/* More alerts indicator */}
      {alerts.length > 3 && (
        <div className="text-center text-gray-400 text-sm py-1">
          +{alerts.length - 3} more alerts
        </div>
      )}
    </div>
  )
}
