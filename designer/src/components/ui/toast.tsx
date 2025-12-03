import * as React from 'react'
import { cn } from '@/lib/utils'
import FontIcon from '../../common/FontIcon'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './collapsible'

type Toast = {
  id: number
  message: string
  variant?: 'default' | 'destructive'
  icon?: 'checkmark-filled' | 'alert-triangle' | 'close'
  timeoutId?: number
}

type ToastGroup = {
  key: string
  toasts: Toast[]
  variant?: 'default' | 'destructive'
  icon?: 'checkmark-filled' | 'alert-triangle' | 'close'
  timeoutId?: number
}

type ToastContextValue = {
  toast: (opts: Omit<Toast, 'id' | 'timeoutId'>) => void
  dismiss: (id: number) => void
}

const ToastContext = React.createContext<ToastContextValue | undefined>(
  undefined
)

export function useToast() {
  const ctx = React.useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}

// Generate a grouping key based on variant and icon
function getGroupKey(toast: Toast): string {
  return `${toast.variant || 'default'}-${toast.icon || 'none'}`
}

// Group toasts by variant+icon, returning groups (3+) and individuals (1-2)
function groupToasts(toasts: Toast[]): {
  groups: ToastGroup[]
  individuals: Toast[]
} {
  const groups: ToastGroup[] = []
  const individuals: Toast[] = []

  // Group toasts by their grouping key
  const grouped = new Map<string, Toast[]>()
  for (const toast of toasts) {
    const key = getGroupKey(toast)
    if (!grouped.has(key)) {
      grouped.set(key, [])
    }
    grouped.get(key)!.push(toast)
  }

  // Separate into groups (3+) and individuals (1-2)
  for (const [key, toastList] of grouped.entries()) {
    if (toastList.length >= 3) {
      groups.push({
        key,
        toasts: toastList,
        variant: toastList[0].variant,
        icon: toastList[0].icon,
      })
    } else {
      individuals.push(...toastList)
    }
  }

  return { groups, individuals }
}

// Get summary message for a group
function getGroupMessage(group: ToastGroup): string {
  const count = group.toasts.length
  if (group.variant === 'destructive') {
    return `${count} error${count > 1 ? 's' : ''} occurred`
  }
  if (group.icon === 'alert-triangle') {
    return `${count} warning${count > 1 ? 's' : ''}`
  }
  return `${count} successful action${count > 1 ? 's' : ''}`
}

// Individual toast item component
function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast
  onDismiss: (id: number) => void
}) {
  return (
    <div
      role="status"
      className={cn(
        'min-w-[320px] max-w-[420px] rounded-lg border shadow-lg ring-1 ring-black/5 bg-card text-card-foreground px-4 py-3.5 flex items-center gap-3',
        toast.variant === 'destructive'
          ? 'border-destructive/50'
          : 'border-teal-600/40'
      )}
    >
      <div
        className={cn(
          'w-7 h-7 flex-shrink-0 rounded-full grid place-items-center',
          toast.variant === 'destructive'
            ? 'bg-destructive text-destructive-foreground'
            : toast.icon === 'alert-triangle'
              ? 'bg-muted text-muted-foreground'
              : 'bg-teal-600 text-teal-50 dark:bg-teal-400 dark:text-teal-900'
        )}
      >
        <FontIcon
          type={
            toast.icon ||
            (toast.variant === 'destructive' ? 'close' : 'checkmark-filled')
          }
          className="w-4 h-4"
        />
      </div>
      <div className="text-sm md:text-base leading-5 flex-1 break-words">
        {toast.message}
      </div>
      <button
        onClick={() => onDismiss(toast.id)}
        className="flex-shrink-0 w-5 h-5 rounded-sm hover:opacity-80 hover:bg-muted/50 flex items-center justify-center transition-colors"
        aria-label="Dismiss"
      >
        <FontIcon type="close" className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

// Grouped toast component with expand/collapse
function ToastGroup({
  group,
  onDismiss,
}: {
  group: ToastGroup
  onDismiss: (ids: number[]) => void
}) {
  const [isOpen, setIsOpen] = React.useState(false)

  const handleDismiss = () => {
    onDismiss(group.toasts.map(t => t.id))
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div
        role="status"
        className={cn(
          'min-w-[320px] max-w-[420px] rounded-lg border shadow-lg ring-1 ring-black/5 bg-card text-card-foreground',
          group.variant === 'destructive'
            ? 'border-destructive/50'
            : 'border-teal-600/40'
        )}
      >
        <div className="px-4 py-3.5 flex items-center gap-3">
          <CollapsibleTrigger asChild>
            <button
              className="flex-1 flex items-center gap-3 hover:bg-muted/30 transition-colors rounded-lg -ml-3 -mr-3 pl-3 pr-3"
            >
              <div
                className={cn(
                  'w-7 h-7 flex-shrink-0 rounded-full grid place-items-center',
                  group.variant === 'destructive'
                    ? 'bg-destructive text-destructive-foreground'
                    : group.icon === 'alert-triangle'
                      ? 'bg-muted text-muted-foreground'
                      : 'bg-teal-600 text-teal-50 dark:bg-teal-400 dark:text-teal-900'
                )}
              >
                <FontIcon
                  type={
                    group.icon ||
                    (group.variant === 'destructive' ? 'close' : 'checkmark-filled')
                  }
                  className="w-4 h-4"
                />
              </div>
              <div className="text-sm md:text-base leading-5 flex-1 break-words text-left">
                {getGroupMessage(group)}
              </div>
              <FontIcon
                type={isOpen ? 'chevron-up' : 'chevron-down'}
                className="w-4 h-4 flex-shrink-0"
              />
            </button>
          </CollapsibleTrigger>
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 w-5 h-5 rounded-sm hover:opacity-80 hover:bg-muted/50 flex items-center justify-center transition-colors"
            aria-label="Dismiss"
          >
            <FontIcon type="close" className="w-3.5 h-3.5" />
          </button>
        </div>
        <CollapsibleContent>
          <div className="px-4 pb-3.5 pt-2 space-y-2 border-t border-border/50">
            {group.toasts.map(toast => (
              <div
                key={toast.id}
                className="text-sm md:text-base leading-5 text-muted-foreground pl-11 break-words"
              >
                {toast.message}
              </div>
            ))}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([])
  const timeoutRefs = React.useRef<Map<number, number>>(new Map())

  const dismiss = React.useCallback((id: number) => {
    setToasts(prev => {
      const newToasts = prev.filter(t => t.id !== id)
      // Clear timeout if it exists
      const timeoutId = timeoutRefs.current.get(id)
      if (timeoutId) {
        window.clearTimeout(timeoutId)
        timeoutRefs.current.delete(id)
      }
      return newToasts
    })
  }, [])

  const dismissGroup = React.useCallback((ids: number[]) => {
    setToasts(prev => {
      const newToasts = prev.filter(t => !ids.includes(t.id))
      // Clear all timeouts for dismissed toasts
      ids.forEach(id => {
        const timeoutId = timeoutRefs.current.get(id)
        if (timeoutId) {
          window.clearTimeout(timeoutId)
          timeoutRefs.current.delete(id)
        }
      })
      return newToasts
    })
  }, [])

  const toast = React.useCallback((opts: Omit<Toast, 'id' | 'timeoutId'>) => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, ...opts }])
  }, [])

  // Set up auto-dismiss timeouts for new toasts
  React.useEffect(() => {
    toasts.forEach(toast => {
      // Skip if timeout already exists
      if (timeoutRefs.current.has(toast.id)) {
        return
      }

      const timeoutId = window.setTimeout(() => {
        dismiss(toast.id)
      }, 8000)

      timeoutRefs.current.set(toast.id, timeoutId)
    })
  }, [toasts, dismiss])

  // Cleanup all timeouts on unmount
  React.useEffect(() => {
    return () => {
      timeoutRefs.current.forEach(timeoutId => {
        window.clearTimeout(timeoutId)
      })
      timeoutRefs.current.clear()
    }
  }, [])

  const { groups, individuals } = groupToasts(toasts)

  return (
    <ToastContext.Provider value={{ toast, dismiss }}>
      {children}
      <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3">
        {groups.map(group => (
          <ToastGroup
            key={group.key}
            group={group}
            onDismiss={dismissGroup}
          />
        ))}
        {individuals.map(t => (
          <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}
