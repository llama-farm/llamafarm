import * as React from 'react'
import { cn } from '@/lib/utils'
import FontIcon from '../../common/FontIcon'

type Toast = {
  id: number
  message: string
  variant?: 'default' | 'destructive'
}

type ToastContextValue = { toast: (opts: Omit<Toast, 'id'>) => void }

const ToastContext = React.createContext<ToastContextValue | undefined>(
  undefined
)

export function useToast() {
  const ctx = React.useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([])

  const toast = React.useCallback((opts: Omit<Toast, 'id'>) => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, ...opts }])
    window.setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 3000)
  }, [])

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3">
        {toasts.map(t => (
          <div
            key={t.id}
            role="status"
            className={cn(
              'min-w-[320px] max-w-[420px] rounded-lg border shadow-lg ring-1 ring-black/5 bg-card text-card-foreground px-4 py-3.5 flex items-center gap-3',
              t.variant === 'destructive'
                ? 'border-destructive/50'
                : 'border-teal-600/40'
            )}
          >
            <div
              className={cn(
                'w-7 h-7 rounded-full grid place-items-center',
                t.variant === 'destructive'
                  ? 'bg-destructive text-destructive-foreground'
                  : 'bg-teal-600 text-teal-50 dark:bg-teal-400 dark:text-teal-900'
              )}
            >
              <FontIcon
                type={
                  t.variant === 'destructive' ? 'close' : 'checkmark-filled'
                }
                className="w-4 h-4"
              />
            </div>
            <div className="text-sm md:text-base leading-5">{t.message}</div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
