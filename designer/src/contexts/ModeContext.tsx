import {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
} from 'react'

interface ModeResetContextType {
  triggerReset: () => void
  resetCounter: number
}

const ModeResetContext = createContext<ModeResetContextType | null>(null)

export function ModeResetProvider({ children }: { children: ReactNode }) {
  const [resetCounter, setResetCounter] = useState(0)

  const triggerReset = useCallback(() => {
    setResetCounter(prev => prev + 1)
  }, [])

  return (
    <ModeResetContext.Provider value={{ triggerReset, resetCounter }}>
      {children}
    </ModeResetContext.Provider>
  )
}

export function useModeReset() {
  const context = useContext(ModeResetContext)
  if (!context) {
    throw new Error('useModeReset must be used within a ModeResetProvider')
  }
  return context
}
