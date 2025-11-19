import { createContext, useContext, useState, ReactNode } from 'react'

interface DemoModalContextType {
  isOpen: boolean
  autoStartDemoId: string | null
  openModal: (autoStartDemoId?: string) => void
  closeModal: () => void
}

const DemoModalContext = createContext<DemoModalContextType | undefined>(undefined)

export function DemoModalProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [autoStartDemoId, setAutoStartDemoId] = useState<string | null>(null)

  const openModal = (demoId?: string) => {
    setAutoStartDemoId(demoId || null)
    setIsOpen(true)
  }
  
  const closeModal = () => {
    setIsOpen(false)
    setAutoStartDemoId(null)
  }

  return (
    <DemoModalContext.Provider value={{ isOpen, autoStartDemoId, openModal, closeModal }}>
      {children}
    </DemoModalContext.Provider>
  )
}

export function useDemoModal() {
  const context = useContext(DemoModalContext)
  if (!context) {
    throw new Error('useDemoModal must be used within DemoModalProvider')
  }
  return context
}
