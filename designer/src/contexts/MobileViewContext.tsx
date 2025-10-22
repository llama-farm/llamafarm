import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from 'react'
import { useIsMobile } from '../hooks/useMediaQuery'

type View = 'chat' | 'project'

type MobileViewContextValue = {
  isMobile: boolean
  mobileView: View
  setMobileView: (v: View) => void
  markUserChoice: (v: View) => void
}

const MobileViewContext = createContext<MobileViewContextValue | undefined>(
  undefined
)

export function MobileViewProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const isMobile = useIsMobile()
  const [mobileView, setMobileView] = useState<View>('project')
  const userChoiceRef = useRef<View | null>(null)

  const markUserChoice = useCallback((v: View) => {
    userChoiceRef.current = v
    setMobileView(v)
  }, [])

  const value = useMemo<MobileViewContextValue>(
    () => ({ isMobile, mobileView, setMobileView, markUserChoice }),
    [isMobile, mobileView, setMobileView, markUserChoice]
  )

  return (
    <MobileViewContext.Provider value={value}>
      {children}
    </MobileViewContext.Provider>
  )
}

export function useMobileView(): MobileViewContextValue {
  const ctx = useContext(MobileViewContext)
  if (!ctx)
    throw new Error('useMobileView must be used within MobileViewProvider')
  return ctx
}
