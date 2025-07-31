import { createContext, useContext, useState, ReactNode } from 'react'

interface HeaderContextType {
  buildType: string
  setBuildType: (type: string) => void
}

const HeaderContext = createContext<HeaderContextType | undefined>(undefined)

export const HeaderProvider = ({ children }: { children: ReactNode }) => {
  const [buildType, setBuildType] = useState('dashboard')

  return (
    <HeaderContext.Provider value={{ buildType, setBuildType }}>
      {children}
    </HeaderContext.Provider>
  )
}

const useHeader = () => {
  const context = useContext(HeaderContext)
  if (context === undefined) {
    throw new Error('useHeader must be used within a HeaderProvider')
  }
  return context
}

export default useHeader
