import { useEffect, useState } from 'react'

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    try {
      const mql = window.matchMedia(query)
      const onChange = (e: MediaQueryListEvent | MediaQueryList): void => {
        const next = 'matches' in e ? e.matches : (e as MediaQueryList).matches
        setMatches(next)
      }
      onChange(mql)
      // Fallback for older browsers where MediaQueryList uses addListener/removeListener
      if (typeof (mql as any).addEventListener === 'function') {
        mql.addEventListener('change', onChange)
        return () => mql.removeEventListener('change', onChange)
      } else if (typeof (mql as any).addListener === 'function') {
        ;(mql as any).addListener(onChange)
        return () => (mql as any).removeListener(onChange)
      }
      return
    } catch {
      setMatches(false)
      return
    }
  }, [query])

  return matches
}

export function useIsMobile(): boolean {
  return useMediaQuery('(max-width: 767px)')
}

export default useMediaQuery
