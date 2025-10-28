import { useState, useEffect } from 'react'
import type { Mode } from '../components/ModeToggle'
import { useModeReset } from '../contexts/ModeContext'

/**
 * Custom hook that manages mode state with support for global reset signals.
 * When the reset is triggered via context, this hook automatically resets to 'designer'.
 */
export function useModeWithReset(
  initialMode: Mode = 'designer'
): [Mode, (mode: Mode) => void] {
  const [mode, setMode] = useState<Mode>(initialMode)
  const { resetCounter } = useModeReset()

  useEffect(() => {
    setMode('designer')
  }, [resetCounter])

  return [mode, setMode]
}
