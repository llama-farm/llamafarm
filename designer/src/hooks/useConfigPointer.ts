import { useCallback, useEffect, useState } from 'react'
import { findConfigPointer, type ConfigLocation } from '../utils/configNavigation'
import type { ProjectConfig } from '../types/config'
import type { Mode } from '../components/ModeToggle'

interface UseConfigPointerParams {
  mode: Mode
  setMode: (nextMode: Mode) => void
  config: ProjectConfig | undefined
  getLocation: () => ConfigLocation | null
}

interface UseConfigPointerResult {
  configPointer: string | null
  handleModeChange: (nextMode: Mode) => void
  refreshPointer: () => void
}

export function useConfigPointer({
  mode,
  setMode,
  config,
  getLocation,
}: UseConfigPointerParams): UseConfigPointerResult {
  const [configPointer, setConfigPointer] = useState<string | null>(
    mode === 'code' ? '/' : null
  )

  const computePointer = useCallback(
    (targetMode: Mode) => {
      if (targetMode !== 'code') {
        setConfigPointer(null)
        return
      }

      const location = getLocation()
      if (!location) {
        setConfigPointer('/')
        return
      }

      const pointer = findConfigPointer(config, location)
      setConfigPointer(pointer)
    },
    [config, getLocation]
  )

  const handleModeChange = useCallback(
    (nextMode: Mode) => {
      computePointer(nextMode)
      setMode(nextMode)
    },
    [computePointer, setMode]
  )

  useEffect(() => {
    if (mode === 'code') {
      computePointer('code')
    }
  }, [mode, computePointer])

  const refreshPointer = useCallback(() => {
    computePointer(mode)
  }, [computePointer, mode])

  return { configPointer, handleModeChange, refreshPointer }
}
