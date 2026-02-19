import { useCallback } from 'react'
import ModeToggle from '../ModeToggle'
import { Button } from '@/components/ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useBundleModal } from '../../contexts/BundleModalContext'
import Prompts from './GeneratedOutput/Prompts'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject } from '../../hooks/useProjects'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'

const Prompt = () => {
  const [mode, setMode] = useModeWithReset('designer')
  const { openBundleModal } = useBundleModal()
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )

  const projectConfig = (projectResp as any)?.project?.config as ProjectConfig | undefined
  const getPromptsLocation = useCallback(
    () => ({ type: 'prompts' as const }),
    []
  )
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getPromptsLocation,
  })

  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Prompts' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={handleModeChange} />
          <Button
            variant="outline"
            size="sm"
            onClick={openBundleModal}
          >
            Bundle
          </Button>
        </div>
      </div>
      {mode === 'designer' ? (
        <div className="flex-1 min-h-0 pb-6 overflow-auto">
          <Prompts />
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-hidden pb-6">
          <ConfigEditor className="h-full" initialPointer={configPointer} />
        </div>
      )}
    </div>
  )
}

export default Prompt
