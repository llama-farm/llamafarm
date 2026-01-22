import { useEffect, useState, useCallback, useRef } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import FontIcon from '../common/FontIcon'
import ModeToggle from './ModeToggle'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Switch } from './ui/switch'
import ConfigEditor from './ConfigEditor/ConfigEditor'
import TestChat from './TestChat/TestChat'
import { usePackageModal } from '../contexts/PackageModalContext'
import { Input } from './ui/input'
import { useModeWithReset } from '../hooks/useModeWithReset'
import { useConfigPointer } from '../hooks/useConfigPointer'
import { useProject } from '../hooks/useProjects'
import { useActiveProject } from '../hooks/useActiveProject'
import { useOnboardingContext } from '../contexts/OnboardingContext'
import type { ProjectConfig } from '../types/config'
import { DevToolsProvider } from '../contexts/DevToolsContext'
import { DevToolsDrawer } from './DevTools'

// Sample test inputs for each sample dataset
const SAMPLE_TEST_INPUTS: Record<string, string> = {
  // Classifier samples
  sentiment: "This product exceeded my expectations!",
  expense: "Uber to downtown office meeting",
  // Anomaly samples
  'fridge-temp': "42.5",
  'biometric': "85, 99.1, 95, 135, 18",
  'build-status': "failed",
  'support-ticket': "critical, security, phone, escalated, 0",
}

const Test = () => {
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const { openPackageModal } = usePackageModal()
  const onboarding = useOnboardingContext()

  // Model type for Test page: 'inference' (default), 'anomaly', 'classifier', 'document_scanning', 'encoder', or 'speech'
  const [modelType, setModelType] = useState<'inference' | 'anomaly' | 'classifier' | 'document_scanning' | 'encoder' | 'speech'>(() => {
    if (typeof window === 'undefined') return 'inference'
    const stored = localStorage.getItem('lf_test_modelType')
    if (stored === 'anomaly' || stored === 'classifier' || stored === 'document_scanning' || stored === 'encoder' || stored === 'speech') return stored
    return 'inference'
  })

  // Apply mode from URL param (takes precedence over localStorage/onboarding defaults)
  // This is used when navigating from checklist with ?mode=classifier or ?mode=anomaly
  useEffect(() => {
    const modeParam = searchParams.get('mode')
    if (modeParam === 'classifier' || modeParam === 'anomaly') {
      setModelType(modeParam)

      // Also set the model override from onboarding if available
      // This ensures the trained model is auto-selected when coming from checklist
      const { trainedModelName, trainedModelType } = onboarding.state.answers
      if (trainedModelName && trainedModelType === modeParam) {
        const overrideKey = modeParam === 'classifier'
          ? 'lf_test_classifierModel_override'
          : 'lf_test_anomalyModel_override'
        localStorage.setItem(overrideKey, trainedModelName)
      }

      // Clean up the URL param without triggering navigation
      const newParams = new URLSearchParams(searchParams)
      newParams.delete('mode')
      setSearchParams(newParams, { replace: true })
    }
  }, [searchParams, setSearchParams, onboarding.state.answers])

  // Track if we've applied onboarding defaults (only do once per session)
  const appliedOnboardingDefaultsRef = useRef(false)

  // Apply onboarding-based defaults on first visit (based on project type)
  useEffect(() => {
    if (appliedOnboardingDefaultsRef.current) return

    const { projectType, dataStatus, selectedSampleDataset, trainedModelName, trainedModelType } = onboarding.state.answers

    // Only apply if user has completed onboarding
    if (!onboarding.state.onboardingCompleted) return

    appliedOnboardingDefaultsRef.current = true

    // Set model type based on project type
    if (projectType === 'classifier' && trainedModelType === 'classifier' && trainedModelName) {
      setModelType('classifier')
      // Set as one-time override that TestChat will consume and clear
      localStorage.setItem('lf_test_classifierModel_override', trainedModelName)
    } else if (projectType === 'anomaly' && trainedModelType === 'anomaly' && trainedModelName) {
      setModelType('anomaly')
      // Set as one-time override that TestChat will consume and clear
      localStorage.setItem('lf_test_anomalyModel_override', trainedModelName)
    } else if (projectType === 'doc-qa' || projectType === 'exploring') {
      // Doc-QA and exploring projects should always use inference (text generation) mode
      setModelType('inference')
    }

    // Set sample input if available (for classifier/anomaly sample data)
    if (dataStatus === 'sample-data' && selectedSampleDataset && SAMPLE_TEST_INPUTS[selectedSampleDataset]) {
      const sampleInput = SAMPLE_TEST_INPUTS[selectedSampleDataset]
      if (projectType === 'classifier') {
        localStorage.setItem('lf_test_classifierInput', sampleInput)
      } else if (projectType === 'anomaly') {
        localStorage.setItem('lf_test_anomalyInput', sampleInput)
      }
    }
  }, [onboarding.state.answers, onboarding.state.onboardingCompleted])

  // Persist modelType to localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_modelType', modelType)
  }, [modelType])

  const [mode, setMode] = useModeWithReset('designer')

  // Get active project and config for unsaved changes checking
  const activeProject = useActiveProject()
  const { data: projectDetail } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )
  const projectConfig = (projectDetail as any)?.project?.config as
    | ProjectConfig
    | undefined

  // Use config pointer to handle mode changes with unsaved changes check
  const getRootLocation = useCallback(() => ({ type: 'root' as const }), [])
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getRootLocation,
  })

  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const settingsRef = useRef<HTMLDivElement>(null)
  const [showReferences, setShowReferences] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const v = localStorage.getItem('lf_test_showReferences')
    return v == null ? true : v === 'true'
  })
  const [showGenSettings, _setShowGenSettings] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const v = localStorage.getItem('lf_test_showGenSettings')
    return v == null ? false : v === 'true'
  })
  const showPrompts = false
  const [allowRanking, setAllowRanking] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const v = localStorage.getItem('lf_test_allowRanking')
    return v == null ? true : v === 'true'
  })
  const [useTestData, _setUseTestData] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const v = localStorage.getItem('lf_test_useTestData')
    return v == null ? false : v === 'true'
  })
  const [gen, setGen] = useState<{
    temperature: number
    topP: number
    maxTokens: number
    presencePenalty: number
    frequencyPenalty: number
    seed?: number | ''
    streaming: boolean
    jsonMode: boolean
    enableThinking: boolean
    thinkingBudget: number
  }>(() => {
    try {
      const raw = localStorage.getItem('lf_gen_defaults')
      if (raw) {
        const parsed = JSON.parse(raw)
        // Add defaults for new fields if not present
        return {
          enableThinking: false,
          thinkingBudget: 1024,
          ...parsed,
        }
      }
    } catch {}
    return {
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 512,
      presencePenalty: 0,
      frequencyPenalty: 0,
      seed: '',
      streaming: true,
      jsonMode: false,
      enableThinking: false,
      thinkingBudget: 1024,
    }
  })

  // RAG UI state for drawer (persisted via localStorage)
  const [ragEnabledUI, setRagEnabledUI] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const v = localStorage.getItem('lf_testchat_rag_enabled')
    return v == null ? true : v === 'true'
  })
  const [ragTopKUI, setRagTopKUI] = useState<number>(() => {
    if (typeof window === 'undefined') return 10
    const v = localStorage.getItem('lf_testchat_rag_top_k')
    return v ? parseInt(v, 10) : 10
  })
  const [ragThresholdUI, setRagThresholdUI] = useState<number>(() => {
    if (typeof window === 'undefined') return 0.7
    const v = Number(localStorage.getItem('lf_testchat_rag_threshold') || '0.7')
    return Number.isFinite(v) ? v : 0.7
  })

  // Persist preferences
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_showReferences', String(showReferences))
  }, [showReferences])
  // Persist showGenSettings (now controlled in the Generation settings drawer)
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_showGenSettings', String(showGenSettings))
  }, [showGenSettings])
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_allowRanking', String(allowRanking))
  }, [allowRanking])
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_useTestData', String(useTestData))
  }, [useTestData])
  useEffect(() => {
    try {
      localStorage.setItem('lf_gen_defaults', JSON.stringify(gen))
    } catch {}
  }, [gen])

  // Close generation settings when clicking outside
  useEffect(() => {
    if (!isSettingsOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      if (
        settingsRef.current &&
        !settingsRef.current.contains(event.target as Node)
      ) {
        setIsSettingsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isSettingsOpen])

  // Persist RAG settings
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_testchat_rag_enabled', String(ragEnabledUI))
  }, [ragEnabledUI])
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_testchat_rag_top_k', String(ragTopKUI))
  }, [ragTopKUI])
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_testchat_rag_threshold', String(ragThresholdUI))
  }, [ragThresholdUI])

  // Hide settings UI when switching to config view or away from inference mode
  useEffect(() => {
    if (mode !== 'designer' || modelType !== 'inference') {
      setIsSettingsOpen(false)
    }
  }, [mode, modelType])

  return (
    <DevToolsProvider>
      <div className="w-full h-full flex flex-col pb-10 relative">
        <div className="mb-3 flex-shrink-0">
        {/* First row: Title + switcher + Package button */}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl">
            {mode === 'designer' ? 'Test' : 'Config editor'}
          </h2>
          <div className="flex items-center gap-3">
            <ModeToggle mode={mode} onToggle={handleModeChange} />
            <Button
              variant="outline"
              size="sm"
              onClick={openPackageModal}
              disabled
            >
              Package
            </Button>
          </div>
        </div>
        {/* Subtitle on its own row */}
        {mode === 'designer' && (
          <div className="text-sm text-muted-foreground mt-2">
            Chat with your model to test and evaluate responses
          </div>
        )}
      </div>

      {/* Settings bar (designer mode only) */}
      {mode === 'designer' && (
        <div className="mb-4 flex flex-col xl:flex-row xl:flex-wrap items-stretch xl:items-start gap-3">
          <div className="flex-1 xl:min-w-[480px] rounded-xl bg-muted/30 border border-border px-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0 min-h-11 py-2 sm:py-0">
            {/* Toggles group - wrap on smaller screens */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-xs w-full">
              <label
                className={`inline-flex items-center gap-2 flex-shrink-0 ${modelType !== 'inference' ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={modelType !== 'inference' ? 'This setting applies to inference mode only' : undefined}
              >
                <Checkbox
                  id="show-processed"
                  checked={showReferences}
                  disabled={modelType !== 'inference'}
                  onCheckedChange={(v: boolean | 'indeterminate') =>
                    setShowReferences(Boolean(v))
                  }
                />
                <span className="whitespace-nowrap">
                  Show referenced chunks
                </span>
              </label>
              <label
                className={`inline-flex items-center gap-2 flex-shrink-0 ${modelType !== 'inference' ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={modelType !== 'inference' ? 'This setting applies to inference mode only' : undefined}
              >
                <Checkbox
                  id="enable-thinking"
                  checked={gen.enableThinking}
                  disabled={modelType !== 'inference'}
                  onCheckedChange={(v: boolean | 'indeterminate') =>
                    setGen({ ...gen, enableThinking: Boolean(v) })
                  }
                />
                <span className="whitespace-nowrap">Enable Thinking</span>
              </label>
              {/* Show generation settings toggle moved into the drawer */}
              <div
                className={`flex items-center gap-2 sm:ml-auto flex-shrink-0 ${modelType !== 'inference' ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={modelType !== 'inference' ? 'This setting applies to inference mode only' : undefined}
              >
                <span className="text-muted-foreground whitespace-nowrap">
                  Allow ranking
                </span>
                <Switch
                  checked={allowRanking}
                  disabled={modelType !== 'inference'}
                  onCheckedChange={(v: boolean) => setAllowRanking(Boolean(v))}
                  aria-label="Allow ranking"
                />
                <span className="text-muted-foreground whitespace-nowrap">
                  {allowRanking ? 'On' : 'Off'}
                </span>
              </div>
            </div>
          </div>
          {/* Generation settings - enabled for inference mode, disabled for anomaly and classifier */}
          <div className="w-full xl:basis-[320px] xl:flex-none">
            <div className="flex-1 relative" ref={settingsRef}>
              {modelType !== 'inference' ? (
                /* Disabled state for non-inference modes */
                <div
                  className="rounded-xl h-11 w-full flex items-center justify-between pl-4 pr-3 border border-border bg-muted/30 opacity-50 cursor-not-allowed"
                  title="Generation settings apply to inference mode only"
                >
                  <span className="text-base text-muted-foreground">Generation settings</span>
                  <span className="text-xs text-muted-foreground">Inference only</span>
                </div>
              ) : isSettingsOpen ? (
                <button
                  type="button"
                  onClick={() => setIsSettingsOpen(false)}
                  aria-label="Collapse generation settings panel"
                  className="rounded-t-xl bg-card border border-border border-b-0 h-11 px-4 w-full flex items-center justify-between text-left cursor-pointer hover:bg-accent/40 transition-colors"
                >
                  <span className="text-base">Generation settings</span>
                  <FontIcon type="close" className="w-4 h-4" />
                </button>
              ) : (
                <Button
                  variant="outline"
                  className="rounded-xl h-11 w-full text-base justify-between pl-4 pr-3"
                  onClick={() => setIsSettingsOpen(true)}
                  aria-label="Expand generation settings panel"
                >
                  <span>Generation settings</span>
                  <FontIcon type="chevron-down" className="w-4 h-4 ml-2" />
                </Button>
              )}
              {isSettingsOpen && modelType === 'inference' && (
                <div className="absolute left-0 right-0 top-full w-full rounded-b-xl bg-card border border-border border-t-0 p-4 shadow-xl z-50">
                  <div className="w-full">
                    {/* RAG master toggle */}
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">RAG</span>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={ragEnabledUI}
                          onCheckedChange={(v: boolean) => {
                            setRagEnabledUI(Boolean(v))
                          }}
                          aria-label="RAG enabled"
                        />
                        <span className="text-muted-foreground">
                          {ragEnabledUI ? 'On' : 'Off'}
                        </span>
                      </div>
                    </div>
                    {ragEnabledUI && (
                      <>
                        {/* RAG retrieval controls */}
                        <div className="grid grid-cols-3 gap-2 items-center mt-3">
                          <span className="text-sm text-muted-foreground">
                            Top‑K
                          </span>
                          <Input
                            type="number"
                            min={1}
                            max={100}
                            step={1}
                            value={ragTopKUI}
                            onChange={e => {
                              const v = Math.max(
                                1,
                                Math.min(100, Number(e.target.value))
                              )
                              setRagTopKUI(v)
                            }}
                            className="col-span-2"
                          />
                        </div>
                        <div className="grid grid-cols-3 gap-2 items-center mt-3">
                          <span className="text-sm text-muted-foreground">
                            Threshold
                          </span>
                          <Input
                            type="number"
                            min={0}
                            max={1}
                            step={0.05}
                            value={ragThresholdUI}
                            onChange={e => {
                              const val = Number(e.target.value)
                              const v = isNaN(val)
                                ? 0
                                : Math.max(0, Math.min(1, val))
                              setRagThresholdUI(v)
                            }}
                            className="col-span-2"
                          />
                        </div>
                      </>
                    )}
                    <div className="h-px w-full bg-border mt-3" />
                    <div className="grid grid-cols-3 gap-2 items-center mt-3">
                      <span className="text-sm text-muted-foreground">
                        Max tokens
                      </span>
                      <Input
                        type="number"
                        step="64"
                        min="1"
                        max="32768"
                        value={gen.maxTokens}
                        onChange={e =>
                          setGen({
                            ...gen,
                            maxTokens: Number(e.target.value),
                          })
                        }
                        className="col-span-2"
                        title="Maximum tokens in the response"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center mt-3">
                      <span className="text-sm text-muted-foreground">
                        Thinking budget
                      </span>
                      <Input
                        type="number"
                        step="128"
                        min="0"
                        value={gen.thinkingBudget}
                        onChange={e =>
                          setGen({
                            ...gen,
                            thinkingBudget: Number(e.target.value),
                            enableThinking: true,
                          })
                        }
                        className="col-span-2"
                        title="Max tokens for thinking process"
                      />
                    </div>
                    {/* Helper text for inference mode */}
                    <div className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border">
                      These settings apply to inference model types only
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 flex flex-col relative">
        {/* Main work area */}
        <div className="flex-1 min-h-0 pb-2 pr-0">
          {mode !== 'designer' ? (
            <div className="h-full overflow-hidden">
              <ConfigEditor className="h-full" initialPointer={configPointer} />
            </div>
          ) : (
            <div className="h-full">
              <TestChat
                {...({
                  modelType,
                  onModelTypeChange: setModelType,
                  showReferences,
                  allowRanking,
                  useTestData,
                  showPrompts,
                  showThinking: gen.enableThinking, // Show thinking in UI when enabled
                  showGenSettings,
                  genSettings: gen,
                  ragEnabled: ragEnabledUI,
                  ragTopK: ragTopKUI,
                  ragScoreThreshold: ragThresholdUI,
                  focusInput: (location.state as any)?.focusInput,
                } as any)}
              />
            </div>
          )}
        </div>
      </div>

      {/* Dev Tools Drawer - fixed to bottom, only in designer mode */}
      {mode === 'designer' && <DevToolsDrawer />}
      </div>
    </DevToolsProvider>
  )
}

export default Test
