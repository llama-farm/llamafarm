import { useEffect, useMemo, useState } from 'react'
import FontIcon from '../common/FontIcon'
import ModeToggle, { Mode } from './ModeToggle'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Switch } from './ui/switch'
import ConfigEditor from './ConfigEditor/ConfigEditor'
import TestChat from './TestChat/TestChat'
import { usePackageModal } from '../contexts/PackageModalContext'
import { Input } from './ui/input'

interface TestCase {
  id: number
  name: string
  source: string
  score: number
  environment: 'Local' | 'Production' | 'Staging'
  lastRun: string
  input?: string
  expected?: string
}

const scorePillClasses = (score: number) => {
  if (score >= 95) return 'bg-teal-300 text-black'
  if (score >= 75) return 'bg-primary text-primary-foreground'
  return 'bg-amber-300 text-black'
}

const Test = () => {
  const { openPackageModal } = usePackageModal()
  const [running, setRunning] = useState<Record<number, boolean>>({})
  const [tests, setTests] = useState<TestCase[]>([
    {
      id: 1,
      name: 'Aircraft maintenance queries',
      source: 'From prompts',
      score: 99.5,
      environment: 'Local',
      lastRun: '2hr ago',
      input:
        'The hydraulic pump on the F-16 showed a pressure drop during taxi. What are the most likely causes and the next steps for inspection?',
      expected:
        'A pressure drop in the hydraulic pump during taxi on an F-16 could be caused by fluid leakage, air in the system, or a failing pressure sensor. Recommended next steps include inspecting hydraulic lines for leaks, checking fluid levels, and running a diagnostic on the pressure sensor.',
    },
    {
      id: 2,
      name: 'Basic user queries',
      source: 'From prompts',
      score: 82.5,
      environment: 'Local',
      lastRun: '1d ago',
    },
    {
      id: 3,
      name: 'Aircraft maintenance queries',
      source: 'From prompts',
      score: 76.5,
      environment: 'Production',
      lastRun: '8/1/25',
    },
    {
      id: 4,
      name: 'API integration',
      source: 'Custom',
      score: 99.5,
      environment: 'Production',
      lastRun: '7/30/25',
    },
    {
      id: 5,
      name: 'Aircraft maintenance queries',
      source: 'From chat history',
      score: 54,
      environment: 'Local',
      lastRun: '7/30/25',
    },
    {
      id: 6,
      name: 'Security validation',
      source: 'Custom',
      score: 99.5,
      environment: 'Staging',
      lastRun: '7/30/25',
    },
  ])

  // New tests are created via a button and edited in the existing modal

  const [isEditOpen, setIsEditOpen] = useState<boolean>(false)
  const [editId, setEditId] = useState<number | null>(null)
  const [editForm, setEditForm] = useState<{
    name: string
    input: string
    expected: string
  }>({
    name: '',
    input: '',
    expected: '',
  })

  // Create modal state
  const [isCreateOpen, setIsCreateOpen] = useState<boolean>(false)
  const [createForm, setCreateForm] = useState<{
    name: string
    input: string
    expected: string
  }>({ name: '', input: '', expected: '' })

  // Environment controls removed from the panel UI

  // No inline form; validation handled in edit modal

  const nextId = useMemo(
    () => tests.reduce((max, t) => (t.id > max ? t.id : max), 0) + 1,
    [tests]
  )

  const handleRun = (id: number, override?: TestCase) => {
    const row = override ?? tests.find(t => t.id === id)
    // Fire event for the chat area to consume
    if (row && typeof window !== 'undefined') {
      try {
        window.dispatchEvent(
          new CustomEvent('lf-test-run', {
            detail: {
              id: row.id,
              name: row.name,
              input: row.input ?? '',
              expected: row.expected ?? '',
            },
          })
        )
      } catch {}
    }

    // Briefly show Running… on the clicked button
    setRunning(prev => ({ ...prev, [id]: true }))
    setTimeout(() => {
      setRunning(prev => ({ ...prev, [id]: false }))
    }, 800)

    // Collapse panels appropriately
    setIsPanelOpen(false)

    // Update list metadata immediately
    setTests(prev =>
      prev.map(t =>
        t.id === id
          ? {
              ...t,
              lastRun: 'just now',
              // Tiny nudge to the score to simulate a run
              score: Math.max(
                0,
                Math.min(
                  100,
                  Number((t.score + (Math.random() - 0.5) * 2).toFixed(1))
                )
              ),
            }
          : t
      )
    )
  }

  // Environment no longer selectable in the panel

  const openCreateModal = () => {
    setCreateForm({ name: '', input: '', expected: '' })
    setIsCreateOpen(true)
  }

  const saveCreate = (runAfterSave: boolean) => {
    const newRow: TestCase = {
      id: nextId,
      name: createForm.name.trim() || 'New test',
      source: 'Custom',
      score: 0,
      environment: 'Local',
      lastRun: '-',
      input: createForm.input.trim(),
      expected: createForm.expected.trim(),
    }
    setTests(prev => [newRow, ...prev])
    setIsCreateOpen(false)
    if (runAfterSave) {
      // Defer so state updates apply before run feedback
      setTimeout(() => handleRun(newRow.id, newRow), 0)
    }
  }

  const deleteTest = (id: number) => {
    setTests(prev => prev.filter(t => t.id !== id))
  }

  const openEdit = (id: number) => {
    const row = tests.find(t => t.id === id)
    if (!row) return
    setEditId(id)
    setEditForm({
      name: row.name,
      input: row.input ?? '',
      expected: row.expected ?? '',
    })
    setIsEditOpen(true)
  }

  const saveEdit = (runAfterSave: boolean) => {
    if (editId == null) return
    setTests(prev =>
      prev.map(t =>
        t.id === editId
          ? {
              ...t,
              name: editForm.name.trim(),
              input: editForm.input.trim(),
              expected: editForm.expected.trim(),
            }
          : t
      )
    )
    if (runAfterSave) {
      const edited = tests.find(t => t.id === editId)
      const merged: TestCase | undefined = edited
        ? {
            ...edited,
            name: editForm.name.trim(),
            input: editForm.input.trim(),
            expected: editForm.expected.trim(),
          }
        : undefined
      setTimeout(() => handleRun(editId, merged), 0)
    }
    setIsEditOpen(false)
  }

  const [mode, setMode] = useState<Mode>('designer')
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [showReferences, setShowReferences] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const v = localStorage.getItem('lf_test_showReferences')
    return v == null ? true : v === 'true'
  })
  const [showGenSettings, setShowGenSettings] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const v = localStorage.getItem('lf_test_showGenSettings')
    return v == null ? false : v === 'true'
  })
  const [showPrompts, setShowPrompts] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const v = localStorage.getItem('lf_test_showPrompts')
    return v == null ? false : v === 'true'
  })
  const [showThinking, setShowThinking] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const v = localStorage.getItem('lf_test_showThinking')
    return v == null ? false : v === 'true'
  })
  const [allowRanking, setAllowRanking] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const v = localStorage.getItem('lf_test_allowRanking')
    return v == null ? true : v === 'true'
  })
  const [useTestData, setUseTestData] = useState<boolean>(() => {
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
  }>(() => {
    try {
      const raw = localStorage.getItem('lf_gen_defaults')
      if (raw) return JSON.parse(raw)
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
    }
  })

  // Local diagnose loading for origin CTAs
  const [diagnosing, setDiagnosing] = useState<Record<string, boolean>>({})

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
    localStorage.setItem('lf_test_showPrompts', String(showPrompts))
  }, [showPrompts])
  useEffect(() => {
    if (typeof window === 'undefined') return
    localStorage.setItem('lf_test_showThinking', String(showThinking))
  }, [showThinking])
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

  // Hide settings UI and close any open panels when switching to config view
  useEffect(() => {
    if (mode !== 'designer') {
      setIsPanelOpen(false)
      setIsSettingsOpen(false)
    }
  }, [mode])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <div>
          <h2 className="text-2xl ">
            {mode === 'designer' ? 'Test' : 'Config editor'}
          </h2>
          {mode === 'designer' && (
            <div className="text-sm text-muted-foreground mt-1">
              Chat with your model to test and evaluate responses
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <Button variant="outline" size="sm" onClick={openPackageModal}>
            Package
          </Button>
        </div>
      </div>

      {/* Settings bar (designer mode only) */}
      {mode === 'designer' && (
        <div className="mb-4 flex flex-col lg:flex-row lg:flex-wrap items-stretch lg:items-start gap-3">
          <div className="flex-1 lg:min-w-[560px] rounded-xl bg-muted/30 border border-border px-4 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-2 lg:gap-0 h-auto lg:h-11 py-2 lg:py-0">
            {/* Toggles group with wrapping; includes Allow ranking so it wraps on small screens */}
            <div className="flex flex-wrap items-center gap-2 md:gap-3 gap-y-2 text-xs min-w-0 w-full">
              <label className="inline-flex items-center gap-2">
                <Checkbox
                  id="show-processed"
                  checked={showReferences}
                  onCheckedChange={(v: boolean | 'indeterminate') =>
                    setShowReferences(Boolean(v))
                  }
                />
                <span className="whitespace-nowrap">
                  Show referenced chunks
                </span>
              </label>
              <label className="inline-flex items-center gap-2">
                <Checkbox
                  id="show-prompts"
                  checked={showPrompts}
                  onCheckedChange={(v: boolean | 'indeterminate') =>
                    setShowPrompts(Boolean(v))
                  }
                />
                <span className="whitespace-nowrap">Show prompts sent</span>
              </label>
              <label className="inline-flex items-center gap-2">
                <Checkbox
                  id="show-thinking"
                  checked={showThinking}
                  onCheckedChange={(v: boolean | 'indeterminate') =>
                    setShowThinking(Boolean(v))
                  }
                />
                <span className="whitespace-nowrap">Show thinking steps</span>
              </label>
              {/* Show generation settings toggle moved into the drawer */}
              <div className="flex items-center gap-2 lg:ml-auto">
                <span className="text-muted-foreground whitespace-nowrap">
                  Allow ranking
                </span>
                <Switch
                  checked={allowRanking}
                  onCheckedChange={(v: boolean) => setAllowRanking(Boolean(v))}
                  aria-label="Allow ranking"
                />
                <span className="text-muted-foreground whitespace-nowrap">
                  {allowRanking ? 'On' : 'Off'}
                </span>
              </div>
            </div>
          </div>
          <div className="w-full lg:basis-[640px] lg:flex-none flex flex-col lg:flex-row gap-2">
            <div className="flex-1 relative">
              {isPanelOpen ? (
                <button
                  type="button"
                  onClick={() => setIsPanelOpen(false)}
                  aria-label="Collapse tests panel"
                  className="rounded-t-xl bg-card border border-border border-b-0 h-11 px-4 w-full flex items-center justify-between text-left cursor-pointer hover:bg-accent/40 transition-colors"
                >
                  <span className="text-base">Tests</span>
                  <FontIcon type="close" className="w-4 h-4" />
                </button>
              ) : (
                <Button
                  variant="outline"
                  className="rounded-xl h-11 w-full text-base justify-between pl-4 pr-3"
                  onClick={() => {
                    setIsPanelOpen(true)
                    setIsSettingsOpen(false)
                  }}
                  aria-label="Expand tests panel"
                >
                  <span>Tests</span>
                  <FontIcon type="chevron-down" className="w-4 h-4 ml-2" />
                </Button>
              )}
              {isPanelOpen && (
                <div className="absolute left-0 right-0 top-full w-full rounded-b-xl bg-card border border-border border-t-0 p-4 shadow-xl z-50">
                  <div className="w-full">
                    <Button variant="outline" onClick={openCreateModal}>
                      <span className="mr-2">New test</span>
                      <FontIcon type="add" className="w-4 h-4" />
                    </Button>
                    <div className="mt-4 h-px w-full bg-border" />
                  </div>
                  <div className="w-full rounded-md overflow-hidden border border-border mt-4">
                    <div className="max-h-[60vh] overflow-auto">
                      <div className="flex flex-col divide-y divide-border">
                        {tests.map(test => (
                          <div
                            key={test.id}
                            className="px-4 py-4 bg-card/60 hover:bg-accent/40 transition-colors"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                  <div className="text-sm truncate">
                                    {test.name}
                                  </div>
                                  <FontIcon
                                    type="edit"
                                    isButton
                                    handleOnClick={() => openEdit(test.id)}
                                    className="w-4 h-4 text-primary"
                                  />
                                </div>
                                <div className="text-xs text-muted-foreground mt-1">
                                  Last run {test.lastRun}
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleRun(test.id)}
                                  disabled={Boolean(running[test.id])}
                                >
                                  {running[test.id] ? 'Running…' : 'Run'}
                                </Button>
                              </div>
                            </div>
                            <div className="mt-3 flex items-center">
                              <span
                                className={`px-2 py-0.5 rounded-2xl text-xs ${scorePillClasses(test.score)}`}
                              >
                                {test.score}%
                              </span>
                              {test.score < 80 && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="ml-3 h-7 px-2 py-0 text-teal-700 border-teal-500/50 hover:bg-teal-500/10 dark:text-teal-300"
                                  onClick={() => {
                                    setDiagnosing(prev => ({
                                      ...prev,
                                      [String(test.id)]: true,
                                    }))
                                    try {
                                      window.dispatchEvent(
                                        new CustomEvent('lf-diagnose', {
                                          detail: {
                                            source: 'low_score',
                                            testId: test.id,
                                            testName: test.name,
                                            input: test.input ?? '',
                                            expected: test.expected ?? '',
                                            matchScore: test.score,
                                          },
                                        })
                                      )
                                    } catch {}
                                    setIsPanelOpen(false)
                                    setTimeout(
                                      () =>
                                        setDiagnosing(prev => ({
                                          ...prev,
                                          [String(test.id)]: false,
                                        })),
                                      1000
                                    )
                                  }}
                                >
                                  {diagnosing[String(test.id)] ? (
                                    <span className="inline-flex items-center gap-2">
                                      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-teal-500 border-t-transparent" />
                                      <span>Diagnosing…</span>
                                    </span>
                                  ) : (
                                    'Diagnose'
                                  )}
                                </Button>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="flex-1 relative">
              {isSettingsOpen ? (
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
                  onClick={() => {
                    setIsSettingsOpen(true)
                    setIsPanelOpen(false)
                  }}
                  aria-label="Expand generation settings panel"
                >
                  <span>Generation settings</span>
                  <FontIcon type="chevron-down" className="w-4 h-4 ml-2" />
                </Button>
              )}
              {isSettingsOpen && (
                <div className="absolute left-0 right-0 top-full w-full rounded-b-xl bg-card border border-border border-t-0 p-4 shadow-xl z-50">
                  <div className="w-full">
                    <div className="h-px w-full bg-border" />
                  </div>
                  <div className="mt-4 space-y-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">
                        Use test data
                      </span>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={useTestData}
                          onCheckedChange={(v: boolean) =>
                            setUseTestData(Boolean(v))
                          }
                          aria-label="Use test data"
                        />
                        <span className="text-muted-foreground">
                          {useTestData ? 'On' : 'Off'}
                        </span>
                      </div>
                    </div>
                    <div className="h-px w-full bg-border" />
                    <label className="inline-flex items-center gap-2">
                      <Checkbox
                        checked={showGenSettings}
                        onCheckedChange={(v: boolean | 'indeterminate') =>
                          setShowGenSettings(Boolean(v))
                        }
                      />
                      <span className="whitespace-nowrap">
                        Show generation settings in responses
                      </span>
                    </label>
                    <div className="h-px w-full bg-border" />
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">Temperature</span>
                      <Input
                        type="number"
                        step="0.1"
                        min="0"
                        max="2"
                        value={gen.temperature}
                        onChange={e =>
                          setGen({
                            ...gen,
                            temperature: Number(e.target.value),
                          })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">Top‑p</span>
                      <Input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={gen.topP}
                        onChange={e =>
                          setGen({ ...gen, topP: Number(e.target.value) })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">Max tokens</span>
                      <Input
                        type="number"
                        step="1"
                        min="1"
                        value={gen.maxTokens}
                        onChange={e =>
                          setGen({ ...gen, maxTokens: Number(e.target.value) })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">
                        Presence penalty
                      </span>
                      <Input
                        type="number"
                        step="0.1"
                        min="-2"
                        max="2"
                        value={gen.presencePenalty}
                        onChange={e =>
                          setGen({
                            ...gen,
                            presencePenalty: Number(e.target.value),
                          })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">
                        Frequency penalty
                      </span>
                      <Input
                        type="number"
                        step="0.1"
                        min="-2"
                        max="2"
                        value={gen.frequencyPenalty}
                        onChange={e =>
                          setGen({
                            ...gen,
                            frequencyPenalty: Number(e.target.value),
                          })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 items-center">
                      <span className="text-muted-foreground">Seed</span>
                      <Input
                        type="number"
                        step="1"
                        value={gen.seed as any}
                        onChange={e =>
                          setGen({
                            ...gen,
                            seed:
                              e.target.value === ''
                                ? ''
                                : Number(e.target.value),
                          })
                        }
                        className="col-span-2"
                      />
                    </div>
                    <div className="flex items-center gap-4">
                      <label className="inline-flex items-center gap-2">
                        <Checkbox
                          checked={gen.streaming}
                          onCheckedChange={(v: boolean | 'indeterminate') =>
                            setGen({ ...gen, streaming: Boolean(v) })
                          }
                        />
                        <span>Streaming</span>
                      </label>
                      <label className="inline-flex items-center gap-2">
                        <Checkbox
                          checked={gen.jsonMode}
                          onCheckedChange={(v: boolean | 'indeterminate') =>
                            setGen({ ...gen, jsonMode: Boolean(v) })
                          }
                        />
                        <span>JSON mode</span>
                      </label>
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
              <ConfigEditor className="h-full" />
            </div>
          ) : (
            <div className="h-full">
              <TestChat
                {...({
                  showReferences,
                  allowRanking,
                  useTestData,
                  showPrompts,
                  showThinking,
                  showGenSettings,
                } as any)}
              />
            </div>
          )}
        </div>
        {/* Helper text below chat window, on dark background */}
        {mode === 'designer' && (
          <div className="px-1 pb-3">
            <div className="text-[11px] text-muted-foreground">
              Not sure where to start? Think about the questions people will
              actually ask to test model reliability.
            </div>
          </div>
        )}
      </div>

      {isEditOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/70">
          <div className="w-[860px] max-w-[95vw] rounded-xl overflow-hidden bg-card text-foreground shadow-xl">
            <div className="flex items-center justify-between px-5 py-4 border-b border-border">
              <div className="text-sm">Edit test case</div>
              <FontIcon
                type="close"
                isButton
                handleOnClick={() => setIsEditOpen(false)}
                className="w-5 h-5 text-foreground"
              />
            </div>

            <div className="p-5 flex flex-col gap-3">
              <div>
                <label className="text-xs text-muted-foreground">
                  Test name
                </label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={e =>
                    setEditForm({ ...editForm, name: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Input</label>
                <textarea
                  rows={3}
                  value={editForm.input}
                  onChange={e =>
                    setEditForm({ ...editForm, input: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground code-like"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Expected output (baseline)
                </label>
                <textarea
                  rows={3}
                  value={editForm.expected}
                  onChange={e =>
                    setEditForm({ ...editForm, expected: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground code-like"
                />
              </div>
            </div>

            <div className="px-5 py-4 flex items-center justify-between bg-muted">
              <div className="flex items-center gap-2">
                <span className="text-sm">Match score</span>
                <span
                  className={`px-2 py-0.5 rounded-2xl text-xs ${scorePillClasses(tests.find(t => t.id === editId)?.score ?? 0)}`}
                >
                  {tests.find(t => t.id === editId)?.score ?? 0}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                {editId != null && (
                  <Button
                    variant="destructive"
                    onClick={() => {
                      deleteTest(editId)
                      setIsEditOpen(false)
                    }}
                  >
                    Delete
                  </Button>
                )}
                <Button variant="secondary" onClick={() => saveEdit(false)}>
                  Save changes
                </Button>
                <Button onClick={() => saveEdit(true)}>Save and run</Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {isCreateOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/70">
          <div className="w-[860px] max-w-[95vw] rounded-xl overflow-hidden bg-card text-foreground shadow-xl">
            <div className="flex items-center justify-between px-5 py-4 border-b border-border">
              <div className="text-sm">New test</div>
              <FontIcon
                type="close"
                isButton
                handleOnClick={() => setIsCreateOpen(false)}
                className="w-5 h-5 text-foreground"
              />
            </div>

            <div className="p-5 flex flex-col gap-3">
              <div>
                <label className="text-xs text-muted-foreground">
                  Test name
                </label>
                <input
                  type="text"
                  placeholder="Test name here"
                  value={createForm.name}
                  onChange={e =>
                    setCreateForm({ ...createForm, name: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Input</label>
                <textarea
                  rows={3}
                  placeholder="Enter the input prompt to test"
                  value={createForm.input}
                  onChange={e =>
                    setCreateForm({ ...createForm, input: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground code-like"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Expected output (baseline)
                </label>
                <textarea
                  rows={3}
                  placeholder="Add expected baseline output"
                  value={createForm.expected}
                  onChange={e =>
                    setCreateForm({ ...createForm, expected: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground code-like"
                />
              </div>
            </div>

            <div className="px-5 py-4 flex items-center justify-end bg-muted">
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  onClick={() => setIsCreateOpen(false)}
                >
                  Cancel
                </Button>
                <Button onClick={() => saveCreate(true)}>Save and run</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Test
