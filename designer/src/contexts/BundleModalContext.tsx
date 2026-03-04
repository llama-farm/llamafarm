import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import confetti from 'canvas-confetti'
import FontIcon from '../common/FontIcon'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '../components/ui/dialog'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Checkbox } from '../components/ui/checkbox'
import {
  createBundleStream,
  getBundleDownloadUrl,
  getBundleVersion,
  type BundleRequest,
} from '../api/bundleService'
import { useEstimateBundleSize } from '../hooks/useBundles'
import { useQueryClient } from '@tanstack/react-query'
import { bundleKeys } from '../hooks/useBundles'
import { formatBytes } from '../utils/formatBytes'

type BundleModalContextValue = {
  openBundleModal: () => void
  closeBundleModal: () => void
}

const BundleModalContext = createContext<BundleModalContextValue | undefined>(
  undefined
)

export const useBundleModal = (): BundleModalContextValue => {
  const ctx = useContext(BundleModalContext)
  if (!ctx) {
    throw new Error('useBundleModal must be used within a BundleModalProvider')
  }
  return ctx
}

type Step = {
  name: string
  label: string
  status: 'pending' | 'downloading' | 'complete'
  size?: number
}

type Platform = 'linux' | 'darwin' | 'windows'
type Arch = 'x86_64' | 'arm64'
type Accelerator = 'cuda' | 'rocm' | 'vulkan' | 'cpu' | 'metal'

const PLATFORMS: { value: Platform; label: string; icon: string }[] = [
  { value: 'linux', label: 'Linux', icon: '🐧' },
  { value: 'darwin', label: 'Mac', icon: '🍎' },
  { value: 'windows', label: 'Windows', icon: '🪟' },
]

const ARCHITECTURES: { value: Arch; label: string }[] = [
  { value: 'x86_64', label: 'Intel / AMD' },
  { value: 'arm64', label: 'ARM' },
]

const ACCELERATORS: { value: Accelerator; label: string; desc: string }[] = [
  { value: 'cuda', label: 'NVIDIA (CUDA)', desc: 'GeForce, Tesla, RTX GPUs' },
  { value: 'rocm', label: 'AMD (ROCm)', desc: 'Radeon Pro, Instinct GPUs' },
  { value: 'vulkan', label: 'Vulkan', desc: 'Cross-platform GPU compute' },
  { value: 'cpu', label: 'CPU only', desc: 'Works everywhere' },
  { value: 'metal', label: 'Metal', desc: 'Apple Silicon GPU' },
]

const INVALID_COMBOS = new Set(['darwin-x86_64'])

function isAcceleratorValid(acc: Accelerator, platform: Platform): boolean {
  if (acc === 'metal' && platform !== 'darwin') return false
  return true
}

function isComboValid(platform: Platform, arch: Arch): boolean {
  return !INVALID_COMBOS.has(`${platform}-${arch}`)
}

function detectPlatform(): Platform {
  const ua = navigator.userAgent.toLowerCase()
  if (ua.includes('mac')) return 'darwin'
  if (ua.includes('win')) return 'windows'
  return 'linux'
}

function detectArch(): Arch {
  // Best guess from user agent
  const ua = navigator.userAgent.toLowerCase()
  if (ua.includes('arm') || ua.includes('aarch64')) return 'arm64'
  // Mac with Apple Silicon often reports as Intel in UA, but we can check
  if (ua.includes('mac') && (navigator as any).userAgentData?.architecture === 'arm') {
    return 'arm64'
  }
  return 'x86_64'
}

type ProviderProps = { children: React.ReactNode }

export const BundleModalProvider: React.FC<ProviderProps> = ({ children }) => {
  const queryClient = useQueryClient()
  const [isOpen, setIsOpen] = useState(false)
  const [phase, setPhase] = useState<'wizard' | 'progress' | 'success'>('wizard')

  // Wizard state
  const [platform, setPlatform] = useState<Platform>(detectPlatform)
  const [arch, setArch] = useState<Arch>(detectArch)
  const [accelerator, setAccelerator] = useState<Accelerator>('cpu')
  const [addons, setAddons] = useState<string[]>([])
  const [version, setVersion] = useState('')
  const estimateMutation = useEstimateBundleSize()

  const [bundleError, setBundleError] = useState<string | null>(null)

  // Progress state
  const [steps, setSteps] = useState<Step[]>([])
  const [overallProgress, setOverallProgress] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [isMinimized, setIsMinimized] = useState(false)

  // Success state
  const [completedBundle, setCompletedBundle] = useState<any>(null)
  const [copiedCommand, setCopiedCommand] = useState(false)

  // Auto-fix invalid combos
  useEffect(() => {
    if (!isComboValid(platform, arch)) {
      if (platform === 'darwin') setArch('arm64')
    }
    if (!isAcceleratorValid(accelerator, platform)) {
      setAccelerator('cpu')
    }
  }, [platform, arch, accelerator])

  // Estimate size when config changes
  useEffect(() => {
    if (phase !== 'wizard') return
    const req: BundleRequest = { platform, arch, accelerator, addons, version }
    estimateMutation.mutate(req)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform, arch, accelerator, addons, version, phase])

  const openBundleModal = useCallback(() => {
    setPlatform(detectPlatform())
    setArch(detectArch())
    setAccelerator('cpu')
    setAddons([])
    setVersion('')
    setPhase('wizard')
    setIsMinimized(false)
    setIsOpen(true)
    // Fetch latest version from server
    getBundleVersion().then((v) => {
      if (v && v !== 'dev') setVersion(v)
    }).catch(() => {})
  }, [])

  const closeBundleModal = useCallback(() => setIsOpen(false), [])

  const value = useMemo(
    () => ({ openBundleModal, closeBundleModal }),
    [openBundleModal, closeBundleModal]
  )

  const startBundle = useCallback(async () => {
    const req: BundleRequest = { platform, arch, accelerator, addons, version }
    setBundleError(null)
    setPhase('progress')
    setOverallProgress(0)
    setStartTime(Date.now())

    const baseSteps: Step[] = [
      { name: 'cli', label: 'CLI binary', status: 'pending' },
      { name: 'server', label: 'Server', status: 'pending' },
      { name: 'rag', label: 'RAG', status: 'pending' },
      { name: 'runtime', label: 'Runtime', status: 'pending' },
    ]
    if (accelerator !== 'cpu') {
      baseSteps.push({
        name: 'torch',
        label: `${accelerator.toUpperCase()} torch wheels`,
        status: 'pending',
      })
    }
    for (const a of addons) {
      baseSteps.push({ name: a, label: a.toUpperCase(), status: 'pending' })
    }
    baseSteps.push({ name: 'packaging', label: 'Packaging archive', status: 'pending' })
    setSteps(baseSteps)

    const controller = await createBundleStream(
      req,
      (data) => {
        setOverallProgress(data.progress ?? 0)
        setSteps((prev) =>
          prev.map((s) =>
            s.name === data.step
              ? { ...s, status: data.status, size: data.size ?? s.size }
              : s
          )
        )
      },
      (data) => {
        setOverallProgress(1)
        setCompletedBundle(data)
        setPhase('success')
        setIsMinimized(false)
        setIsOpen(true)
        queryClient.invalidateQueries({ queryKey: bundleKeys.list() })
      },
      (msg) => {
        console.error('Bundle error:', msg)
        setBundleError(typeof msg === 'string' ? msg : String(msg))
        setPhase('wizard')
      }
    )
    abortRef.current = controller
  }, [platform, arch, accelerator, addons, version, queryClient])

  const cancelBundle = useCallback(() => {
    abortRef.current?.abort()
    setPhase('wizard')
  }, [])

  // Confetti on success
  useEffect(() => {
    if (phase !== 'success') return
    try {
      if (
        window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      )
        return
    } catch {}

    const isDark = document.documentElement.classList.contains('dark')
    const colors = isDark
      ? ['#14b8a6', '#f472b6', '#38bdf8', '#ffffff']
      : ['#0d9488', '#ec4899', '#38bdf8', '#0f172a']

    confetti({ particleCount: 60, spread: 60, angle: 60, origin: { x: 0.15, y: 0.2 }, colors })
    confetti({ particleCount: 60, spread: 60, angle: 120, origin: { x: 0.85, y: 0.2 }, colors })
    setTimeout(
      () =>
        confetti({ particleCount: 80, spread: 70, origin: { x: 0.5, y: 0.25 }, colors }),
      300
    )
  }, [phase])

  const toggleAddon = (name: string) => {
    setAddons((prev) =>
      prev.includes(name) ? prev.filter((a) => a !== name) : [...prev, name]
    )
  }

  const estimatedSize = estimateMutation.data?.estimated_bytes ?? 0

  return (
    <BundleModalContext.Provider value={value}>
      {children}
      <Dialog
        open={isOpen}
        onOpenChange={(open) => {
          if (!open && phase === 'progress') {
            setIsMinimized(true)
            setIsOpen(false)
            return
          }
          setIsOpen(open)
        }}
      >
        <DialogContent
          className="sm:max-w-2xl top-[45%]"
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          {/* WIZARD PHASE */}
          {phase === 'wizard' && (
            <>
              <DialogHeader>
                <DialogTitle className="text-lg text-foreground">
                  Create Bundle
                </DialogTitle>
                <DialogDescription>
                  Build a distributable archive for deploying LlamaFarm to
                  another machine.
                </DialogDescription>
              </DialogHeader>

              <div className="flex flex-col gap-5 mt-2">
                {bundleError && (
                  <div className="rounded-md border border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/30 px-4 py-3 text-sm text-red-700 dark:text-red-300">
                    <div className="font-medium mb-1">Bundle failed</div>
                    <div className="text-xs break-all">{bundleError}</div>
                  </div>
                )}
                {/* Platform */}
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Where is this going?
                  </label>
                  <div className="mt-2 flex gap-2">
                    {PLATFORMS.map((p) => (
                      <button
                        key={p.value}
                        type="button"
                        onClick={() => setPlatform(p.value)}
                        className={`flex-1 flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm transition-colors ${
                          platform === p.value
                            ? 'border-primary bg-primary/10 text-foreground font-medium'
                            : 'border-border bg-card text-muted-foreground hover:bg-accent/20'
                        }`}
                      >
                        <span className="text-lg">{p.icon}</span>
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Architecture */}
                <div>
                  <label className="text-sm font-medium text-foreground">
                    What chip?
                  </label>
                  <div className="mt-2 flex gap-2">
                    {ARCHITECTURES.map((a) => {
                      const disabled = !isComboValid(platform, a.value)
                      return (
                        <button
                          key={a.value}
                          type="button"
                          disabled={disabled}
                          onClick={() => setArch(a.value)}
                          className={`flex-1 rounded-lg border px-4 py-3 text-sm transition-colors ${
                            disabled
                              ? 'opacity-40 cursor-not-allowed border-border bg-muted text-muted-foreground'
                              : arch === a.value
                                ? 'border-primary bg-primary/10 text-foreground font-medium'
                                : 'border-border bg-card text-muted-foreground hover:bg-accent/20'
                          }`}
                        >
                          {a.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Accelerator */}
                <div>
                  <label className="text-sm font-medium text-foreground">
                    GPU acceleration
                  </label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {ACCELERATORS.map((a) => {
                      const disabled = !isAcceleratorValid(a.value, platform)
                      return (
                        <button
                          key={a.value}
                          type="button"
                          disabled={disabled}
                          onClick={() => setAccelerator(a.value)}
                          className={`rounded-lg border px-3 py-2 text-sm transition-colors ${
                            disabled
                              ? 'opacity-40 cursor-not-allowed border-border bg-muted text-muted-foreground'
                              : accelerator === a.value
                                ? 'border-primary bg-primary/10 text-foreground font-medium'
                                : 'border-border bg-card text-muted-foreground hover:bg-accent/20'
                          }`}
                        >
                          {a.label}
                        </button>
                      )
                    })}
                  </div>
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    Not sure? &quot;CPU only&quot; works everywhere. Pick NVIDIA if the
                    target has a GeForce/Tesla GPU.
                  </p>
                </div>

                {/* Addons */}
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Addons (optional)
                  </label>
                  <div className="mt-2 flex flex-col gap-2">
                    {['stt', 'tts'].map((addon) => (
                      <label
                        key={addon}
                        className="flex items-center gap-2 cursor-pointer"
                      >
                        <Checkbox
                          checked={addons.includes(addon)}
                          onCheckedChange={() => toggleAddon(addon)}
                        />
                        <span className="text-sm text-foreground">
                          {addon === 'stt'
                            ? 'Speech-to-Text (STT)'
                            : 'Text-to-Speech (TTS)'}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Version */}
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Version
                  </label>
                  <Input
                    className="mt-1"
                    placeholder="Leave blank for current version"
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                  />
                </div>
              </div>

              <DialogFooter>
                <div className="w-full flex items-center justify-between gap-2">
                  <div className="text-xs text-muted-foreground">
                    {estimatedSize > 0
                      ? `Estimated size: ~${formatBytes(estimatedSize)}`
                      : '\u00A0'}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" onClick={closeBundleModal}>
                      Cancel
                    </Button>
                    <Button onClick={startBundle}>Create Bundle</Button>
                  </div>
                </div>
              </DialogFooter>
            </>
          )}

          {/* PROGRESS PHASE */}
          {phase === 'progress' && (
            <>
              <DialogHeader>
                <DialogTitle className="text-lg text-foreground">
                  Building bundle...
                </DialogTitle>
                <DialogDescription>Leave this window open</DialogDescription>
              </DialogHeader>

              <div className="flex flex-col gap-3">
                {/* Overall progress bar */}
                <div className="w-full rounded-md border border-border p-3 bg-card">
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-full rounded-full bg-accent/20">
                      <div
                        className="h-2 rounded-full bg-primary transition-all"
                        style={{
                          width: `${Math.floor(overallProgress * 100)}%`,
                        }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground whitespace-nowrap">
                      {Math.floor(overallProgress * 100)}%
                    </div>
                  </div>
                  {startTime && overallProgress > 0 && (
                    <div className="mt-1 text-xs text-muted-foreground text-right">
                      {(() => {
                        const elapsed = (Date.now() - startTime) / 1000
                        const total = elapsed / overallProgress
                        const remaining = Math.max(0, total - elapsed)
                        const m = Math.floor(remaining / 60)
                        const s = Math.floor(remaining % 60)
                        return `~${m > 0 ? m + 'm ' : ''}${s}s left`
                      })()}
                    </div>
                  )}
                </div>

                {/* Step list */}
                <div className="flex flex-col gap-1.5">
                  {steps.map((step) => (
                    <div
                      key={step.name}
                      className="flex items-center gap-2 text-sm"
                    >
                      <span className="w-5 text-center flex items-center justify-center">
                        {step.status === 'complete'
                          ? <span className="w-4 h-4 text-green-500"><FontIcon type="checkmark-filled" /></span>
                          : step.status === 'downloading'
                            ? <span className="w-4 h-4 animate-spin"><FontIcon type="loading" /></span>
                            : <span className="w-3 h-3 rounded-full border border-muted-foreground" />}
                      </span>
                      <span
                        className={
                          step.status === 'complete'
                            ? 'text-foreground'
                            : step.status === 'downloading'
                              ? 'text-primary font-medium'
                              : 'text-muted-foreground'
                        }
                      >
                        {step.label}
                      </span>
                      {step.size ? (
                        <span className="text-xs text-muted-foreground ml-auto">
                          {formatBytes(step.size)}
                        </span>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>

              <DialogFooter>
                <div className="w-full flex items-center justify-end gap-2">
                  <Button variant="outline" onClick={cancelBundle}>
                    Cancel
                  </Button>
                  <Button
                    onClick={() => {
                      setIsMinimized(true)
                      setIsOpen(false)
                    }}
                  >
                    Background
                  </Button>
                </div>
              </DialogFooter>
            </>
          )}

          {/* SUCCESS PHASE */}
          {phase === 'success' && completedBundle && (
            <>
              <DialogHeader>
                <DialogTitle className="text-lg text-foreground">
                  Bundle ready!
                </DialogTitle>
                <DialogDescription>
                  {completedBundle.filename}
                </DialogDescription>
              </DialogHeader>

              <div className="flex flex-col gap-3">
                <div className="rounded-md border border-border bg-card p-4 text-sm">
                  <div className="font-medium text-foreground">
                    {completedBundle.filename}
                  </div>
                  <div className="text-muted-foreground mt-1">
                    {formatBytes(completedBundle.size)} ·{' '}
                    {completedBundle.platform} · {completedBundle.arch} ·{' '}
                    {completedBundle.accelerator}
                  </div>
                </div>

                <div className="rounded-md border border-border bg-card p-4 text-sm">
                  <div className="font-medium text-foreground mb-2">
                    Next steps
                  </div>
                  <ol className="list-decimal pl-5 space-y-1 text-muted-foreground">
                    <li>Transfer this file to your target machine</li>
                    <li>
                      Run:{' '}
                      <code className="text-foreground font-mono text-xs bg-muted px-1 py-0.5 rounded">
                        ./install.sh {completedBundle.filename}
                      </code>
                    </li>
                  </ol>
                </div>
              </div>

              <DialogFooter>
                <div className="w-full flex items-center justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      navigator.clipboard.writeText(
                        `./install.sh ${completedBundle.filename}`
                      )
                      setCopiedCommand(true)
                      setTimeout(() => setCopiedCommand(false), 1500)
                    }}
                  >
                    {copiedCommand ? 'Copied!' : 'Copy install command'}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      window.open(
                        getBundleDownloadUrl(completedBundle.id),
                        '_blank'
                      )
                    }}
                  >
                    Download
                  </Button>
                  <Button
                    onClick={() => {
                      setPhase('wizard')
                      closeBundleModal()
                    }}
                  >
                    Done
                  </Button>
                </div>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Floating pill when minimized during progress */}
      {isMinimized && phase === 'progress' && (
        <div
          role="button"
          tabIndex={0}
          onClick={() => {
            setIsMinimized(false)
            setIsOpen(true)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setIsMinimized(false)
              setIsOpen(true)
            }
          }}
          className="fixed bottom-4 right-4 z-[100] w-[320px] rounded-lg border border-border bg-card text-card-foreground shadow-lg p-3 text-left"
          aria-label="Show bundle progress"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">Building bundle...</div>
            <button
              type="button"
              className="h-7 px-2 rounded-md border border-input text-xs hover:bg-accent/30"
              onClick={(e) => {
                e.stopPropagation()
                setIsMinimized(false)
                setIsOpen(true)
              }}
            >
              View
            </button>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-2 w-full rounded-full bg-accent/20">
              <div
                className="h-2 rounded-full bg-primary transition-all"
                style={{ width: `${Math.floor(overallProgress * 100)}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground whitespace-nowrap">
              {Math.floor(overallProgress * 100)}%
            </div>
          </div>
        </div>
      )}
    </BundleModalContext.Provider>
  )
}
