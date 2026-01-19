import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  useRef,
  useEffect,
} from 'react'
import confetti from 'canvas-confetti'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '../components/ui/dialog'
import { Input } from '../components/ui/input'
import FontIcon from '../common/FontIcon'
import { useNavigate } from 'react-router-dom'

type PackageModalContextValue = {
  openPackageModal: () => void
  closePackageModal: () => void
}

const PackageModalContext = createContext<PackageModalContextValue | undefined>(
  undefined
)

export const usePackageModal = (): PackageModalContextValue => {
  const ctx = useContext(PackageModalContext)
  if (!ctx) {
    throw new Error(
      'usePackageModal must be used within a PackageModalProvider'
    )
  }
  return ctx
}

type ProviderProps = {
  children: React.ReactNode
}

export const PackageModalProvider: React.FC<ProviderProps> = ({ children }) => {
  const navigate = useNavigate()
  const [isOpen, setIsOpen] = useState(false)
  const [version, setVersion] = useState<string>(() => {
    try {
      const raw = localStorage.getItem('lf_versions')
      if (raw) {
        const arr = JSON.parse(raw)
        if (Array.isArray(arr) && arr.length > 0) {
          const current = arr.find((v: any) => v?.isCurrent)
          const picked = current || arr[0]
          return picked?.name || picked?.id || 'v1.0.0'
        }
      }
    } catch {}
    return 'v1.0.0'
  })
  const [versionError, setVersionError] = useState<string>('')
  const [savingTo, setSavingTo] = useState<string>(() => {
    try {
      // Best-effort default path; OS-agnostic fallback
      const home =
        (typeof window !== 'undefined' && (window as any).process?.env?.HOME) ||
        '/Users/you'
      return `${home}/Downloads/`
    } catch {
      return '/Users/you/Downloads/'
    }
  })
  const [description, setDescription] = useState<string>('')
  const [changesOpen, setChangesOpen] = useState<boolean>(false)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const [isPackaging, setIsPackaging] = useState<boolean>(false)
  const [progress, setProgress] = useState<number>(0)
  const [currentStep, setCurrentStep] = useState<string>('')
  const packagingTimerRef = useRef<number | null>(null)
  const [isSuccess, setIsSuccess] = useState<boolean>(false)
  const [packStartTs, setPackStartTs] = useState<number | null>(null)
  const [copiedPath, setCopiedPath] = useState<boolean>(false)
  const [dirHandle, setDirHandle] = useState<any>(null)
  const [isMinimized, setIsMinimized] = useState<boolean>(false)

  const versionExists = useCallback((name: string): boolean => {
    if (!name) return false
    try {
      const raw = localStorage.getItem('lf_versions')
      if (!raw) return false
      const arr = JSON.parse(raw)
      if (!Array.isArray(arr)) return false
      return arr.some((v: any) => {
        const matches = (v?.id || v?.name) === name
        if (!matches) return false
        // Allow reuse if previous attempt was stopped or still packaging
        const status = v?.status
        if (status === 'stopped' || status === 'packaging') return false
        return true
      })
    } catch {
      return false
    }
  }, [])

  const openFolderPicker = useCallback(async () => {
    try {
      const anyWindow = window as any
      if (anyWindow && typeof anyWindow.showDirectoryPicker === 'function') {
        const handle = await anyWindow.showDirectoryPicker({
          mode: 'readwrite',
        })
        // Full absolute path is not exposed for privacy; show selected folder name
        setSavingTo(`${handle.name}/`)
        setDirHandle(handle)
        return
      }
    } catch (error: any) {
      // Only fall back if the API is not supported, not if user canceled
      if (error?.name === 'AbortError') {
        // User canceled the picker, don't fall back
        return
      }
      // For other errors (like API not supported), fall back
    }
    try {
      folderInputRef.current?.click()
    } catch {}
  }, [])

  const openPackageModal = useCallback(() => {
    try {
      const raw = localStorage.getItem('lf_versions')
      if (raw) {
        const arr = JSON.parse(raw)
        if (Array.isArray(arr) && arr.length > 0) {
          const current = arr.find((v: any) => v?.isCurrent)
          const picked = current || arr[0]
          const nextName = picked?.name || picked?.id
          if (nextName && typeof nextName === 'string') {
            setVersion(nextName)
          }
        } else {
          setVersion('v1.0.0')
        }
      } else {
        setVersion('v1.0.0')
      }
    } catch {
      setVersion('v1.0.0')
    }
    setIsMinimized(false)
    setIsOpen(true)
  }, [])
  const closePackageModal = useCallback(() => setIsOpen(false), [])

  const value = useMemo(
    () => ({ openPackageModal, closePackageModal }),
    [openPackageModal, closePackageModal]
  )

  useEffect(() => {
    const handler = () => {
      openPackageModal()
    }
    window.addEventListener('lf_open_package_modal', handler as EventListener)
    return () =>
      window.removeEventListener(
        'lf_open_package_modal',
        handler as EventListener
      )
  }, [openPackageModal])

  // Fire lightweight canvas confetti on success (CDN script, no package install)
  useEffect(() => {
    if (!isSuccess) return
    try {
      if (
        window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      ) {
        return
      }
    } catch {}

    const isDark = document.documentElement.classList.contains('dark')
    const colors = isDark
      ? ['#14b8a6', '#f472b6', '#38bdf8', '#ffffff'] // teal-500, pink-400, sky-400, white
      : ['#0d9488', '#ec4899', '#38bdf8', '#0f172a'] // teal-600, pink-500, sky-400, slate-900

    confetti({
      particleCount: 60,
      spread: 60,
      angle: 60,
      origin: { x: 0.15, y: 0.2 },
      colors,
    })
    confetti({
      particleCount: 60,
      spread: 60,
      angle: 120,
      origin: { x: 0.85, y: 0.2 },
      colors,
    })
    setTimeout(
      () =>
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { x: 0.5, y: 0.25 },
          colors,
        }),
      300
    )
  }, [isSuccess])

  return (
    <PackageModalContext.Provider value={value}>
      {children}
      <Dialog
        open={isOpen}
        onOpenChange={open => {
          if (!open && isPackaging) {
            setIsMinimized(true)
            setIsOpen(false)
            return
          }
          setIsOpen(open)
        }}
      >
        <DialogContent
          className="sm:max-w-2xl top-[45%]"
          onOpenAutoFocus={e => e.preventDefault()}
        >
          <DialogHeader>
            <DialogTitle
              className="text-lg text-foreground inline-flex w-fit"
              onClick={() => {
                if (isSuccess) {
                  try {
                    const confetti = (window as any).confetti
                    if (confetti) {
                      const isDark =
                        document.documentElement.classList.contains('dark')
                      const colors = isDark
                        ? ['#14b8a6', '#f472b6', '#38bdf8', '#ffffff']
                        : ['#0d9488', '#ec4899', '#38bdf8', '#0f172a']
                      confetti({
                        particleCount: 60,
                        spread: 60,
                        angle: 60,
                        origin: { x: 0.15, y: 0.2 },
                        colors,
                      })
                      confetti({
                        particleCount: 60,
                        spread: 60,
                        angle: 120,
                        origin: { x: 0.85, y: 0.2 },
                        colors,
                      })
                      setTimeout(
                        () =>
                          confetti({
                            particleCount: 80,
                            spread: 70,
                            origin: { x: 0.5, y: 0.25 },
                            colors,
                          }),
                        300
                      )
                    }
                  } catch {}
                }
              }}
              role={isSuccess ? 'button' : undefined}
              tabIndex={isSuccess ? 0 : undefined}
              onKeyDown={e => {
                if (isSuccess && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault()
                  const confetti = (window as any).confetti
                  if (confetti) {
                    const isDark =
                      document.documentElement.classList.contains('dark')
                    const colors = isDark
                      ? ['#14b8a6', '#f472b6', '#38bdf8', '#ffffff']
                      : ['#0d9488', '#ec4899', '#38bdf8', '#0f172a']
                    confetti({
                      particleCount: 60,
                      spread: 60,
                      angle: 60,
                      origin: { x: 0.15, y: 0.2 },
                      colors,
                    })
                    confetti({
                      particleCount: 60,
                      spread: 60,
                      angle: 120,
                      origin: { x: 0.85, y: 0.2 },
                      colors,
                    })
                    setTimeout(
                      () =>
                        confetti({
                          particleCount: 80,
                          spread: 70,
                          origin: { x: 0.5, y: 0.25 },
                          colors,
                        }),
                      300
                    )
                  }
                }
              }}
              style={{ cursor: isSuccess ? 'pointer' : 'default' }}
            >
              {isSuccess
                ? '🎉  Project successfully packaged!'
                : isPackaging
                  ? 'Packaging Your LlamaFarm Project...'
                  : 'Package Your LlamaFarm Project'}
            </DialogTitle>
            {isSuccess ? (
              <DialogDescription>
                {`Your ${version}.llamafarm project has been packaged successfully.`}
              </DialogDescription>
            ) : isPackaging ? (
              <DialogDescription>Leave this window open</DialogDescription>
            ) : (
              <DialogDescription>
                Generate a deployable version of your config file with all
                inputs, prompts, and model settings baked in. Export it to run
                locally, test in staging, or hand it off to your team.
              </DialogDescription>
            )}
          </DialogHeader>
          {!isPackaging && !isSuccess ? (
            <div className="flex flex-col gap-3">
              {/* Version */}
              <div>
                <label className="text-xs text-muted-foreground">Version</label>
                <div className="mt-1 grid grid-cols-3 gap-2 items-center">
                  <Input
                    value={version}
                    onChange={e => {
                      const v = e.target.value
                      setVersion(v)
                      if (/^[a-zA-Z0-9._-]+$/.test(v)) {
                        setVersionError('')
                      } else {
                        setVersionError(
                          'Use letters, numbers, dashes (-), underscores (_), or dots (.) only'
                        )
                      }
                    }}
                    className={`col-span-2 ${versionError ? 'border-red-500 dark:border-red-300' : ''}`}
                    placeholder="v1.0.1"
                    aria-invalid={versionError ? true : undefined}
                    data-invalid={versionError ? true : undefined}
                  />
                  <div className="col-span-1 text-sm text-muted-foreground truncate">
                    .llamafarm
                  </div>
                </div>
              </div>
              <div className="h-2 -mt-1">
                {versionError ? (
                  <div className="text-xs leading-3 mt-0 text-red-500 dark:text-red-300">
                    {versionError}
                  </div>
                ) : null}
              </div>

              {/* Saving to (read-only) */}
              <div className="-mt-2">
                <label className="text-xs text-muted-foreground">
                  Saving to
                </label>
                <div className="mt-0 grid grid-cols-[1fr_auto] gap-2 items-center">
                  <Input value={savingTo} readOnly className="truncate" />
                  <button
                    type="button"
                    className="w-9 h-9 inline-flex items-center justify-center rounded-md border border-input text-slate-900 dark:text-white hover:bg-accent/30 leading-none"
                    aria-label="Change folder"
                    onClick={openFolderPicker}
                  >
                    <FontIcon type="folder" className="w-5 h-5" />
                  </button>
                  {/* Hidden input fallback for browsers without showDirectoryPicker */}
                  <input
                    ref={folderInputRef}
                    type="file"
                    // @ts-expect-error Non-standard attribute supported in Blink/WebKit
                    webkitdirectory=""
                    style={{ display: 'none' }}
                    onChange={e => {
                      try {
                        const files = e.target.files
                        if (files && files.length > 0) {
                          const first: any = files[0]
                          const rel: string = first?.webkitRelativePath || ''
                          const top = rel.split('/')[0] || 'Selected folder'
                          setSavingTo(`${top}/`)
                        }
                        ;(e.target as HTMLInputElement).value = ''
                      } catch {}
                    }}
                  />
                </div>
              </div>

              {/* Description (optional) */}
              <div>
                <label className="text-xs text-muted-foreground">
                  Description (optional)
                </label>
                <Input
                  className="mt-1"
                  placeholder="Add a short description of what’s included in this package"
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                />
              </div>

              {/* What's changed (accordion) */}
              <section className="rounded-lg border border-border bg-card">
                <button
                  type="button"
                  className="w-full flex items-center justify-between px-3 py-2 text-left"
                  onClick={() => setChangesOpen(o => !o)}
                  aria-expanded={changesOpen}
                >
                  <div className="text-sm font-medium">
                    What’s changed since v1.0.0
                  </div>
                  <FontIcon
                    type="chevron-down"
                    className={`w-4 h-4 transition-transform ${changesOpen ? 'rotate-180' : ''}`}
                  />
                </button>
                {changesOpen ? (
                  <div className="px-3 pb-3">
                    <div className="rounded-md border border-border bg-background p-3 text-xs">
                      <div className="mb-1 font-medium">
                        Prompts (3 modified)
                      </div>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>
                          <span className="font-mono">system_prompt.txt</span> –
                          updated instructions
                        </li>
                        <li>
                          <span className="font-mono">
                            few_shot_examples.txt
                          </span>{' '}
                          – added 2 examples
                        </li>
                        <li>
                          <span className="font-mono">
                            validation_prompt.txt
                          </span>{' '}
                          – fixed typos
                        </li>
                      </ul>
                      <div className="mt-3 font-medium">Datasets (1 added)</div>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>
                          <span className="font-mono">
                            dataset_telemetry_2025.csv
                          </span>
                        </li>
                      </ul>
                      <div className="mt-3 font-medium">Model</div>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>Hyperparameters tuned for retrieval fidelity</li>
                      </ul>
                    </div>
                  </div>
                ) : null}
              </section>
            </div>
          ) : isPackaging ? (
            <div className="flex flex-col gap-3">
              <div className="w-full rounded-md border border-border p-3 bg-card">
                <div className="flex items-center gap-3">
                  <div className="h-2 w-full rounded-full bg-accent/20">
                    <div
                      className="h-2 rounded-full bg-primary transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="text-xs text-muted-foreground whitespace-nowrap">{`${Math.floor(progress)}%`}</div>
                </div>
                <div className="mt-2 text-xs text-muted-foreground flex items-center justify-between">
                  <span>{currentStep}</span>
                  {packStartTs ? (
                    <span className="whitespace-nowrap">
                      {(() => {
                        const elapsed = (Date.now() - packStartTs) / 1000
                        const pct = Math.max(progress, 1)
                        const estTotal = (elapsed / pct) * 100
                        const remaining = Math.max(0, estTotal - elapsed)
                        const minutes = Math.floor(remaining / 60)
                        const seconds = Math.floor(remaining % 60)
                        return `~${minutes > 0 ? minutes + 'm ' : ''}${seconds}s left`
                      })()}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <div>
                <div className="text-sm text-foreground font-medium">Name</div>
                <div className="mt-1 grid grid-cols-[1fr] gap-2 items-center">
                  <Input
                    readOnly
                    value={`${version}.llamafarm (2.34GB)`}
                    className="truncate"
                  />
                </div>
              </div>
              <div className="mb-5">
                <div className="text-sm text-foreground font-medium">
                  Saved to
                </div>
                <div className="mt-1 grid grid-cols-[1fr_auto_auto] gap-2 items-center">
                  <Input readOnly value={savingTo} className="truncate" />
                  <button
                    type="button"
                    className={`h-9 px-3 rounded-md border text-sm hover:bg-accent/30 inline-flex items-center justify-center gap-2 leading-none ${copiedPath ? 'border-teal-400 text-teal-400' : 'border-input'}`}
                    onClick={() => {
                      try {
                        navigator.clipboard.writeText(savingTo)
                        setCopiedPath(true)
                        window.setTimeout(() => setCopiedPath(false), 1200)
                      } catch {}
                    }}
                  >
                    {copiedPath ? (
                      <>
                        <FontIcon type="checkmark-filled" className="w-4 h-4" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      'Copy path'
                    )}
                  </button>
                  <button
                    type="button"
                    className="h-9 px-3 rounded-md border border-input text-sm hover:bg-accent/30"
                    onClick={async () => {
                      try {
                        const anyWindow = window as any
                        if (anyWindow?.showDirectoryPicker) {
                          if (dirHandle) {
                            await anyWindow.showDirectoryPicker({
                              startIn: dirHandle,
                            })
                          } else {
                            await anyWindow.showDirectoryPicker({
                              startIn: 'downloads',
                            })
                          }
                        }
                      } catch {}
                    }}
                  >
                    Open folder
                  </button>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            {!isPackaging && !isSuccess ? (
              <div className="w-full flex items-center justify-between gap-2">
                <div className="text-xs text-muted-foreground">
                  Estimated size ~2.3GB
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                    onClick={closePackageModal}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                    onClick={() => {
                      // Prevent packaging if version invalid or duplicate
                      if (!version || versionError || versionExists(version)) {
                        if (versionExists(version)) {
                          setVersionError(
                            'A version with this name already exists'
                          )
                        }
                        return
                      }
                      setIsPackaging(true)
                      setPackStartTs(Date.now())
                      setProgress(0)
                      const steps = [
                        'Validating configuration...',
                        'Collecting prompts and inputs...',
                        'Embedding and compressing model files...',
                        'Bundling datasets and assets...',
                        'Generating package metadata...',
                      ]
                      let stepIndex = 0
                      setCurrentStep(steps[stepIndex])
                      // Insert a transient packaging row at the top for live feedback
                      try {
                        const raw0 = localStorage.getItem('lf_versions')
                        const list0 = raw0 ? (JSON.parse(raw0) as any[]) : []
                        // Clear current flags and drop any existing rows with same version id/name
                        const cleared0 = (Array.isArray(list0) ? list0 : [])
                          .map(v => ({ ...v, isCurrent: false }))
                          .filter(v => (v?.id || v?.name) !== version)
                        const transient = {
                          id: version,
                          name: version,
                          description: description || 'Packaging…',
                          date: new Date().toLocaleString(),
                          isCurrent: true,
                          size: '—',
                          path: savingTo,
                          status: 'packaging',
                        }
                        localStorage.setItem(
                          'lf_versions',
                          JSON.stringify([transient, ...cleared0])
                        )
                        try {
                          window.dispatchEvent(
                            new CustomEvent('lf_versions_updated', {
                              detail: { source: 'packager-start' },
                            })
                          )
                        } catch {}
                      } catch {}

                      const id = window.setInterval(() => {
                        setProgress(p => {
                          const next = Math.min(100, p + Math.random() * 3 + 1)
                          if (
                            next > (stepIndex + 1) * 20 &&
                            stepIndex < steps.length - 1
                          ) {
                            stepIndex += 1
                            setCurrentStep(steps[stepIndex])
                          }
                          if (next >= 100) {
                            window.clearInterval(id)
                            packagingTimerRef.current = null
                            setTimeout(() => {
                              // Persist new version into Versions table storage as current and top
                              try {
                                const raw = localStorage.getItem('lf_versions')
                                const list = raw
                                  ? (JSON.parse(raw) as any[])
                                  : []
                                const newEntry = {
                                  id: version,
                                  name: version,
                                  description:
                                    description || 'Packaged via modal',
                                  date: new Date().toLocaleString(),
                                  isCurrent: true,
                                  size: '2.34GB',
                                  path: savingTo,
                                  status: 'success',
                                }
                                // Remove any prior entries with this version to avoid duplicates
                                const cleared = (
                                  Array.isArray(list) ? list : []
                                )
                                  .filter(v => (v?.id || v?.name) !== version)
                                  .map(v => ({ ...v, isCurrent: false }))
                                const updated = [newEntry, ...cleared]
                                localStorage.setItem(
                                  'lf_versions',
                                  JSON.stringify(updated)
                                )
                                try {
                                  window.dispatchEvent(
                                    new CustomEvent('lf_versions_updated', {
                                      detail: { source: 'packager' },
                                    })
                                  )
                                } catch {}
                              } catch {}

                              setIsPackaging(false)
                              setIsSuccess(true)
                              // Always surface success modal
                              setIsMinimized(false)
                              setIsOpen(true)
                            }, 1200)
                          }
                          return next
                        })
                      }, 600)
                      packagingTimerRef.current = id
                    }}
                    type="button"
                  >
                    Package
                  </button>
                </div>
              </div>
            ) : isPackaging ? (
              <div className="w-full flex items-center justify-between gap-2">
                <div className="text-xs text-muted-foreground">
                  Estimated size ~2.3GB
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="px-3 py-2 rounded-md text-sm bg-secondary text-secondary-foreground border border-input hover:bg-secondary/80"
                    onClick={() => {
                      if (packagingTimerRef.current) {
                        window.clearInterval(packagingTimerRef.current)
                        packagingTimerRef.current = null
                      }
                      setIsPackaging(false)
                      setProgress(0)
                      setCurrentStep('')
                      try {
                        const raw = localStorage.getItem('lf_versions')
                        const list = raw ? (JSON.parse(raw) as any[]) : []
                        // Mark the transient row as stopped
                        const updated = (Array.isArray(list) ? list : []).map(
                          v =>
                            (v?.id || v?.name) === version
                              ? { ...v, status: 'stopped' }
                              : v
                        )
                        localStorage.setItem(
                          'lf_versions',
                          JSON.stringify(updated)
                        )
                        try {
                          window.dispatchEvent(
                            new CustomEvent('lf_versions_updated', {
                              detail: { source: 'packager-stopped' },
                            })
                          )
                        } catch {}
                      } catch {}
                    }}
                    type="button"
                  >
                    Stop
                  </button>
                  <button
                    className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                    onClick={() => {
                      setIsMinimized(true)
                      setIsOpen(false)
                    }}
                    type="button"
                  >
                    Background
                  </button>
                </div>
              </div>
            ) : (
              <div className="w-full flex items-center justify-end gap-2">
                <button
                  className="px-3 py-2 rounded-md text-sm border border-input bg-background hover:bg-accent/30"
                  type="button"
                  onClick={() => {
                    setIsSuccess(false)
                    closePackageModal()
                    navigate('/chat/versions')
                  }}
                >
                  Go to versions
                </button>
                <button
                  className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                  type="button"
                  onClick={() => {
                    setIsSuccess(false)
                    closePackageModal()
                  }}
                >
                  Done
                </button>
              </div>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {isMinimized && isPackaging ? (
        <div
          role="button"
          tabIndex={0}
          onClick={() => {
            setIsMinimized(false)
            setIsOpen(true)
          }}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setIsMinimized(false)
              setIsOpen(true)
            }
          }}
          className="fixed bottom-4 right-4 z-[100] w-[320px] rounded-lg border border-border bg-card text-card-foreground shadow-lg p-3 text-left"
          aria-label="Show packaging progress"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">{`Packaging ${version}...`}</div>
            <button
              type="button"
              className="h-7 px-2 rounded-md border border-input text-xs hover:bg-accent/30"
              onClick={e => {
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
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground whitespace-nowrap">{`${Math.floor(progress)}%`}</div>
          </div>
          <div className="mt-2 text-xs text-muted-foreground truncate">
            {currentStep}
          </div>
        </div>
      ) : null}
    </PackageModalContext.Provider>
  )
}
