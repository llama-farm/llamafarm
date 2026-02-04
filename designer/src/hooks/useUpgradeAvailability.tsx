import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  compareSemver,
  getStoredLatestRelease,
  normalizeVersion,
  shouldCheck,
  storeLatestRelease,
  isDismissed,
  setDismissed,
  type DismissContext,
  getGithubReleasesUrl,
} from '@/utils/versionUtils'
import { getVersionCheck } from '@/api/systemService'

const TWELVE_HOURS_MS = 12 * 60 * 60 * 1000

type UpgradeAvailabilityValue = {
  isLoading: boolean
  currentVersion: string
  latestVersion: string
  upgradeAvailable: boolean
  releasesUrl: string
  isDismissedFor: (ctx: DismissContext) => boolean
  dismiss: (ctx: DismissContext) => void
  refreshLatest: (signal?: AbortSignal) => Promise<void>
  _dismissCounter: number
}

const UpgradeAvailabilityContext =
  createContext<UpgradeAvailabilityValue | null>(null)

export function UpgradeAvailabilityProvider({
  children,
}: {
  children: ReactNode
}) {
  const [currentVersion, setCurrentVersion] = useState<string>('unknown')
  const [{ info, checkedAt }, setCache] = useState(() =>
    getStoredLatestRelease()
  )
  const initialCheckedAt = useRef(checkedAt)
  const [isLoading, setIsLoading] = useState(false)
  const [dismissCounter, setDismissCounter] = useState(0)

  // Fetch version info on mount - always get current version, cache latest release
  useEffect(() => {
    const abort = new AbortController()
    const run = async () => {
      const shouldUpdateCache = shouldCheck(
        initialCheckedAt.current,
        TWELVE_HOURS_MS
      )
      if (shouldUpdateCache) {
        setIsLoading(true)
      }
      try {
        const res = await getVersionCheck(abort.signal)
        // Always update current version from server
        if (res?.current_version) {
          setCurrentVersion(res.current_version)
        }
        // Only update cache if stale
        if (shouldUpdateCache && res?.latest_version) {
          const mapped = {
            latestVersion: res.latest_version,
            htmlUrl: res.release_url || getGithubReleasesUrl(),
            publishedAt: res.published_at,
          }
          storeLatestRelease(mapped)
          setCache({ info: mapped, checkedAt: Date.now() })
        }
      } catch (error) {
        // Ignore canceled requests (from React StrictMode cleanup)
        if ((error as any)?.isCanceled) return
        console.error('Failed to fetch version info:', error)
      }
      setIsLoading(false)
    }
    run()
    return () => abort.abort()
  }, [])

  const normalizedCurrent =
    currentVersion && currentVersion !== 'unknown'
      ? normalizeVersion(currentVersion)
      : null
  const latestVersion = useMemo(
    () => normalizeVersion(info?.latestVersion),
    [info?.latestVersion]
  )
  const upgradeAvailable = useMemo(() => {
    if (!latestVersion || !normalizedCurrent) return false
    return compareSemver(latestVersion, normalizedCurrent) > 0
  }, [latestVersion, normalizedCurrent])

  const isDismissedFor = (ctx: DismissContext) => {
    if (!latestVersion) return true
    return isDismissed(latestVersion, ctx)
  }

  const dismiss = (ctx: DismissContext) => {
    if (!latestVersion) return
    setDismissed(latestVersion, ctx, true)
    // trigger re-render for consumers so banners hide immediately
    setDismissCounter(c => c + 1)
  }

  const releasesUrl = info?.htmlUrl || getGithubReleasesUrl()

  const refreshLatest = async (signal?: AbortSignal) => {
    try {
      setIsLoading(true)
      const res = await getVersionCheck(signal)
      const latestVersion = res?.latest_version || ''
      const htmlUrl = res?.release_url || getGithubReleasesUrl()
      const publishedAt = res?.published_at
      const serverCurrentVersion = res?.current_version
      if (serverCurrentVersion) {
        setCurrentVersion(serverCurrentVersion)
      }
      if (latestVersion) {
        const mapped = { latestVersion, htmlUrl, publishedAt }
        storeLatestRelease(mapped)
        setCache({ info: mapped, checkedAt: Date.now() })
      }
    } catch (error) {
      // Ignore canceled requests
      if ((error as any)?.isCanceled) return
      console.error('Failed to fetch version info:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const value: UpgradeAvailabilityValue = {
    isLoading,
    currentVersion: normalizedCurrent ?? currentVersion,
    latestVersion,
    upgradeAvailable,
    releasesUrl,
    isDismissedFor,
    dismiss,
    refreshLatest,
    _dismissCounter: dismissCounter,
  }

  return (
    <UpgradeAvailabilityContext.Provider value={value}>
      {children}
    </UpgradeAvailabilityContext.Provider>
  )
}

export function useUpgradeAvailability(): UpgradeAvailabilityValue {
  const ctx = useContext(UpgradeAvailabilityContext)
  if (!ctx) {
    throw new Error(
      'useUpgradeAvailability must be used within UpgradeAvailabilityProvider'
    )
  }
  return ctx
}
