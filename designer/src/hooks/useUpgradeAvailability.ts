import { useEffect, useMemo, useState } from 'react'
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
  storeCurrentVersion,
  getStoredCurrentVersion,
} from '@/utils/versionUtils'
import { getVersionCheck } from '@/api/systemService'

const TWELVE_HOURS_MS = 12 * 60 * 60 * 1000

export function useUpgradeAvailability() {
  const [currentVersion, setCurrentVersion] = useState<string | null>(() => {
    const stored = getStoredCurrentVersion()
    // Ignore invalid/old hardcoded versions - always fetch fresh from server
    if (stored === '0.0.0' || stored === '0.1.0') {
      return null
    }
    return stored
  })
  const [{ info, checkedAt }, setCache] = useState(() =>
    getStoredLatestRelease()
  )
  const [isLoading, setIsLoading] = useState(false)
  const [dismissCounter, setDismissCounter] = useState(0)

  useEffect(() => {
    const abort = new AbortController()
    const run = async () => {
      // Always fetch current version, but only check for latest version if 12 hours have passed
      const shouldCheckLatest = shouldCheck(checkedAt, TWELVE_HOURS_MS)
      setIsLoading(true)
      try {
        const res = await getVersionCheck(abort.signal)
        const serverCurrentVersion = res?.current_version
        // Always update current version from server (no caching)
        // Ignore invalid hardcoded version "0.1.0"
        if (serverCurrentVersion && serverCurrentVersion !== '0.1.0') {
          setCurrentVersion(serverCurrentVersion)
          storeCurrentVersion(serverCurrentVersion)
        }
        // Only update latest version cache if enough time has passed
        if (shouldCheckLatest) {
          const latestVersion = res?.latest_version || ''
          const htmlUrl = res?.release_url || getGithubReleasesUrl()
          const publishedAt = res?.published_at
          if (latestVersion) {
            const mapped = { latestVersion, htmlUrl, publishedAt }
            storeLatestRelease(mapped)
            setCache({ info: mapped, checkedAt: Date.now() })
          }
        }
      } catch (error) {
        console.error('Failed to fetch version info:', error)
      }
      setIsLoading(false)
    }
    run()
    return () => abort.abort()
  }, [checkedAt])

  const normalizedCurrent = currentVersion ? normalizeVersion(currentVersion) : null
  const latestVersion = useMemo(
    () => normalizeVersion(info?.latestVersion),
    [info?.latestVersion]
  )
  const upgradeAvailable = useMemo(() => {
    if (!latestVersion || !normalizedCurrent) return false
    // Never show upgrade banner in dev mode
    if (currentVersion === 'dev') return false
    return compareSemver(latestVersion, normalizedCurrent) > 0
  }, [latestVersion, normalizedCurrent, currentVersion])

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
      // Ignore invalid hardcoded version "0.1.0"
      if (serverCurrentVersion && serverCurrentVersion !== '0.1.0') {
        setCurrentVersion(serverCurrentVersion)
        storeCurrentVersion(serverCurrentVersion)
      }
      if (latestVersion) {
        const mapped = { latestVersion, htmlUrl, publishedAt }
        storeLatestRelease(mapped)
        setCache({ info: mapped, checkedAt: Date.now() })
      }
    } catch (error) {
      console.error('Failed to fetch version info:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return {
    isLoading,
    currentVersion: normalizedCurrent,
    latestVersion,
    upgradeAvailable,
    releasesUrl,
    isDismissedFor,
    dismiss,
    refreshLatest,
    // expose to allow optional subscriptions; not used outside
    _dismissCounter: dismissCounter,
  }
}
