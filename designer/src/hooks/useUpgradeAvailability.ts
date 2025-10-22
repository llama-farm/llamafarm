import { useEffect, useMemo, useState } from 'react'
import {
  compareSemver,
  getCurrentVersion,
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

export function useUpgradeAvailability() {
  const [currentVersion] = useState<string>(() => getCurrentVersion())
  const [{ info, checkedAt }, setCache] = useState(() =>
    getStoredLatestRelease()
  )
  const [isLoading, setIsLoading] = useState(false)
  const [dismissCounter, setDismissCounter] = useState(0)

  useEffect(() => {
    const abort = new AbortController()
    const run = async () => {
      if (!shouldCheck(checkedAt, TWELVE_HOURS_MS)) return
      setIsLoading(true)
      try {
        const res = await getVersionCheck(abort.signal)
        const latestVersion = res?.latest_version || ''
        const htmlUrl = res?.release_url || getGithubReleasesUrl()
        const publishedAt = res?.published_at
        if (latestVersion) {
          const mapped = { latestVersion, htmlUrl, publishedAt }
          storeLatestRelease(mapped)
          setCache({ info: mapped, checkedAt: Date.now() })
        }
      } catch (error) {
        console.error('Failed to fetch version info:', error)
      }
      setIsLoading(false)
    }
    run()
    return () => abort.abort()
  }, [checkedAt])

  const normalizedCurrent = normalizeVersion(currentVersion)
  const latestVersion = useMemo(
    () => normalizeVersion(info?.latestVersion),
    [info?.latestVersion]
  )
  const upgradeAvailable = useMemo(() => {
    if (!latestVersion) return false
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
