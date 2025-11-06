import { useState, useRef, useCallback } from 'react'
import modelService from '../api/modelService'
import type { DownloadEvent, DownloadProgress } from '../types/model'

interface FileProgress {
  name: string
  downloaded: number
  total: number
  completed: boolean
}

export function useModelDownload() {
  const [isDownloading, setIsDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const filesRef = useRef<Map<string, FileProgress>>(new Map())
  const startTimeRef = useRef<number>(0)
  const lastUpdateTimeRef = useRef<number>(0)
  const lastBytesRef = useRef<number>(0)
  const speedSamplesRef = useRef<number[]>([])

  const calculateSpeed = (currentBytes: number, _elapsedSeconds: number): number => {
    const now = Date.now()
    const timeSinceLastUpdate = (now - lastUpdateTimeRef.current) / 1000

    if (timeSinceLastUpdate < 0.1) {
      // Too soon, use previous speed estimate
      return speedSamplesRef.current[speedSamplesRef.current.length - 1] || 0
    }

    const bytesSinceLastUpdate = currentBytes - lastBytesRef.current
    const instantSpeed = bytesSinceLastUpdate / timeSinceLastUpdate

    // Keep last 10 samples for smoothing
    speedSamplesRef.current.push(instantSpeed)
    if (speedSamplesRef.current.length > 10) {
      speedSamplesRef.current.shift()
    }

    lastUpdateTimeRef.current = now
    lastBytesRef.current = currentBytes

    // Return average of recent samples
    return (
      speedSamplesRef.current.reduce((sum, s) => sum + s, 0) /
      speedSamplesRef.current.length
    )
  }

  const updateProgress = useCallback(() => {
    const files = Array.from(filesRef.current.values())
    if (files.length === 0) return

    const filesCompleted = files.filter(f => f.completed).length
    const totalFiles = files.length

    const overallDownloaded = files.reduce((sum, f) => sum + f.downloaded, 0)
    const overallTotal = files.reduce((sum, f) => sum + f.total, 0)
    const overallProgress =
      overallTotal > 0 ? Math.round((overallDownloaded / overallTotal) * 100) : 0

    const currentFile = files.find(f => !f.completed) || files[files.length - 1]
    const currentFileProgress =
      currentFile && currentFile.total > 0
        ? Math.round((currentFile.downloaded / currentFile.total) * 100)
        : 0

    const elapsedTime = (Date.now() - startTimeRef.current) / 1000
    const downloadSpeed = calculateSpeed(overallDownloaded, elapsedTime)

    const remainingBytes = overallTotal - overallDownloaded
    const estimatedTimeRemaining =
      downloadSpeed > 0 ? remainingBytes / downloadSpeed : 0

    setProgress({
      currentFile: currentFile?.name || '',
      currentFileProgress,
      currentFileDownloaded: currentFile?.downloaded || 0,
      currentFileTotal: currentFile?.total || 0,
      filesCompleted,
      totalFiles,
      overallDownloaded,
      overallTotal,
      overallProgress,
      downloadSpeed,
      estimatedTimeRemaining,
      elapsedTime,
    })
  }, [])

  const downloadModel = useCallback(
    async (modelName: string, provider: string = 'universal') => {
      setIsDownloading(true)
      setError(null)
      setProgress(null)
      filesRef.current.clear()
      speedSamplesRef.current = []
      startTimeRef.current = Date.now()
      lastUpdateTimeRef.current = Date.now()
      lastBytesRef.current = 0

      try {
        for await (const event of modelService.downloadModel({
          model_name: modelName,
          provider,
        })) {
          handleEvent(event)
        }

        setIsDownloading(false)
        return true
      } catch (err: any) {
        setError(err.message || 'Download failed')
        setIsDownloading(false)
        return false
      }
    },
    [updateProgress]
  )

  const handleEvent = (event: DownloadEvent) => {
    switch (event.event) {
      case 'start': {
        const fileName = event.desc
        filesRef.current.set(fileName, {
          name: fileName,
          downloaded: event.n,
          total: event.total || 0,
          completed: false,
        })
        updateProgress()
        break
      }

      case 'progress': {
        const fileName = event.desc
        const file = filesRef.current.get(fileName)
        if (file) {
          file.downloaded = event.n
          file.total = event.total || file.total
          filesRef.current.set(fileName, file)
          updateProgress()
        }
        break
      }

      case 'end': {
        const fileName = event.desc
        const file = filesRef.current.get(fileName)
        if (file) {
          file.downloaded = event.total || event.n
          file.total = event.total || file.total
          file.completed = true
          filesRef.current.set(fileName, file)
          updateProgress()
        }
        break
      }

      case 'done': {
        // Download complete
        updateProgress()
        break
      }

      case 'error': {
        setError(event.message)
        setIsDownloading(false)
        break
      }
    }
  }

  const reset = useCallback(() => {
    setIsDownloading(false)
    setProgress(null)
    setError(null)
    filesRef.current.clear()
    speedSamplesRef.current = []
  }, [])

  return {
    isDownloading,
    progress,
    error,
    downloadModel,
    reset,
  }
}
