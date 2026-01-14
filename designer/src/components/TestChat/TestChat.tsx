import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { ChatboxMessage } from '../../types/chatbox'
import { Badge } from '../ui/badge'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProjectChatParams } from '../../hooks/useProjectChat'
import { useStreamingChatCompletionMessage } from '../../hooks/useChatCompletions'
import { useProjectChatStreamingSession } from '../../hooks/useProjectChatSession'
import { useProjectSession } from '../../hooks/useProjectSession'
import { useChatbox } from '../../hooks/useChatbox'
import { ChatStreamChunk } from '../../types/chat'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useProjectModels } from '../../hooks/useProjectModels'
import { useProject } from '../../hooks/useProjects'
import { useListAnomalyModels, useScoreAnomaly, useLoadAnomaly, useListClassifierModels, usePredictClassifier, useLoadClassifier, useScanDocument, useCreateEmbeddings, useRerankDocuments } from '../../hooks/useMLModels'
import { Selector } from '../ui/selector'
import {
  DOCUMENT_SCANNING_BACKEND_DISPLAY,
  DOCUMENT_SCANNING_LANGUAGES,
  type DocumentScanningBackend,
  type DocumentScanningResultItem,
  type DocumentScanningHistoryEntry,
  type EncoderSubMode,
  type EncoderHistoryEntry,
  type RerankResult,
  COMMON_EMBEDDING_MODELS,
  COMMON_RERANKING_MODELS,
  EMBEDDING_SAMPLES,
  RERANKING_SAMPLES,
} from '../../types/ml'

export interface TestChatProps {
  // Mode selection
  modelType: 'inference' | 'anomaly' | 'classifier' | 'document_scanning' | 'encoder'
  onModelTypeChange: (type: 'inference' | 'anomaly' | 'classifier' | 'document_scanning' | 'encoder') => void
  // Existing props
  showReferences: boolean
  allowRanking: boolean
  useTestData?: boolean
  showPrompts?: boolean
  showThinking?: boolean
  showGenSettings?: boolean
  genSettings?: {
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
  }
  ragEnabled?: boolean
  ragTopK?: number
  ragScoreThreshold?: number
  focusInput?: boolean
}

const containerClasses =
  // Match page background with clear outlines
  'w-full h-full flex flex-col rounded-xl border border-border bg-background text-foreground'

const inputContainerClasses =
  'flex flex-col gap-2 p-3 md:p-4 bg-background/60 border-t border-border rounded-b-xl'

const textareaClasses =
  'w-full h-auto min-h-[3rem] md:min-h-[3.5rem] resize-none bg-transparent border-none placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm md:text-base leading-relaxed overflow-y-auto text-foreground placeholder-foreground/60'

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-teal-500/20 border border-teal-500/30">
          <FontIcon type="test" className="w-5 h-5 text-teal-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          Start testing your model
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          Send a message to evaluate responses and run diagnostics.
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Tip: Press Enter to send
        </div>
      </div>
    </div>
  )
}

// Anomaly Detection Empty States
function AnomalyEmptyState({
  hasModels,
  onCreateModel,
}: {
  hasModels: boolean
  onCreateModel: () => void
}) {
  if (!hasModels) {
    // No Models Empty State
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            No anomaly models yet
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Create an anomaly detection model to start testing
          </div>
          <button
            onClick={onCreateModel}
            className="mt-4 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90"
          >
            Create Anomaly Model
          </button>
        </div>
      </div>
    )
  }

  // Has Models Empty State
  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-teal-500/20 border border-teal-500/30">
          <FontIcon type="test" className="w-5 h-5 text-teal-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          Start testing anomaly detection
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          Paste or enter data to check for anomalies
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Tip: Paste a table row or comma-separated values
        </div>
      </div>
    </div>
  )
}

// Classifier Empty States
function ClassifierEmptyState({
  hasModels,
  onCreateModel,
}: {
  hasModels: boolean
  onCreateModel: () => void
}) {
  if (!hasModels) {
    // No Models Empty State
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-purple-500/20 border border-purple-500/30">
            <FontIcon type="prompt" className="w-5 h-5 text-purple-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            No classifier models yet
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Create a classifier model to start testing
          </div>
          <button
            onClick={onCreateModel}
            className="mt-4 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90"
          >
            Create Classifier Model
          </button>
        </div>
      </div>
    )
  }

  // Has Models Empty State
  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-purple-500/20 border border-purple-500/30">
          <FontIcon type="prompt" className="w-5 h-5 text-purple-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          Start testing your classifier
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          Enter text to see predicted labels and confidence scores
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Tip: Press Enter to classify
        </div>
      </div>
    </div>
  )
}

// Inference No Models Empty State
function InferenceNoModelsState({
  onAddModel,
}: {
  onAddModel: () => void
}) {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
          <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          No inference models configured
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          Add an inference model to your project to start chatting
        </div>
        <button
          onClick={onAddModel}
          className="mt-4 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90"
        >
          Add Inference Model
        </button>
      </div>
    </div>
  )
}

// Classifier Result Display
function ClassifierResultDisplay({
  result,
  error,
  isLoading,
  inputText,
}: {
  result: {
    predictions: Array<{
      label: string
      score: number
    }>
    isMultiLabel: boolean
    threshold: number
  } | null
  error: string | null
  isLoading: boolean
  inputText: string
}) {
  const [detailsOpen, setDetailsOpen] = useState(true)
  const [copied, setCopied] = useState(false)

  const handleCopyInput = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(inputText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard access denied or not available
    }
  }, [inputText])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="mt-3 text-sm text-muted-foreground">
            Classifying...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center w-full pt-4 pb-4">
        <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            Classification Error
          </div>
          <div className="mt-2 text-sm text-amber-400">
            {error}
          </div>
        </div>
      </div>
    )
  }

  if (!result || result.predictions.length === 0) {
    return null
  }

  const topPrediction = result.predictions[0]
  const otherPredictions = result.predictions.slice(1)
  const isLowConfidence = topPrediction.score < 0.5

  // For multi-label: show labels above threshold
  const aboveThreshold = result.predictions.filter(p => p.score >= result.threshold)
  const belowThreshold = result.predictions.filter(p => p.score < result.threshold)

  return (
    <div className="flex flex-col items-center w-full pt-4 pb-4">
      <div className="w-full max-w-md">
        {result.isMultiLabel ? (
          // Multi-label display
          <>
            {/* Applied Labels */}
            {aboveThreshold.length > 0 && (
              <div className="mb-4">
                <div className="text-xs text-muted-foreground mb-2">Applied Labels</div>
                <div className="flex flex-wrap gap-2">
                  {aboveThreshold.map((pred, idx) => (
                    <Badge
                      key={idx}
                      className="bg-primary/20 text-primary border border-primary/30 px-3 py-1"
                    >
                      {pred.label}
                      <span className="ml-2 text-xs opacity-70">
                        {(pred.score * 100).toFixed(1)}%
                      </span>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Below threshold */}
            {belowThreshold.length > 0 && (
              <div className="mt-4">
                <div className="text-xs text-muted-foreground mb-2">Below threshold</div>
                <div className="space-y-2">
                  {belowThreshold.map((pred, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{pred.label}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-muted-foreground/40 rounded-full"
                            style={{ width: `${pred.score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground w-12 text-right">
                          {(pred.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          // Single-label display
          <>
            {/* Primary Prediction */}
            <div className="flex flex-col items-center gap-3 py-6">
              <Badge
                className={`text-lg px-4 py-2 ${
                  isLowConfidence
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    : 'bg-primary/20 text-primary border border-primary/30'
                }`}
              >
                {topPrediction.label}
              </Badge>

              {/* Confidence Score */}
              <div className="w-full max-w-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">Confidence</span>
                  <span className="text-sm font-medium">
                    {(topPrediction.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      isLowConfidence ? 'bg-amber-500' : 'bg-primary'
                    }`}
                    style={{ width: `${topPrediction.score * 100}%` }}
                  />
                </div>
              </div>

              {/* Low confidence indicator */}
              {isLowConfidence && (
                <Badge className="bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs">
                  Low confidence
                </Badge>
              )}
            </div>

            {/* Other possibilities */}
            {otherPredictions.length > 0 && (
              <div className="mt-4 border-t border-border pt-4">
                <div className="text-xs text-muted-foreground mb-3">Other possibilities</div>
                <div className="space-y-2">
                  {otherPredictions.map((pred, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{pred.label}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-muted-foreground/40 rounded-full"
                            style={{ width: `${pred.score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground w-12 text-right">
                          {(pred.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Collapsible Details */}
        <div className="rounded-md border border-border mt-4">
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className="w-full flex items-center justify-between px-3 py-2 text-sm text-muted-foreground hover:bg-accent/40"
          >
            <span>Details</span>
            <FontIcon
              type={detailsOpen ? 'chevron-up' : 'chevron-down'}
              className="w-4 h-4"
            />
          </button>
          {detailsOpen && (
            <div className="px-3 py-2 border-t border-border">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">
                  Input text:
                </span>
                <button
                  onClick={handleCopyInput}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  title="Copy input text"
                >
                  <FontIcon type={copied ? 'checkmark-filled' : 'copy'} className="w-3.5 h-3.5" />
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <div className="text-sm break-words">
                {inputText}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Anomaly Result Display
function AnomalyResultDisplay({
  result,
  error,
  isLoading,
}: {
  result: {
    score: number
    isAnomaly: boolean
    threshold: number
    parsedInput: string[]
  } | null
  error: string | null
  isLoading: boolean
}) {
  const [detailsOpen, setDetailsOpen] = useState(true)
  const [copied, setCopied] = useState(false)

  const handleCopyInput = useCallback(async () => {
    if (!result) return
    try {
      await navigator.clipboard.writeText(result.parsedInput.join(', '))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard access denied or not available
    }
  }, [result])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="mt-3 text-sm text-muted-foreground">
            Analyzing data...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center w-full pt-4 pb-4">
        <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            Detection Error
          </div>
          <div className="mt-2 text-sm text-amber-400">
            {error}
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return null
  }

  return (
    <div className="flex flex-col items-center w-full pt-4 pb-4">
      <div className="w-full max-w-md">
        {/* Score Display - Large & Prominent */}
        <div className="flex flex-col items-center gap-3 py-6">
          <div className="text-5xl font-bold tabular-nums">
            {result.score.toFixed(3)}
          </div>

          {/* Status Badge */}
          <Badge
            className={
              result.isAnomaly
                ? 'bg-destructive/20 text-destructive border border-destructive/30'
                : 'bg-primary/20 text-primary border border-primary/30'
            }
          >
            {result.isAnomaly ? 'Anomaly Detected' : 'Normal'}
          </Badge>

          {/* Threshold Reference */}
          <div className="text-xs text-muted-foreground">
            Threshold: {result.threshold.toFixed(2)}
          </div>
        </div>

        {/* Collapsible Details */}
        <div className="rounded-md border border-border mt-4">
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className="w-full flex items-center justify-between px-3 py-2 text-sm text-muted-foreground hover:bg-accent/40"
          >
            <span>Details</span>
            <FontIcon
              type={detailsOpen ? 'chevron-up' : 'chevron-down'}
              className="w-4 h-4"
            />
          </button>
          {detailsOpen && (
            <div className="px-3 py-2 border-t border-border">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">
                  Parsed input values:
                </span>
                <button
                  onClick={handleCopyInput}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  title="Copy input values"
                >
                  <FontIcon type={copied ? 'checkmark-filled' : 'copy'} className="w-3.5 h-3.5" />
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <div className="font-mono text-sm break-all">
                {result.parsedInput.join(', ')}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Sidebar history item for anomaly tests
function AnomalyHistoryItem({
  item,
  onRerun,
}: {
  item: {
    id: string
    timestamp: Date
    score: number
    isAnomaly: boolean
    parsedInput: string[]
    modelName: string
    error?: string
  }
  onRerun: (input: string) => void
}) {
  const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const isError = !!item.error

  return (
    <button
      onClick={() => onRerun(item.parsedInput.join(', '))}
      className="w-full text-left px-2 py-1.5 rounded-md border border-border/50 hover:bg-muted/40 hover:border-border transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Badge
            className={`text-[10px] px-1.5 py-0 ${
              isError
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                : item.isAnomaly
                  ? 'bg-destructive/20 text-destructive border-destructive/30'
                  : 'bg-primary/20 text-primary border-primary/30'
            }`}
          >
            {isError ? 'Error' : item.isAnomaly ? 'Anomaly' : 'Normal'}
          </Badge>
          {!isError && (
            <span
              className={`font-mono text-xs tabular-nums ${
                item.isAnomaly ? 'text-destructive' : 'text-primary'
              }`}
            >
              {item.score.toFixed(3)}
            </span>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground">{timeStr}</span>
      </div>
      <div
        className="text-[10px] text-muted-foreground truncate mt-1"
        title={isError ? item.error : item.parsedInput.join(', ')}
      >
        {item.parsedInput.join(', ')}
      </div>
    </button>
  )
}

// Sidebar history item for classifier tests
function ClassifierHistoryItem({
  item,
  onRerun,
}: {
  item: {
    id: string
    timestamp: Date
    inputText: string
    topLabel: string
    topScore: number
    modelName: string
    error?: string
  }
  onRerun: (input: string) => void
}) {
  const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const isError = !!item.error
  const isLowConfidence = !isError && item.topScore < 0.5

  return (
    <button
      onClick={() => onRerun(item.inputText)}
      className="w-full text-left px-2 py-1.5 rounded-md border border-border/50 hover:bg-muted/40 hover:border-border transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Badge
            className={`text-[10px] px-1.5 py-0 ${
              isError
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                : isLowConfidence
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                  : 'bg-primary/20 text-primary border-primary/30'
            }`}
          >
            {isError ? 'Error' : item.topLabel}
          </Badge>
          {!isError && (
            <span className="text-xs tabular-nums">
              {(item.topScore * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground">{timeStr}</span>
      </div>
      <div
        className="text-[10px] text-muted-foreground truncate mt-1"
        title={isError ? item.error : item.inputText}
      >
        {item.inputText}
      </div>
    </button>
  )
}

// Document Scanning Result Display
function DocumentScanningResultDisplay({
  results,
  error,
  isLoading,
  fileName,
}: {
  results: DocumentScanningResultItem[] | null
  error: string | null
  isLoading: boolean
  fileName: string
}) {
  const [copied, setCopied] = useState(false)
  const [selectedPage, setSelectedPage] = useState(0)

  const handleCopyText = useCallback(() => {
    if (!results) return
    const fullText = results.map(r => r.text).join('\n\n--- Page Break ---\n\n')
    navigator.clipboard.writeText(fullText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [results])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="mt-3 text-sm text-muted-foreground">
            Extracting text...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center w-full pt-4 pb-4">
        <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">
            Scanning Error
          </div>
          <div className="mt-2 text-sm text-amber-400">
            {error}
          </div>
        </div>
      </div>
    )
  }

  if (!results || results.length === 0) {
    return null
  }

  const currentResult = results[selectedPage] || results[0]
  const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length

  return (
    <div className="flex flex-col h-full p-4">
      {/* Header with file info and actions */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-border">
        <div className="flex items-center gap-3">
          <FontIcon type="data" className="w-5 h-5 text-sky-400" />
          <div>
            <div className="text-sm font-medium">{fileName}</div>
            <div className="text-xs text-muted-foreground">
              {results.length} page{results.length > 1 ? 's' : ''} • {(avgConfidence * 100).toFixed(1)}% avg confidence
            </div>
          </div>
        </div>
        <button
          onClick={handleCopyText}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-secondary/80 hover:bg-secondary"
        >
          <FontIcon type={copied ? 'checkmark-filled' : 'copy'} className="w-4 h-4" />
          {copied ? 'Copied!' : 'Copy All'}
        </button>
      </div>

      {/* Page selector for multi-page documents */}
      {results.length > 1 && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-xs text-muted-foreground">Page:</span>
          <div className="flex gap-1 flex-wrap">
            {results.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedPage(idx)}
                className={`px-2 py-0.5 text-xs rounded ${
                  idx === selectedPage
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted hover:bg-muted/80'
                }`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Text content */}
      <div className="flex-1 overflow-y-auto">
        <div className="rounded-lg border border-border bg-muted/30 p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">
              Page {selectedPage + 1} • {(currentResult.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>
          <div className="whitespace-pre-wrap text-sm leading-relaxed font-mono">
            {currentResult.text || '(No text detected)'}
          </div>
        </div>
      </div>
    </div>
  )
}

// Helper to validate scan files
function isValidScanFile(file: File): boolean {
  const validExtensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif']
  const ext = '.' + file.name.split('.').pop()?.toLowerCase()
  return validExtensions.includes(ext)
}

// Document Scanning Container with always-active drop zone
function DocumentScanningContainer({
  onFileSelect,
  disabled,
  children,
  onInputRefReady,
}: {
  onFileSelect: (file: File) => void
  disabled: boolean
  children: React.ReactNode
  /** Callback to expose the file input trigger function to parent */
  onInputRefReady?: (triggerBrowse: () => void) => void
}) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Expose browse trigger to parent on mount
  useEffect(() => {
    if (onInputRefReady) {
      onInputRefReady(() => inputRef.current?.click())
    }
  }, [onInputRefReady])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    // Only set to false if leaving the container entirely
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (disabled) return

    const file = e.dataTransfer.files[0]
    if (file && isValidScanFile(file)) {
      onFileSelect(file)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && isValidScanFile(file)) {
      onFileSelect(file)
    }
    // Reset input so same file can be selected again
    e.target.value = ''
  }

  return (
    <div
      className="relative flex-1 overflow-y-auto flex flex-col"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input for click-to-browse */}
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      {/* Main content (results, loading, or empty state) */}
      {children}

      {/* Drop overlay - appears when dragging */}
      {isDragging && !disabled && (
        <div className="absolute inset-0 z-10 bg-background/90 backdrop-blur-sm flex items-center justify-center">
          <div className="rounded-xl border-2 border-dashed border-primary bg-primary/10 p-8 text-center">
            <FontIcon type="upload" className="w-12 h-12 text-primary mx-auto mb-3" />
            <div className="text-sm font-medium text-foreground">Drop file to scan</div>
            <div className="text-xs text-muted-foreground mt-1">
              PDF, PNG, JPG, GIF, WebP, BMP, TIFF
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Document Scanning Empty State (shown when no file selected)
function DocumentScanningEmptyState({
  onBrowseClick,
  disabled,
}: {
  onBrowseClick: () => void
  disabled: boolean
}) {
  return (
    <div className="flex-1 p-4 flex items-center justify-center">
      <div
        onClick={() => !disabled && onBrowseClick()}
        className={`
          w-full max-w-md rounded-xl border-2 border-dashed transition-colors cursor-pointer
          flex flex-col items-center justify-center gap-3 p-8
          border-border hover:border-primary/50 hover:bg-muted/20
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <FontIcon type="upload" className="w-10 h-10 text-muted-foreground" />
        <div className="text-center">
          <div className="text-sm font-medium">Drop file here or click to browse</div>
          <div className="text-xs text-muted-foreground mt-1">
            Supports PDF, PNG, JPG, GIF, WebP, BMP, TIFF
          </div>
        </div>
      </div>
    </div>
  )
}

// Sidebar history item for document scanning
function DocumentScanningHistoryItem({
  item,
  onSelect,
}: {
  item: DocumentScanningHistoryEntry
  onSelect: () => void
}) {
  const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const isError = !!item.error

  return (
    <button
      onClick={onSelect}
      className="w-full text-left px-2 py-1.5 rounded-md border border-border/50 hover:bg-muted/40 hover:border-border transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FontIcon type="data" className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-xs font-medium truncate max-w-[100px]">{item.fileName}</span>
        </div>
        <span className="text-[10px] text-muted-foreground">{timeStr}</span>
      </div>
      <div className="flex items-center gap-2 mt-1">
        <Badge className={`text-[10px] px-1.5 py-0 ${
          isError
            ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-sky-500/20 text-sky-400 border-sky-500/30'
        }`}>
          {isError ? 'Error' : `${item.pageCount} pg`}
        </Badge>
        {!isError && (
          <span className="text-[10px] text-muted-foreground">
            {(item.avgConfidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div className="text-[10px] text-muted-foreground truncate mt-1">
        {isError ? item.error : item.previewText}
      </div>
    </button>
  )
}

// Encoder Empty State
function EncoderEmptyState({ subMode }: { subMode: EncoderSubMode }) {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-sky-500/20 border border-sky-500/40">
          <FontIcon type="data" className="w-5 h-5 text-sky-400" />
        </div>
        <div className="text-lg font-medium text-foreground">
          {subMode === 'embedding'
            ? 'Test Embedding Similarity'
            : 'Test Document Reranking'}
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          {subMode === 'embedding'
            ? 'Enter two texts to compare their semantic similarity'
            : 'Enter a query and documents to see relevance rankings'}
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          {subMode === 'embedding'
            ? 'Tip: Press Cmd+Enter to compare'
            : 'Tip: Press Cmd+Enter to rank'}
        </div>
      </div>
    </div>
  )
}

// Embedding Similarity Result Display
function EmbeddingSimilarityDisplay({
  result,
  error,
  isLoading,
  onCompareAnother,
}: {
  result: {
    texts: string[]
    similarity: number
  } | null
  error: string | null
  isLoading: boolean
  onCompareAnother: () => void
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="mt-3 text-sm text-muted-foreground">
            Comparing texts...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center w-full pt-4 pb-4">
        <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">Embedding Error</div>
          <div className="mt-2 text-sm text-amber-400">{error}</div>
          <button
            onClick={onCompareAnother}
            className="mt-4 px-4 py-2 rounded-lg border border-input bg-background hover:bg-accent text-sm"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!result) return null

  const { texts, similarity } = result

  // Get similarity label and color
  const getSimilarityLabel = (score: number) => {
    if (score >= 0.9) return { label: 'Very High Similarity', color: 'text-green-500' }
    if (score >= 0.7) return { label: 'High Similarity', color: 'text-green-400' }
    if (score >= 0.5) return { label: 'Moderate Similarity', color: 'text-yellow-500' }
    if (score >= 0.3) return { label: 'Low Similarity', color: 'text-orange-500' }
    return { label: 'Very Low Similarity', color: 'text-red-500' }
  }

  const { label, color } = getSimilarityLabel(similarity)

  // Get progress bar color
  const getProgressColor = (score: number) => {
    if (score >= 0.7) return 'bg-green-500'
    if (score >= 0.5) return 'bg-yellow-500'
    if (score >= 0.3) return 'bg-orange-500'
    return 'bg-red-500'
  }

  return (
    <div className="flex flex-col items-center p-6 space-y-6">
      {/* Score display */}
      <div className="text-center">
        <div className="text-5xl font-bold tabular-nums">{similarity.toFixed(3)}</div>
        <div className={`text-lg font-medium mt-1 ${color}`}>{label}</div>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md">
        <div className="w-full bg-muted rounded-full h-3">
          <div
            className={`${getProgressColor(similarity)} h-full rounded-full transition-all`}
            style={{ width: `${similarity * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>0.0</span>
          <span>1.0</span>
        </div>
      </div>

      {/* Text previews */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="text-xs font-medium text-muted-foreground mb-1">Text A</div>
          <div className="text-sm">{texts[0].length > 100 ? texts[0].slice(0, 100) + '...' : texts[0]}</div>
        </div>
        <div className="rounded-lg border border-border bg-card/50 p-3">
          <div className="text-xs font-medium text-muted-foreground mb-1">Text B</div>
          <div className="text-sm">{texts[1].length > 100 ? texts[1].slice(0, 100) + '...' : texts[1]}</div>
        </div>
      </div>

      {/* Compare another button */}
      <button
        onClick={onCompareAnother}
        className="px-4 py-2 rounded-lg border border-input bg-background hover:bg-accent text-sm"
      >
        Compare Another
      </button>
    </div>
  )
}

// Rerank Result Display
function RerankResultDisplay({
  result,
  error,
  isLoading,
  query,
  onRankAgain,
}: {
  result: RerankResult[] | null
  error: string | null
  isLoading: boolean
  query: string
  onRankAgain: () => void
}) {
  const [expandedDocs, setExpandedDocs] = useState<Set<number>>(new Set())

  const toggleExpand = (index: number) => {
    setExpandedDocs(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-center px-6 py-10">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="mt-3 text-sm text-muted-foreground">
            Ranking documents...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center w-full pt-4 pb-4">
        <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
            <FontIcon type="alert-triangle" className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-lg font-medium text-foreground">Reranking Error</div>
          <div className="mt-2 text-sm text-amber-400">{error}</div>
          <button
            onClick={onRankAgain}
            className="mt-4 px-4 py-2 rounded-lg border border-input bg-background hover:bg-accent text-sm"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!result || result.length === 0) return null

  // Find max score for progress bar scaling
  const maxScore = Math.max(...result.map(r => r.relevance_score), 0.01)

  // Get progress bar color based on relative rank
  const getProgressColor = (rank: number, total: number) => {
    const position = rank / (total - 1 || 1)
    if (position <= 0.33) return 'bg-green-500'
    if (position <= 0.66) return 'bg-yellow-500'
    return 'bg-orange-500'
  }

  return (
    <div className="flex flex-col p-4 space-y-4">
      {/* Query display */}
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
        <div className="text-xs font-medium text-primary mb-1">Query:</div>
        <div className="text-sm">{query}</div>
      </div>

      {/* Ranked results */}
      <div className="text-sm font-medium">Ranked by Relevance:</div>
      <div className="space-y-2">
        {result.map((item, rank) => {
          const docText = item.document || ''
          const isLong = docText.length > 150
          const isExpanded = expandedDocs.has(item.index)
          const displayText = isLong && !isExpanded ? docText.slice(0, 150) + '...' : docText

          return (
            <div
              key={item.index}
              className="rounded-lg border border-border p-3"
            >
              <div className="flex items-start gap-3">
                {/* Rank badge */}
                <div className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
                  ${rank === 0 ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}
                `}>
                  #{rank + 1}
                </div>
                <div className="flex-1 min-w-0">
                  {/* Relevance score with progress bar */}
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-medium tabular-nums w-12">
                      {item.relevance_score.toFixed(2)}
                    </span>
                    <div className="flex-1 bg-muted rounded-full h-2">
                      <div
                        className={`${getProgressColor(rank, result.length)} h-full rounded-full transition-all`}
                        style={{ width: `${(item.relevance_score / maxScore) * 100}%` }}
                      />
                    </div>
                  </div>
                  {/* Document text */}
                  <div className="text-sm">{displayText}</div>
                  {isLong && (
                    <button
                      onClick={() => toggleExpand(item.index)}
                      className="text-xs text-primary hover:underline mt-1"
                    >
                      {isExpanded ? 'Show less' : 'Show more'}
                    </button>
                  )}
                  {/* Original position */}
                  <div className="mt-2 text-xs text-muted-foreground">
                    Originally: Document {item.index + 1}
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Rank again button */}
      <div className="flex justify-center pt-2">
        <button
          onClick={onRankAgain}
          className="px-4 py-2 rounded-lg border border-input bg-background hover:bg-accent text-sm"
        >
          Rank Again
        </button>
      </div>
    </div>
  )
}

// Encoder History Item
function EncoderHistoryItem({
  item,
  onRerun,
}: {
  item: EncoderHistoryEntry
  onRerun: () => void
}) {
  const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const isError = !!item.error

  return (
    <button
      onClick={onRerun}
      className="w-full text-left px-2 py-1.5 rounded-md border border-border/50 hover:bg-muted/40 hover:border-border transition-colors"
    >
      <div className="flex items-center justify-between">
        <Badge
          className={`text-[10px] px-1.5 py-0 ${
            isError
              ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
              : item.mode === 'embedding'
                ? 'bg-sky-500/20 text-sky-400 border-sky-500/40'
                : 'bg-violet-500/20 text-violet-400 border-violet-500/40'
          }`}
        >
          {isError ? 'Error' : item.mode === 'embedding' ? 'Embed' : 'Rerank'}
        </Badge>
        <span className="text-[10px] text-muted-foreground">{timeStr}</span>
      </div>
      <div className="text-[10px] text-muted-foreground truncate mt-1">
        {item.mode === 'embedding'
          ? `${item.texts?.length || 0} texts`
          : item.query?.substring(0, 40)}
      </div>
    </button>
  )
}

export default function TestChat({
  modelType,
  onModelTypeChange,
  showReferences,
  allowRanking,
  useTestData,
  showPrompts,
  showThinking,
  showGenSettings,
  genSettings,
  ragEnabled = true,
  ragTopK = 10,
  ragScoreThreshold = 0.7,
  focusInput = false,
}: TestChatProps) {
  const navigate = useNavigate()

  // Determine mock mode as early as possible
  const MOCK_MODE = Boolean(useTestData)
  // Get active project for project chat API
  const activeProject = useActiveProject()
  const chatParams = useProjectChatParams(activeProject)

  // ============================================================================
  // Anomaly Detection State & Hooks
  // ============================================================================

  // Fetch anomaly models (only when in anomaly mode)
  const { data: anomalyModelsData, isLoading: isLoadingAnomalyModels } =
    useListAnomalyModels({ enabled: modelType === 'anomaly' })
  const scoreAnomalyMutation = useScoreAnomaly()
  const loadAnomalyMutation = useLoadAnomaly()

  // Get all anomaly models sorted (most recent first by 'created' field)
  const allAnomalyModels = useMemo(() => {
    if (!anomalyModelsData?.data) return []
    return [...anomalyModelsData.data].sort((a, b) => {
      const dateA = a.created ? new Date(a.created).getTime() : 0
      const dateB = b.created ? new Date(b.created).getTime() : 0
      return dateB - dateA
    })
  }, [anomalyModelsData])

  // Get only the latest version per base_name (for dropdown)
  const sortedAnomalyModels = useMemo(() => {
    const latestByBaseName = new Map<string, typeof allAnomalyModels[0]>()
    for (const model of allAnomalyModels) {
      const baseName = model.base_name || model.name
      // Since allAnomalyModels is sorted newest first, first occurrence is the latest
      if (!latestByBaseName.has(baseName)) {
        latestByBaseName.set(baseName, model)
      }
    }
    return Array.from(latestByBaseName.values())
  }, [allAnomalyModels])

  // Selected anomaly model (stores the actual model name, not base_name)
  const [selectedAnomalyModel, setSelectedAnomalyModel] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('lf_test_anomalyModel')
  })

  // Get full info for the selected model
  const selectedAnomalyModelInfo = useMemo(() => {
    if (!selectedAnomalyModel) return null
    return allAnomalyModels.find(m => m.name === selectedAnomalyModel) || null
  }, [selectedAnomalyModel, allAnomalyModels])

  // Validate and auto-select anomaly model (similar to inference model logic)
  useEffect(() => {
    // Don't validate until models are loaded
    if (sortedAnomalyModels.length === 0) {
      return
    }

    // Build list of valid model names
    const validModelNames = sortedAnomalyModels.map(m => m.name)

    // If user has a selected model and it's valid, keep it
    if (selectedAnomalyModel && validModelNames.includes(selectedAnomalyModel)) {
      return
    }

    // Selected model is invalid or doesn't exist - fall back to first available
    if (validModelNames.length > 0) {
      setSelectedAnomalyModel(validModelNames[0])
    } else {
      setSelectedAnomalyModel(null)
    }
  }, [selectedAnomalyModel, sortedAnomalyModels])

  // Persist valid anomaly model selection to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && selectedAnomalyModel) {
      const validModelNames = sortedAnomalyModels.map(m => m.name)
      if (validModelNames.includes(selectedAnomalyModel)) {
        localStorage.setItem('lf_test_anomalyModel', selectedAnomalyModel)
      }
    }
  }, [selectedAnomalyModel, sortedAnomalyModels])

  // Anomaly input and result state
  const [anomalyInput, setAnomalyInput] = useState('')
  const [anomalyResult, setAnomalyResult] = useState<{
    score: number
    isAnomaly: boolean
    threshold: number
    parsedInput: string[]
  } | null>(null)
  const [anomalyError, setAnomalyError] = useState<string | null>(null)

  // Anomaly test history - persists between clears
  const [anomalyHistory, setAnomalyHistory] = useState<Array<{
    id: string
    timestamp: Date
    score: number
    isAnomaly: boolean
    threshold: number
    parsedInput: string[]
    modelName: string
    error?: string
  }>>([])
  const [showAnomalyHistory, setShowAnomalyHistory] = useState(true)
  const anomalyHistoryScrollRef = useRef<HTMLDivElement>(null)

  // ============================================================================
  // Classifier State & Hooks
  // ============================================================================

  // Fetch classifier models (only when in classifier mode)
  const { data: classifierModelsData, isLoading: isLoadingClassifierModels } =
    useListClassifierModels({ enabled: modelType === 'classifier' })
  const predictClassifierMutation = usePredictClassifier()
  const loadClassifierMutation = useLoadClassifier()

  // Get all classifier models sorted (most recent first by 'created' field)
  const allClassifierModels = useMemo(() => {
    if (!classifierModelsData?.data) return []
    return [...classifierModelsData.data].sort((a, b) => {
      const dateA = a.created ? new Date(a.created).getTime() : 0
      const dateB = b.created ? new Date(b.created).getTime() : 0
      return dateB - dateA
    })
  }, [classifierModelsData])

  // Get only the latest version per base_name (for dropdown)
  const sortedClassifierModels = useMemo(() => {
    const latestByBaseName = new Map<string, typeof allClassifierModels[0]>()
    for (const model of allClassifierModels) {
      const baseName = model.base_name || model.name
      // Since allClassifierModels is sorted newest first, first occurrence is the latest
      if (!latestByBaseName.has(baseName)) {
        latestByBaseName.set(baseName, model)
      }
    }
    return Array.from(latestByBaseName.values())
  }, [allClassifierModels])

  // Selected classifier model
  const [selectedClassifierModel, setSelectedClassifierModel] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('lf_test_classifierModel')
  })

  // Get full info for the selected classifier model
  const selectedClassifierModelInfo = useMemo(() => {
    if (!selectedClassifierModel) return null
    return allClassifierModels.find(m => m.name === selectedClassifierModel) || null
  }, [selectedClassifierModel, allClassifierModels])

  // Validate and auto-select classifier model
  useEffect(() => {
    if (sortedClassifierModels.length === 0) {
      return
    }

    const validModelNames = sortedClassifierModels.map(m => m.name)

    if (selectedClassifierModel && validModelNames.includes(selectedClassifierModel)) {
      return
    }

    if (validModelNames.length > 0) {
      setSelectedClassifierModel(validModelNames[0])
    } else {
      setSelectedClassifierModel(null)
    }
  }, [selectedClassifierModel, sortedClassifierModels])

  // Persist valid classifier model selection to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && selectedClassifierModel) {
      const validModelNames = sortedClassifierModels.map(m => m.name)
      if (validModelNames.includes(selectedClassifierModel)) {
        localStorage.setItem('lf_test_classifierModel', selectedClassifierModel)
      }
    }
  }, [selectedClassifierModel, sortedClassifierModels])

  // Classifier input and result state
  const [classifierInput, setClassifierInput] = useState('')
  const [classifierResult, setClassifierResult] = useState<{
    predictions: Array<{
      label: string
      score: number
    }>
    isMultiLabel: boolean
    threshold: number
  } | null>(null)
  const [classifierError, setClassifierError] = useState<string | null>(null)
  const [lastClassifierInput, setLastClassifierInput] = useState('')

  // Classifier test history - persists between clears
  const [classifierHistory, setClassifierHistory] = useState<Array<{
    id: string
    timestamp: Date
    inputText: string
    topLabel: string
    topScore: number
    predictions: Array<{ label: string; score: number }>
    modelName: string
    error?: string
  }>>([])
  const [showClassifierHistory, setShowClassifierHistory] = useState(true)
  const classifierHistoryScrollRef = useRef<HTMLDivElement>(null)

  // ============================================================================
  // Document Scanning State & Hooks
  // ============================================================================

  const scanDocumentMutation = useScanDocument()

  // Document scanning backend selection (persisted)
  const [selectedScanBackend, setSelectedScanBackend] = useState<DocumentScanningBackend>(() => {
    if (typeof window === 'undefined') return 'surya'
    const stored = localStorage.getItem('lf_test_scanBackend')
    if (stored && ['surya', 'easyocr', 'tesseract'].includes(stored)) {
      return stored as DocumentScanningBackend
    }
    return 'surya'
  })

  // Document scanning language selection (persisted)
  const [selectedScanLanguage, setSelectedScanLanguage] = useState<string>(() => {
    if (typeof window === 'undefined') return 'en'
    return localStorage.getItem('lf_test_scanLanguage') || 'en'
  })

  // Persist document scanning settings
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lf_test_scanBackend', selectedScanBackend)
    }
  }, [selectedScanBackend])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lf_test_scanLanguage', selectedScanLanguage)
    }
  }, [selectedScanLanguage])

  // Document scanning file state
  const [scanFile, setScanFile] = useState<File | null>(null)

  // Document scanning results state
  const [scanResults, setScanResults] = useState<DocumentScanningResultItem[] | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)

  // Document scanning history
  const [scanHistory, setScanHistory] = useState<DocumentScanningHistoryEntry[]>([])
  const [showScanHistory, setShowScanHistory] = useState(true)
  const scanHistoryScrollRef = useRef<HTMLDivElement>(null)
  // Ref to trigger file browse from empty state
  const triggerScanBrowseRef = useRef<(() => void) | null>(null)

  // Track if user has completed a scan before (for first-time message)
  const [hasScannedBefore, setHasScannedBefore] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem('lf_scan_completed') === 'true'
  })

  // ============================================================================
  // Encoder State & Hooks (Embeddings & Reranking)
  // ============================================================================

  const createEmbeddingsMutation = useCreateEmbeddings()
  const rerankMutation = useRerankDocuments()

  // Encoder sub-mode: embedding or reranking
  const [encoderSubMode, setEncoderSubMode] = useState<EncoderSubMode>(() => {
    if (typeof window === 'undefined') return 'embedding'
    const stored = localStorage.getItem('lf_test_encoderSubMode')
    return (stored === 'reranking') ? 'reranking' : 'embedding'
  })

  // Persist encoder sub-mode
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lf_test_encoderSubMode', encoderSubMode)
    }
  }, [encoderSubMode])

  // Selected embedding model (persisted)
  // Migrates old short names (e.g., "all-MiniLM-L6-v2") to full HuggingFace paths
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string>(() => {
    if (typeof window === 'undefined') return COMMON_EMBEDDING_MODELS[0].value
    const stored = localStorage.getItem('lf_test_embeddingModel')
    if (!stored) return COMMON_EMBEDDING_MODELS[0].value
    // Check if stored value is a valid full model path
    const isValid = COMMON_EMBEDDING_MODELS.some(m => m.value === stored)
    if (isValid) return stored
    // Try to find by label (short name) for backwards compatibility
    const byLabel = COMMON_EMBEDDING_MODELS.find(m => m.label === stored)
    if (byLabel) {
      localStorage.setItem('lf_test_embeddingModel', byLabel.value)
      return byLabel.value
    }
    return COMMON_EMBEDDING_MODELS[0].value
  })

  // Selected reranking model (persisted)
  // Migrates old short names to full HuggingFace paths
  const [selectedRerankingModel, setSelectedRerankingModel] = useState<string>(() => {
    if (typeof window === 'undefined') return COMMON_RERANKING_MODELS[0].value
    const stored = localStorage.getItem('lf_test_rerankingModel')
    if (!stored) return COMMON_RERANKING_MODELS[0].value
    // Check if stored value is a valid full model path
    const isValid = COMMON_RERANKING_MODELS.some(m => m.value === stored)
    if (isValid) return stored
    // Try to find by label (short name) for backwards compatibility
    const byLabel = COMMON_RERANKING_MODELS.find(m => m.label === stored)
    if (byLabel) {
      localStorage.setItem('lf_test_rerankingModel', byLabel.value)
      return byLabel.value
    }
    return COMMON_RERANKING_MODELS[0].value
  })

  // Persist model selections
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lf_test_embeddingModel', selectedEmbeddingModel)
    }
  }, [selectedEmbeddingModel])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('lf_test_rerankingModel', selectedRerankingModel)
    }
  }, [selectedRerankingModel])

  // Embedding mode state - two side-by-side inputs
  const [embeddingTextA, setEmbeddingTextA] = useState<string>('')
  const [embeddingTextB, setEmbeddingTextB] = useState<string>('')
  const [embeddingResult, setEmbeddingResult] = useState<{
    texts: string[]
    similarity: number
  } | null>(null)
  const [embeddingSampleIndex, setEmbeddingSampleIndex] = useState<number>(0)

  // Reranking mode state
  const [rerankQuery, setRerankQuery] = useState<string>('')
  const [rerankDocuments, setRerankDocuments] = useState<string[]>(['', '']) // Array of document texts
  const [rerankResult, setRerankResult] = useState<RerankResult[] | null>(null)
  const [rerankSampleIndex, setRerankSampleIndex] = useState<number>(0)

  // Shared state
  const [encoderError, setEncoderError] = useState<string | null>(null)

  // Encoder history
  const [encoderHistory, setEncoderHistory] = useState<EncoderHistoryEntry[]>([])
  const [showEncoderHistory, setShowEncoderHistory] = useState(true)
  const encoderHistoryScrollRef = useRef<HTMLDivElement>(null)
  const rerankDocumentsScrollRef = useRef<HTMLDivElement>(null)

  // Reranking right panel tab state
  const [rerankRightPanelTab, setRerankRightPanelTab] = useState<'inputs' | 'history'>('inputs')

  // Project chat streaming session management
  const projectChatStreamingSession = useProjectChatStreamingSession()

  // Project chat streaming message sending - using unified interface
  const projectChatStreamingMessage = useStreamingChatCompletionMessage()

  // Load available models for this project
  const { data: modelsData, isFetching: modelsLoading } = useProjectModels(
    chatParams?.namespace,
    chatParams?.projectId,
    !!chatParams
  )
  const apiModels = modelsData?.models || []
  const apiDefaultModel = apiModels.find(m => m.default)
  // Fallback to project runtime.model when models endpoint is missing/empty
  const { data: projectDetail } = useProject(
    chatParams?.namespace || '',
    chatParams?.projectId || '',
    !!chatParams
  )
  const runtimeCfg: any = (projectDetail as any)?.project?.config?.runtime || {}
  const cfgModels: Array<{ name: string; model: string }> = Array.isArray(
    runtimeCfg?.models
  )
    ? runtimeCfg.models
    : []
  const cfgDefaultName: string | undefined = runtimeCfg?.default_model
  // const cfgDefaultModelId: string | undefined = cfgModels.find(
  //   m => m.name === cfgDefaultName
  // )?.model

  // Unified view of models: prefer API; fallback to config-defined models
  const unifiedModels =
    apiModels.length > 0
      ? apiModels.map(m => ({
          name: (m as any).name ?? m.model,
          model: m.model,
          default: !!m.default,
        }))
      : cfgModels.map(m => ({
          name: m.name,
          model: m.model,
          default: m.name === cfgDefaultName,
        }))

  const defaultModel =
    apiModels.length > 0 ? apiDefaultModel : unifiedModels.find(m => m.default)
  const fallbackDefaultName: string | undefined = cfgDefaultName
  const [selectedModel, setSelectedModel] = useState<string | undefined>(() => {
    if (typeof window === 'undefined') return undefined

    // Load from localStorage but validate it exists in current config
    const savedModel = localStorage.getItem('lf_testchat_selected_model')
    if (savedModel) {
      // Check if this model exists in the unified models list
      // Note: unifiedModels might not be populated yet during initial render
      // so we'll validate again in the useEffect below
      return savedModel
    }
    return undefined
  })

  // Validate selected model and set default if needed
  useEffect(() => {
    // Don't validate until models are loaded - this prevents resetting
    // a valid localStorage selection before we know what models are available
    if (unifiedModels.length === 0) {
      return
    }

    // Build list of valid model names from current config
    const validModelNames = unifiedModels.map(m => m.name)

    // If user has a selected model and it's valid, keep it
    if (selectedModel && validModelNames.includes(selectedModel)) {
      return
    }

    // Selected model is invalid or doesn't exist - fall back to default
    const apiDefaultName = (defaultModel as any)?.name
    if (apiDefaultName && validModelNames.includes(apiDefaultName)) {
      setSelectedModel(apiDefaultName)
    } else if (
      fallbackDefaultName &&
      validModelNames.includes(fallbackDefaultName)
    ) {
      setSelectedModel(fallbackDefaultName)
    } else if (validModelNames.length > 0) {
      // Use first available model as last resort
      setSelectedModel(validModelNames[0])
    } else {
      // No valid models available, clear selection
      setSelectedModel(undefined)
    }
  }, [
    (defaultModel as any)?.name,
    fallbackDefaultName,
    selectedModel,
    unifiedModels,
  ])

  // Persist valid model selection to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && selectedModel) {
      // Only save if it's a valid model
      const validModelNames = unifiedModels.map(m => m.name)
      if (validModelNames.includes(selectedModel)) {
        localStorage.setItem('lf_testchat_selected_model', selectedModel)
      }
    }
  }, [selectedModel, unifiedModels])

  // Get current default database from config
  const getCurrentDatabase = useCallback(() => {
    try {
      const ragConfig = (projectDetail as any)?.project?.config?.rag
      // First try explicit default_database, then fall back to first database
      if (ragConfig?.default_database) {
        return ragConfig.default_database
      }
      // If no default set, use first database
      if (ragConfig?.databases && Array.isArray(ragConfig.databases) && ragConfig.databases.length > 0) {
        return ragConfig.databases[0]?.name || ''
      }
      return ''
    } catch {
      return ''
    }
  }, [projectDetail])

  // Get available databases from project config
  const availableDatabases = useMemo(() => {
    try {
      const ragConfig = (projectDetail as any)?.project?.config?.rag
      if (!ragConfig?.databases || !Array.isArray(ragConfig.databases)) {
        return []
      }

      return ragConfig.databases
        .filter((db: any) => db?.name) // Only include databases with names
        .map((db: any) => String(db.name))
    } catch {
      return []
    }
  }, [projectDetail])

  // Selected database state - persisted to localStorage for UI preference
  const [selectedDatabase, setSelectedDatabase] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('lf_testchat_selected_database')
  })

  // Get the current database value for rendering - prefer UI selection, fall back to config default
  const currentDatabase = selectedDatabase && availableDatabases.includes(selectedDatabase)
    ? selectedDatabase
    : getCurrentDatabase()

  // Validate and persist database selection
  useEffect(() => {
    // Don't validate until databases are loaded
    if (availableDatabases.length === 0) {
      return
    }

    // If user has a selected database and it's valid, keep it
    if (selectedDatabase && availableDatabases.includes(selectedDatabase)) {
      localStorage.setItem('lf_testchat_selected_database', selectedDatabase)
      return
    }

    // Selected database is invalid - clear it and fall back to config default
    if (selectedDatabase) {
      setSelectedDatabase(null)
      localStorage.removeItem('lf_testchat_selected_database')
    }
  }, [selectedDatabase, availableDatabases])

  // Get retrieval strategies for the current database
  const { availableStrategies, defaultStrategy } = useMemo(() => {
    try {
      const ragConfig = (projectDetail as any)?.project?.config?.rag
      if (!ragConfig?.databases || !Array.isArray(ragConfig.databases)) {
        return { availableStrategies: [], defaultStrategy: null }
      }

      const dbName = currentDatabase
      const dbConfig = ragConfig.databases.find(
        (db: any) => db?.name === dbName
      )
      if (!dbConfig?.retrieval_strategies || !Array.isArray(dbConfig.retrieval_strategies)) {
        return { availableStrategies: [], defaultStrategy: null }
      }

      const strategies = dbConfig.retrieval_strategies
        .filter((s: any) => s?.name)
        .map((s: any) => ({
          name: String(s.name),
          type: String(s.type || ''),
          isDefault: Boolean(s.default),
        }))

      // Find the default strategy: explicit default_retrieval_strategy, or one marked default, or first
      let defaultStrat: string | null = null
      if (dbConfig.default_retrieval_strategy) {
        defaultStrat = String(dbConfig.default_retrieval_strategy)
      } else {
        const markedDefault = strategies.find((s: any) => s.isDefault)
        if (markedDefault) {
          defaultStrat = markedDefault.name
        } else if (strategies.length > 0) {
          defaultStrat = strategies[0].name
        }
      }

      return { availableStrategies: strategies, defaultStrategy: defaultStrat }
    } catch {
      return { availableStrategies: [], defaultStrategy: null }
    }
  }, [projectDetail, currentDatabase])

  // Selected retrieval strategy state - persisted to localStorage
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('lf_testchat_selected_strategy')
  })

  // Validate and persist strategy selection
  useEffect(() => {
    // Don't validate until strategies are loaded
    if (availableStrategies.length === 0) {
      return
    }

    const validStrategyNames = availableStrategies.map((s: any) => s.name)

    // If user has a selected strategy and it's valid for current database, keep it
    if (selectedStrategy && validStrategyNames.includes(selectedStrategy)) {
      localStorage.setItem('lf_testchat_selected_strategy', selectedStrategy)
      return
    }

    // Selected strategy is invalid or doesn't exist for this database - fall back to default
    if (defaultStrategy) {
      setSelectedStrategy(defaultStrategy)
      localStorage.setItem('lf_testchat_selected_strategy', defaultStrategy)
    } else if (validStrategyNames.length > 0) {
      setSelectedStrategy(validStrategyNames[0])
      localStorage.setItem('lf_testchat_selected_strategy', validStrategyNames[0])
    } else {
      setSelectedStrategy(null)
      localStorage.removeItem('lf_testchat_selected_strategy')
    }
  }, [currentDatabase, defaultStrategy, availableStrategies, selectedStrategy])

  // Project session management for Project Chat (with persistence)
  const projectSession = useProjectSession({
    chatService: 'project',
    autoCreate: false, // Sessions created on first message
  })

  // Fallback chatbox for when project chat is not available
  const {
    messages: fallbackMessages,
    inputValue: fallbackInputValue,
    isSending: fallbackIsSending,
    isClearing: fallbackIsClearing,
    error: fallbackError,
    sendMessage: fallbackSendMessage,
    clearChat: fallbackClearChat,
    updateInput: fallbackUpdateInput,
    hasMessages: fallbackHasMessages,
    canSend: fallbackCanSend,
    addMessage: fallbackAddMessage,
    updateMessage: fallbackUpdateMessage,
  } = useChatbox()

  // Use project chat if we have an active project and not in mock mode
  const USE_PROJECT_CHAT = !MOCK_MODE && !!chatParams

  // Project chat state management
  const [projectInputValue, setProjectInputValue] = useState('')
  const [isProjectSending, setIsProjectSending] = useState(false)

  // Convert project session messages to chatbox format
  const projectSessionMessages: ChatboxMessage[] = projectSession.messages.map(
    msg => ({
      id: msg.id,
      type: msg.role === 'user' ? ('user' as const) : ('assistant' as const),
      content: msg.content,
      timestamp: new Date(msg.timestamp),
    })
  )
  // Transient streaming assistant message (not persisted)
  const [streamingMessage, setStreamingMessage] =
    useState<ChatboxMessage | null>(null)

  // Choose which chat system to use
  const messages = USE_PROJECT_CHAT
    ? streamingMessage
      ? [...projectSessionMessages, streamingMessage]
      : projectSessionMessages
    : fallbackMessages
  const inputValue = USE_PROJECT_CHAT ? projectInputValue : fallbackInputValue
  const isSending = USE_PROJECT_CHAT ? isProjectSending : fallbackIsSending
  const isClearing = USE_PROJECT_CHAT ? false : fallbackIsClearing // Project chat doesn't have clearing in this context
  const error = USE_PROJECT_CHAT ? projectSession.error : fallbackError
  const hasMessages = USE_PROJECT_CHAT
    ? projectSessionMessages.length > 0
    : fallbackHasMessages
  const canSend = USE_PROJECT_CHAT
    ? !isProjectSending &&
      projectInputValue.trim().length > 0 &&
      !projectChatStreamingMessage.isPending
    : fallbackCanSend

  // Combined loading state
  const isProjectChatLoading = projectChatStreamingMessage.isPending
  const combinedIsSending = isSending || isProjectChatLoading

  // Combined error state
  const projectChatError =
    projectChatStreamingMessage.error ||
    projectChatStreamingSession.sessionError
  const combinedError = (() => {
    if (error) {
      return typeof error === 'string' ? error : error.message
    }
    if (projectChatError) {
      return typeof projectChatError === 'string'
        ? projectChatError
        : projectChatError.message
    }
    return null
  })()

  // Combined canSend state
  const combinedCanSend = canSend && !isProjectChatLoading
  const projectSessionId = projectSession.sessionId

  // Project chat message management functions
  const addMessage = useCallback(
    (message: Omit<ChatboxMessage, 'id'>) => {
      if (USE_PROJECT_CHAT) {
        // Add message to project session (which handles persistence and creates temp session if needed)
        try {
          const sessionMessage = projectSession.addMessage(
            message.content,
            message.type === 'user' ? 'user' : 'assistant'
          )
          return sessionMessage.id
        } catch (err) {
          console.error('Failed to add message to project session:', err)
          // Generate a fallback ID for UI purposes
          return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
        }
      } else {
        return fallbackAddMessage(message)
      }
    },
    [USE_PROJECT_CHAT, projectSession, fallbackAddMessage]
  )

  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatboxMessage>) => {
      if (USE_PROJECT_CHAT) {
        // For project session, we can't update individual messages after they're saved
        // This is only used for streaming updates before final save
        console.warn(
          'updateMessage called on project session - this should only be used for streaming updates'
        )
      } else {
        fallbackUpdateMessage(id, updates)
      }
    },
    [USE_PROJECT_CHAT, fallbackUpdateMessage]
  )

  const clearChat = useCallback(() => {
    if (USE_PROJECT_CHAT) {
      projectSession.clearHistory()
    } else {
      fallbackClearChat()
    }
  }, [USE_PROJECT_CHAT, projectSession, fallbackClearChat])

  const updateInput = useCallback(
    (value: string) => {
      if (USE_PROJECT_CHAT) {
        setProjectInputValue(value)
      } else {
        fallbackUpdateInput(value)
      }
    },
    [USE_PROJECT_CHAT, fallbackUpdateInput]
  )

  const listRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const anomalyInputRef = useRef<HTMLTextAreaElement | null>(null)
  const classifierInputRef = useRef<HTMLTextAreaElement | null>(null)
  const lastUserInputRef = useRef<string>('')
  const rafRef = useRef<number | null>(null)

  // Sticky scroll state (matching Chatbox pattern)
  const BOTTOM_THRESHOLD = 24 // pixels from bottom to consider "at bottom"
  const [isUserAtBottom, setIsUserAtBottom] = useState(true)
  const [wantsAutoScroll, setWantsAutoScroll] = useState(true)

  // Check if user is at bottom of scroll container
  const checkIfAtBottom = useCallback(() => {
    if (!listRef.current) return false
    const { scrollTop, scrollHeight, clientHeight } = listRef.current
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight
    return distanceFromBottom <= BOTTOM_THRESHOLD
  }, [])

  // Handle scroll events with RAF debouncing
  const handleScroll = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
    }

    rafRef.current = requestAnimationFrame(() => {
      const atBottom = checkIfAtBottom()
      setIsUserAtBottom(atBottom)

      if (atBottom) {
        setWantsAutoScroll(true)
      } else {
        setWantsAutoScroll(false)
      }
    })
  }, [checkIfAtBottom])

  // Jump to latest handler
  const handleJumpToLatest = useCallback(() => {
    if (listRef.current) {
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: 'smooth',
      })
      setWantsAutoScroll(true)
      setIsUserAtBottom(true)
    }
  }, [])

  // Auto-grow textarea up to a comfortable max height before scrolling
  const resizeTextarea = useCallback(() => {
    const el = inputRef.current
    if (!el) return
    const maxHeight = 220 // ~6 lines depending on line-height
    el.style.height = 'auto'
    const newHeight = Math.min(el.scrollHeight, maxHeight)
    el.style.height = `${newHeight}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [])

  // RAF cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [])

  // Handle window resize - recompute scroll position (matching Chatbox)
  useEffect(() => {
    const handleResize = () => {
      // Recompute if we're at bottom after resize
      const atBottom = checkIfAtBottom()
      setIsUserAtBottom(atBottom)

      // If we were wanting to auto-scroll and we're now at bottom, maintain that
      if (wantsAutoScroll && atBottom && listRef.current) {
        listRef.current.scrollTo({
          top: listRef.current.scrollHeight,
          behavior: 'auto',
        })
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [checkIfAtBottom, wantsAutoScroll])

  // Auto-scroll when messages change (only if user wants it)
  useEffect(() => {
    if (wantsAutoScroll && listRef.current) {
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: 'auto', // 'auto' prevents jank during streaming
      })
    }
  }, [messages, wantsAutoScroll])

  // Resize textarea on mount and input changes
  useEffect(() => {
    resizeTextarea()
  }, [inputValue, resizeTextarea])

  // Clear chat only when switching between two different projects (not on initial mount)
  const prevProjectRef = useRef<{ ns: string; id: string } | null>(null)
  useEffect(() => {
    const ns = chatParams?.namespace
    const id = chatParams?.projectId
    if (!MOCK_MODE && ns && id) {
      const prev = prevProjectRef.current
      if (prev && (prev.ns !== ns || prev.id !== id)) {
        // Project actually changed: clear local chat UI and streaming session
        clearChat()
        projectChatStreamingSession.clearSession()
      }
      // Update ref after handling
      prevProjectRef.current = { ns, id }
    }
  }, [
    chatParams?.namespace,
    chatParams?.projectId,
    MOCK_MODE,
    clearChat,
    projectChatStreamingSession,
  ])

  const handleSend = useCallback(async () => {
    const content = inputValue.trim()
    if (!combinedCanSend || !content) return

    // Prevent multiple simultaneous requests
    if (combinedIsSending) {
      return
    }

    // Re-enable auto-scroll when sending a message
    setWantsAutoScroll(true)
    setIsUserAtBottom(true)

    if (MOCK_MODE) {
      // Local-only optimistic flow without backend
      addMessage({ type: 'user', content, timestamp: new Date() })
      lastUserInputRef.current = content
      const assistantId = addMessage({
        type: 'assistant',
        content: 'Loading model (may take a moment if downloading)…',
        timestamp: new Date(),
        isLoading: true,
      })
      updateInput('')
      setTimeout(() => {
        const mockAnswer = `Here is a mock response to: "${content}"\n\n- Point A\n- Point B\n\nThis is sample output while backend is disconnected.`
        const mockPrompts = [
          'System: You are an expert assistant. Answer clearly and concisely.',
          'Instruction: Provide helpful, safe, and accurate output.',
          `User input: ${content}`,
        ]
        const mockThinking = [
          'Identified intent and key entities.',
          'Searched internal knowledge for relevant facts.',
          'Composed structured response with bullet points.',
        ]
        updateMessage(assistantId, {
          content: mockAnswer,
          isLoading: false,
          sources: [
            {
              source: 'dataset/manuals/aircraft_mx_guide.pdf',
              score: 0.83,
              page: 12,
              chunk: 4,
              length: 182,
              content:
                'Hydraulic pressure drops during taxi often indicate minor leaks or entrained air. Inspect lines and fittings.',
            },
            {
              source: 'dataset/bulletins/bulletin-2024-17.md',
              score: 0.71,
              page: 3,
              chunk: 2,
              length: 126,
              content:
                'Pressure sensor calibration drifts were reported in batch 24B. Verify calibration if readings fluctuate.',
            },
          ],
          metadata: {
            prompts: mockPrompts,
            thinking: mockThinking,
            generation: {
              temperature: 0.7,
              topP: 0.9,
              maxTokens: 256,
              presencePenalty: 0.0,
              frequencyPenalty: 0.0,
              seed: undefined,
            },
          },
        })
      }, 1000)
      return
    }

    if (USE_PROJECT_CHAT && chatParams) {
      // Use project chat streaming API
      let accumulatedContent = ''
      const transientId = `stream_${Date.now()}`

      setIsProjectSending(true)

      try {
        // Add user message to project session
        projectSession.addMessage(content, 'user')
        lastUserInputRef.current = content

        // Show transient streaming bubble (not persisted)
        setStreamingMessage({
          id: transientId,
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
          isLoading: true,
        })

        updateInput('')

        // Send streaming message via project chat
        const finalSessionId = await projectChatStreamingMessage.mutateAsync({
          namespace: chatParams.namespace,
          projectId: chatParams.projectId,
          message: content,
          sessionId: projectChatStreamingSession.sessionId || undefined,
          requestOptions: {
            temperature:
              typeof genSettings?.temperature === 'number'
                ? genSettings?.temperature
                : undefined,
            top_p:
              typeof genSettings?.topP === 'number'
                ? genSettings?.topP
                : undefined,
            max_tokens:
              typeof genSettings?.maxTokens === 'number'
                ? genSettings?.maxTokens
                : undefined,
            presence_penalty:
              typeof genSettings?.presencePenalty === 'number'
                ? genSettings?.presencePenalty
                : undefined,
            frequency_penalty:
              typeof genSettings?.frequencyPenalty === 'number'
                ? genSettings?.frequencyPenalty
                : undefined,
            // Thinking/reasoning model parameters
            think: genSettings?.enableThinking === true ? true : undefined,
            thinking_budget:
              genSettings?.enableThinking && genSettings?.thinkingBudget
                ? genSettings.thinkingBudget
                : undefined,
            model:
              selectedModel ||
              (defaultModel as any)?.name ||
              fallbackDefaultName ||
              undefined,
            database: getCurrentDatabase() || undefined,
            rag_enabled: ragEnabled,
            rag_top_k: ragEnabled ? ragTopK : undefined,
            rag_score_threshold: ragEnabled ? ragScoreThreshold : undefined,
            rag_retrieval_strategy: ragEnabled && selectedStrategy ? selectedStrategy : undefined,
          },
          streamingOptions: {
            onChunk: (chunk: ChatStreamChunk) => {
              // Handle content chunks
              if (chunk.choices?.[0]?.delta?.content) {
                accumulatedContent += chunk.choices[0].delta.content
                setStreamingMessage({
                  id: transientId,
                  type: 'assistant',
                  content: accumulatedContent,
                  timestamp: new Date(),
                  isStreaming: true,
                  isLoading: false,
                })
              }
            },
            onError: (error: Error) => {
              console.error('Project chat streaming error:', error)
              // Persist error and clear transient bubble
              setStreamingMessage(null)
              projectSession.addMessage(`Error: ${error.message}`, 'assistant')
            },
            onComplete: () => {
              if (accumulatedContent && accumulatedContent.trim()) {
                // Append final assistant message once and clear transient bubble
                projectSession.addMessage(accumulatedContent, 'assistant')
              }
              setStreamingMessage(null)
            },
          },
        })

        // Update session ID if we got a new one
        if (
          finalSessionId &&
          finalSessionId !== projectChatStreamingSession.sessionId
        ) {
          projectChatStreamingSession.setSessionId(finalSessionId)
        }

        // Create project session if we got a new session ID
        if (finalSessionId) {
          projectSession.createSessionFromServer(finalSessionId)
        }
      } catch (error) {
        console.error('Project chat streaming error:', error)
        // Clear transient bubble and show error in project session
        setStreamingMessage(null)
        projectSession.addMessage(
          `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
          'assistant'
        )
      } finally {
        setIsProjectSending(false)
      }
      return
    }

    // Fallback to original chat system
    const ok = await fallbackSendMessage(content)
    if (ok) updateInput('')
  }, [
    combinedCanSend,
    inputValue,
    combinedIsSending,
    MOCK_MODE,
    USE_PROJECT_CHAT,
    chatParams,
    projectChatStreamingMessage,
    projectChatStreamingSession.sessionId,
    projectChatStreamingSession.setSessionId,
    projectSession,
    addMessage,
    updateMessage,
    updateInput,
    fallbackSendMessage,
    genSettings,
    selectedModel,
    defaultModel,
    fallbackDefaultName,
    getCurrentDatabase,
    ragEnabled,
    ragTopK,
    ragScoreThreshold,
    selectedStrategy,
  ])

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Wire lightweight global events for Retry and Use-as-prompt
  useEffect(() => {
    const onRetry = () => {
      // For now, simply re-send the last user message
      const lastUser = [...messages].reverse().find(m => m.type === 'user')
      if (lastUser) {
        updateInput(lastUser.content)
        setTimeout(() => handleSend(), 0)
      }
    }
    const onUse = (e: Event) => {
      const detail = (e as CustomEvent).detail as { content: string }
      updateInput(detail.content || '')
    }
    window.addEventListener('lf-chat-retry', onRetry as EventListener)
    window.addEventListener('lf-chat-use-as-prompt', onUse as EventListener)
    return () => {
      window.removeEventListener('lf-chat-retry', onRetry as EventListener)
      window.removeEventListener(
        'lf-chat-use-as-prompt',
        onUse as EventListener
      )
    }
  }, [messages, updateInput, handleSend])

  // Auto-focus input when navigated from Build assistant
  useEffect(() => {
    if (focusInput && inputRef.current) {
      // Small delay to ensure component is fully mounted
      const timer = setTimeout(() => {
        inputRef.current?.focus()
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [focusInput])

  // Lightweight evaluator and mock generator for test runs
  const evaluateTest = useCallback(
    (input: string, expected: string, actual: string) => {
      const tokenize = (s: string) =>
        s
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, ' ')
          .split(/\s+/)
          .filter(Boolean)
      const a = new Set(tokenize(expected))
      const b = new Set(tokenize(actual))
      let inter = 0
      a.forEach(t => {
        if (b.has(t)) inter++
      })
      const union = a.size + b.size - inter || 1
      let score = (inter / union) * 100
      // Gentle bias upward and random jitter for realism
      score = Math.max(0, Math.min(100, score + 12 + (Math.random() * 6 - 3)))
      score = Math.round(score * 10) / 10
      const latencyMs = 120 + Math.round(Math.random() * 280)
      const promptTokens = tokenize(input).length
      const completionTokens = tokenize(actual).length
      return {
        score,
        latencyMs,
        tokenUsage: {
          prompt: promptTokens,
          completion: completionTokens,
          total: promptTokens + completionTokens,
        },
      }
    },
    []
  )

  // Handle external test-run events from the Tests panel
  useEffect(() => {
    const onRun = async (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        id: number
        name: string
        input: string
        expected: string
      }

      const input = (detail.input || '').trim()
      // const expected = (detail.expected || '').trim()

      // For test runs, we need to ensure we have a valid project
      if (!chatParams) {
        console.error('Test runs require a valid project configuration.')
        return
      }

      // Re-enable auto-scroll when running a test
      setWantsAutoScroll(true)
      setIsUserAtBottom(true)

      try {
        // Add test input message to project session
        const userMessage = input || '(no input provided)'
        projectSession.addMessage(userMessage, 'user')
        lastUserInputRef.current = input

        // Show transient streaming bubble (same as normal send)
        const transientId = `stream_test_${Date.now()}`
        setStreamingMessage({
          id: transientId,
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
          isLoading: true,
        })

        // Send the actual test input via project chat streaming
        let accumulatedContent = ''
        const finalSessionId = await projectChatStreamingMessage.mutateAsync({
          namespace: chatParams.namespace,
          projectId: chatParams.projectId,
          message: userMessage,
          sessionId: projectChatStreamingSession.sessionId || undefined,
          requestOptions: {
            temperature:
              typeof genSettings?.temperature === 'number'
                ? genSettings?.temperature
                : undefined,
            top_p:
              typeof genSettings?.topP === 'number'
                ? genSettings?.topP
                : undefined,
            max_tokens:
              typeof genSettings?.maxTokens === 'number'
                ? genSettings?.maxTokens
                : undefined,
            presence_penalty:
              typeof genSettings?.presencePenalty === 'number'
                ? genSettings?.presencePenalty
                : undefined,
            frequency_penalty:
              typeof genSettings?.frequencyPenalty === 'number'
                ? genSettings?.frequencyPenalty
                : undefined,
            // Thinking/reasoning model parameters
            think: genSettings?.enableThinking === true ? true : undefined,
            thinking_budget:
              genSettings?.enableThinking && genSettings?.thinkingBudget
                ? genSettings.thinkingBudget
                : undefined,
            model:
              selectedModel ||
              (defaultModel as any)?.name ||
              fallbackDefaultName ||
              undefined,
            database: getCurrentDatabase() || undefined,
            rag_enabled: ragEnabled,
            rag_top_k: ragEnabled ? ragTopK : undefined,
            rag_score_threshold: ragEnabled ? ragScoreThreshold : undefined,
            rag_retrieval_strategy: ragEnabled && selectedStrategy ? selectedStrategy : undefined,
          },
          streamingOptions: {
            onChunk: (chunk: ChatStreamChunk) => {
              if (chunk.choices?.[0]?.delta?.content) {
                accumulatedContent += chunk.choices[0].delta.content
                setStreamingMessage({
                  id: transientId,
                  type: 'assistant',
                  content: accumulatedContent,
                  timestamp: new Date(),
                  isStreaming: true,
                  isLoading: false,
                })
              }
            },
            onError: (error: Error) => {
              console.error('Test streaming error:', error)
              setStreamingMessage(null)
              projectSession.addMessage(`Error: ${error.message}`, 'assistant')
            },
            onComplete: () => {
              if (accumulatedContent && accumulatedContent.trim()) {
                projectSession.addMessage(accumulatedContent, 'assistant')
              }
              setStreamingMessage(null)
            },
          },
        })

        // Update session ID if we got a new one
        if (
          finalSessionId &&
          finalSessionId !== projectChatStreamingSession.sessionId
        ) {
          projectChatStreamingSession.setSessionId(finalSessionId)
        }
      } catch (error) {
        console.error('Test run error:', error)
        // Add error message
        addMessage({
          type: 'assistant',
          content: `Error running test: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
          metadata: {
            isTest: true,
            testId: detail.id,
            testName: detail.name,
            error: true,
          },
        })
      }
    }

    window.addEventListener('lf-test-run', onRun as EventListener)
    return () =>
      window.removeEventListener('lf-test-run', onRun as EventListener)
  }, [
    addMessage,
    updateMessage,
    evaluateTest,
    chatParams,
    projectChatStreamingMessage,
    projectSessionId,
    genSettings,
    selectedModel,
    defaultModel,
    fallbackDefaultName,
    getCurrentDatabase,
    ragEnabled,
    ragTopK,
    ragScoreThreshold,
    selectedStrategy,
    projectSession,
    setStreamingMessage,
    projectChatStreamingSession.sessionId,
    projectChatStreamingSession.setSessionId,
  ])

  // ============================================================================
  // Anomaly Detection Handlers
  // ============================================================================

  // Parse anomaly input (supports comma and tab separated values)
  const parseAnomalyInput = useCallback((input: string): {
    values: string[]
    error: string | null
  } => {
    if (!input.trim()) {
      return { values: [], error: null }
    }

    // Convert tabs to commas (for spreadsheet paste support)
    const normalized = input.replace(/\t/g, ',')

    // Split by comma
    const values = normalized.split(',').map(v => v.trim()).filter(Boolean)

    if (values.length === 0) {
      return { values: [], error: 'Unable to parse input values' }
    }

    return { values, error: null }
  }, [])

  // Handle anomaly detection
  const handleAnomalyDetect = useCallback(async () => {
    if (!selectedAnomalyModel || !anomalyInput.trim()) return
    if (scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending) return

    setAnomalyError(null)
    const { values, error } = parseAnomalyInput(anomalyInput)

    if (error) {
      setAnomalyError(error)
      return
    }

    // Try to parse as numbers for numeric data
    const numericValues = values.map(v => parseFloat(v))
    const isNumeric = numericValues.every(n => !isNaN(n))

    try {
      // Load model first
      const selectedModelInfo = sortedAnomalyModels.find(m => m.name === selectedAnomalyModel)
      await loadAnomalyMutation.mutateAsync({
        model: selectedAnomalyModel,
        backend: selectedModelInfo?.backend,
      })

      // Score the data
      const result = await scoreAnomalyMutation.mutateAsync({
        model: selectedAnomalyModel,
        backend: selectedModelInfo?.backend,
        data: isNumeric ? [numericValues] : [values.reduce((acc, v, i) => {
          acc[`col_${i + 1}`] = v
          return acc
        }, {} as Record<string, unknown>)],
        ...(isNumeric ? {} : {
          schema: values.reduce((acc, _, i) => {
            acc[`col_${i + 1}`] = 'label' as const
            return acc
          }, {} as Record<string, 'label'>)
        }),
      })

      if (result.data && result.data.length > 0) {
        const newResult = {
          score: result.data[0].score,
          isAnomaly: result.data[0].is_anomaly,
          threshold: result.summary.threshold,
          parsedInput: values,
        }
        setAnomalyResult(newResult)

        // Add to history
        setAnomalyHistory(prev => [{
          id: `anomaly-${Date.now()}`,
          timestamp: new Date(),
          ...newResult,
          modelName: selectedAnomalyModel,
        }, ...prev].slice(0, 50)) // Keep last 50 entries

        // Clear input and refocus
        setAnomalyInput('')
        setTimeout(() => anomalyInputRef.current?.focus(), 0)
      } else {
        setAnomalyError('No results returned from model')
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Detection failed'
      setAnomalyError(errorMsg)

      // Add error to history
      setAnomalyHistory(prev => [{
        id: `anomaly-${Date.now()}`,
        timestamp: new Date(),
        score: 0,
        isAnomaly: false,
        threshold: 0,
        parsedInput: values,
        modelName: selectedAnomalyModel,
        error: errorMsg,
      }, ...prev].slice(0, 50))
    }
  }, [selectedAnomalyModel, anomalyInput, parseAnomalyInput, sortedAnomalyModels, loadAnomalyMutation, scoreAnomalyMutation])

  // Clear anomaly results (not the mode or model selection)
  const clearAnomalyResults = useCallback(() => {
    setAnomalyResult(null)
    setAnomalyError(null)
    setAnomalyInput('')
  }, [])

  // ============================================================================
  // Classifier Handlers
  // ============================================================================

  // Handle classification
  const handleClassify = useCallback(async () => {
    if (predictClassifierMutation.isPending || loadClassifierMutation.isPending) return
    if (!selectedClassifierModel || !classifierInput.trim()) {
      if (!classifierInput.trim()) {
        setClassifierError('Enter some text to classify')
      }
      return
    }

    setClassifierError(null)
    setLastClassifierInput(classifierInput.trim())

    try {
      // Load model first
      await loadClassifierMutation.mutateAsync({
        model: selectedClassifierModel,
      })

      // Run prediction
      const result = await predictClassifierMutation.mutateAsync({
        model: selectedClassifierModel,
        texts: [classifierInput.trim()],
      })

      if (result.data && result.data.length > 0) {
        const prediction = result.data[0]
        // Build sorted predictions from all_scores if available
        const predictions: Array<{ label: string; score: number }> = []

        if (prediction.all_scores) {
          // Sort by score descending
          const sortedLabels = Object.entries(prediction.all_scores)
            .sort(([, a], [, b]) => b - a)
            .map(([label, score]) => ({ label, score }))
          predictions.push(...sortedLabels)
        } else {
          // Just use the top prediction
          predictions.push({
            label: prediction.label,
            score: prediction.score,
          })
        }

        const newResult = {
          predictions,
          isMultiLabel: false, // TODO: check if model is multi-label from model info
          threshold: 0.5, // Default threshold for multi-label
        }
        setClassifierResult(newResult)

        // Add to history
        const inputTextForHistory = classifierInput.trim()
        setClassifierHistory(prev => [{
          id: `classifier-${Date.now()}`,
          timestamp: new Date(),
          inputText: inputTextForHistory,
          topLabel: predictions[0]?.label || '',
          topScore: predictions[0]?.score || 0,
          predictions,
          modelName: selectedClassifierModel,
        }, ...prev].slice(0, 50)) // Keep last 50 entries

        // Clear input and refocus
        setClassifierInput('')
        setTimeout(() => classifierInputRef.current?.focus(), 0)
      } else {
        setClassifierError('No results returned from model')
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Classification failed'
      setClassifierError(errorMsg)

      // Add error to history
      setClassifierHistory(prev => [{
        id: `classifier-${Date.now()}`,
        timestamp: new Date(),
        inputText: classifierInput.trim(),
        topLabel: 'Error',
        topScore: 0,
        predictions: [],
        modelName: selectedClassifierModel,
        error: errorMsg,
      }, ...prev].slice(0, 50))
    }
  }, [selectedClassifierModel, classifierInput, loadClassifierMutation, predictClassifierMutation])

  // Clear classifier results (not the mode or model selection)
  const clearClassifierResults = useCallback(() => {
    setClassifierResult(null)
    setClassifierError(null)
    setClassifierInput('')
    setLastClassifierInput('')
  }, [])

  // ============================================================================
  // Document Scanning Handlers
  // ============================================================================

  // Auto-scan when file is selected
  const handleScanFileSelect = useCallback(async (file: File) => {
    setScanFile(file)
    setScanResults(null)
    setScanError(null)

    // Automatically start scanning
    try {
      const result = await scanDocumentMutation.mutateAsync({
        file,
        model: selectedScanBackend,
        languages: selectedScanLanguage,
        returnBoxes: false,
      })

      if (result.data) {
        setScanResults(result.data)

        // Mark that user has completed a scan (for first-time message)
        if (!hasScannedBefore) {
          setHasScannedBefore(true)
          localStorage.setItem('lf_scan_completed', 'true')
        }

        // Add to history
        const avgConfidence = result.data.reduce((sum, r) => sum + r.confidence, 0) / result.data.length
        const previewText = result.data[0]?.text?.substring(0, 100) || ''

        setScanHistory(prev => [{
          id: `scan-${Date.now()}`,
          timestamp: new Date(),
          fileName: file.name,
          pageCount: result.data.length,
          avgConfidence,
          previewText,
          backend: selectedScanBackend,
          results: result.data,
        }, ...prev].slice(0, 50))
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Document scanning failed'
      setScanError(errorMsg)

      // Add error to history
      setScanHistory(prev => [{
        id: `scan-${Date.now()}`,
        timestamp: new Date(),
        fileName: file.name,
        pageCount: 0,
        avgConfidence: 0,
        previewText: '',
        backend: selectedScanBackend,
        results: [],
        error: errorMsg,
      }, ...prev].slice(0, 50))
    }
  }, [selectedScanBackend, selectedScanLanguage, scanDocumentMutation, hasScannedBefore])

  const clearScanResults = useCallback(() => {
    setScanResults(null)
    setScanError(null)
    setScanFile(null)
  }, [])

  const handleScanHistorySelect = useCallback((historyItem: DocumentScanningHistoryEntry) => {
    // Restore results from history
    setScanResults(historyItem.results)
    setScanError(historyItem.error || null)
  }, [])

  // ============================================================================
  // Encoder Handlers
  // ============================================================================

  // Calculate cosine similarity between two vectors
  const cosineSimilarity = useCallback((a: number[], b: number[]): number => {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0)
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
    return normA && normB ? dot / (normA * normB) : 0
  }, [])

  // Handle embedding similarity calculation
  const handleEmbedding = useCallback(async () => {
    const textA = embeddingTextA.trim()
    const textB = embeddingTextB.trim()

    if (!textA || !textB) {
      setEncoderError('Enter text in both fields to compare')
      return
    }

    const texts = [textA, textB]
    setEncoderError(null)
    setEmbeddingResult(null)

    try {
      const response = await createEmbeddingsMutation.mutateAsync({
        model: selectedEmbeddingModel,
        input: texts,
      })

      // Extract embeddings
      const embeddings = response.data.map(d => d.embedding)

      // Calculate similarity between the two texts
      const similarity = cosineSimilarity(embeddings[0], embeddings[1])

      setEmbeddingResult({ texts, similarity })

      // Add to history
      setEncoderHistory(prev => [{
        id: `encoder-${Date.now()}`,
        timestamp: new Date(),
        mode: 'embedding' as const,
        modelName: selectedEmbeddingModel,
        texts,
        similarity,
      }, ...prev].slice(0, 50))

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Embedding failed'
      setEncoderError(errorMsg)

      setEncoderHistory(prev => [{
        id: `encoder-${Date.now()}`,
        timestamp: new Date(),
        mode: 'embedding' as const,
        modelName: selectedEmbeddingModel,
        texts,
        error: errorMsg,
      }, ...prev].slice(0, 50))
    }
  }, [selectedEmbeddingModel, embeddingTextA, embeddingTextB, createEmbeddingsMutation, cosineSimilarity])

  // Handle document reranking
  const handleRerank = useCallback(async () => {
    const query = rerankQuery.trim()
    const documents = rerankDocuments.map(d => d.trim()).filter(d => d)

    if (!query) {
      setEncoderError('Enter a query')
      return
    }
    if (documents.length < 2) {
      setEncoderError('Enter at least 2 documents to rerank')
      return
    }

    setEncoderError(null)
    setRerankResult(null)

    try {
      const response = await rerankMutation.mutateAsync({
        model: selectedRerankingModel,
        query,
        documents,
        return_documents: true,
      })

      setRerankResult(response.data)

      // Add to history
      setEncoderHistory(prev => [{
        id: `encoder-${Date.now()}`,
        timestamp: new Date(),
        mode: 'reranking' as const,
        modelName: selectedRerankingModel,
        query,
        documents,
        results: response.data,
      }, ...prev].slice(0, 50))

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Reranking failed'
      setEncoderError(errorMsg)

      setEncoderHistory(prev => [{
        id: `encoder-${Date.now()}`,
        timestamp: new Date(),
        mode: 'reranking' as const,
        modelName: selectedRerankingModel,
        query,
        documents,
        error: errorMsg,
      }, ...prev].slice(0, 50))
    }
  }, [selectedRerankingModel, rerankQuery, rerankDocuments, rerankMutation])

  // Clear encoder results
  const clearEncoderResults = useCallback(() => {
    setEmbeddingResult(null)
    setRerankResult(null)
    setEncoderError(null)
    setEmbeddingTextA('')
    setEmbeddingTextB('')
    setRerankQuery('')
    setRerankDocuments(['', '']) // Reset to 2 empty documents
  }, [])

  return (
    <div className={containerClasses}>
      {/* Header row actions */}
      <div className="flex flex-col gap-2 px-3 md:px-4 py-2 border-b border-border rounded-t-xl bg-background/50">
        {/* First row: Project info + Clear button */}
        <div className="flex items-center justify-between">
          <div className="text-xs md:text-sm text-muted-foreground">
            {USE_PROJECT_CHAT && chatParams ? (
              <span>
                Project: {chatParams.namespace}/{chatParams.projectId}
                {projectChatStreamingSession.sessionId && (
                  <span className="ml-2 opacity-60">
                    • Session: {projectChatStreamingSession.sessionId.slice(-8)}
                  </span>
                )}
              </span>
            ) : (
              'Session'
            )}
          </div>
          {/* Clear button - far right */}
          <button
            type="button"
            onClick={() => {
              if (modelType === 'inference') {
                // Inference mode: clear chat
                clearChat()
                if (!MOCK_MODE && chatParams) {
                  projectChatStreamingSession.clearSession()
                }
                // Reset sending state to ensure input isn't stuck disabled
                setIsProjectSending(false)
                setStreamingMessage(null)
              } else if (modelType === 'anomaly') {
                // Anomaly mode: only clear results, NOT mode selection or model dropdown
                clearAnomalyResults()
              } else if (modelType === 'classifier') {
                // Classifier mode: only clear results, NOT mode selection or model dropdown
                clearClassifierResults()
              } else if (modelType === 'document_scanning') {
                // Document scanning mode: clear results and file
                clearScanResults()
              } else if (modelType === 'encoder') {
                // Encoder mode: clear results
                clearEncoderResults()
              }
            }}
            disabled={
              modelType === 'inference'
                ? (isClearing || !hasMessages)
                : modelType === 'anomaly'
                  ? (!anomalyResult && !anomalyError)
                  : modelType === 'classifier'
                    ? (!classifierResult && !classifierError)
                    : modelType === 'document_scanning'
                      ? (!scanResults && !scanError && !scanFile)
                      : (!embeddingResult && !rerankResult && !encoderError)
            }
            className="text-xs px-2 py-0.5 rounded bg-secondary/80 hover:bg-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isClearing ? 'Clearing…' : 'Clear'}
          </button>
        </div>
        {/* Second row: Model Type and mode-specific selectors */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 md:gap-x-5">
          {/* Model Type selector - FIRST dropdown */}
          <Selector
            value={modelType}
            options={[
              { value: 'inference', label: 'Text Generation' },
              { value: 'anomaly', label: 'Anomaly Detection' },
              { value: 'classifier', label: 'Classifier' },
              { value: 'document_scanning', label: 'Document Scanning' },
              { value: 'encoder', label: 'Encoder' },
            ]}
            onChange={(v) => onModelTypeChange(v as 'inference' | 'anomaly' | 'classifier' | 'document_scanning' | 'encoder')}
            label="Model Type"
            className="w-[200px]"
          />

          {/* Inference-specific selectors */}
          {modelType === 'inference' && USE_PROJECT_CHAT && (
            <>
              {/* Model selector */}
              <Selector
                value={selectedModel || (defaultModel as any)?.name || fallbackDefaultName || ''}
                options={unifiedModels.map(m => ({
                  value: m.name,
                  label: `${m.name} (${m.model})${m.default ? ' (default)' : ''}`,
                }))}
                onChange={setSelectedModel}
                loading={modelsLoading}
                placeholder="Select model"
                emptyMessage="No models"
                label="Model"
                className="min-w-[140px] max-w-[220px]"
              />

              {/* Database selector */}
              {availableDatabases.length > 0 && (
                <Selector
                  value={currentDatabase || ''}
                  options={availableDatabases.map((dbName: string) => ({
                    value: dbName,
                    label: `${dbName}${dbName === getCurrentDatabase() ? ' (default)' : ''}`,
                  }))}
                  onChange={(v) => setSelectedDatabase(v)}
                  label="Database"
                  className="min-w-[120px] max-w-[180px]"
                />
              )}

              {/* Retrieval Strategy selector */}
              <Selector
                value={selectedStrategy || ''}
                options={availableStrategies.map((s: { name: string; type: string; isDefault: boolean }) => ({
                  value: s.name,
                  label: `${s.name}${s.name === defaultStrategy ? ' (default)' : ''}`,
                }))}
                onChange={(v) => setSelectedStrategy(v || null)}
                disabled={availableStrategies.length === 0}
                placeholder="No strategies"
                emptyMessage="No strategies found"
                label="Strategy"
                className="min-w-[120px] max-w-[180px]"
              />
            </>
          )}

          {/* Anomaly-specific selectors */}
          {modelType === 'anomaly' && (
            <>
              <Selector
                value={selectedAnomalyModel || ''}
                options={sortedAnomalyModels.map(m => ({
                  value: m.name,
                  label: m.base_name || m.name,
                  description: m.description,
                }))}
                onChange={setSelectedAnomalyModel}
                loading={isLoadingAnomalyModels}
                disabled={scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending}
                placeholder="Select model"
                emptyMessage="No anomaly models"
                label="Anomaly Model"
                className="min-w-[140px]"
              />
              {selectedAnomalyModelInfo && (
                <div className="flex items-center text-xs text-muted-foreground self-end mb-2">
                  <span>{selectedAnomalyModelInfo.name}</span>
                </div>
              )}
            </>
          )}

          {/* Classifier-specific selectors */}
          {modelType === 'classifier' && (
            <>
              <Selector
                value={selectedClassifierModel || ''}
                options={sortedClassifierModels.map(m => ({
                  value: m.name,
                  label: m.base_name || m.name,
                  description: m.description,
                }))}
                onChange={setSelectedClassifierModel}
                loading={isLoadingClassifierModels}
                disabled={predictClassifierMutation.isPending || loadClassifierMutation.isPending}
                placeholder="Select model"
                emptyMessage="No classifier models"
                label="Classifier Model"
                className="min-w-[140px]"
              />
              {selectedClassifierModelInfo && (
                <div className="flex items-center text-xs text-muted-foreground self-end mb-2">
                  <span>{selectedClassifierModelInfo.name}</span>
                </div>
              )}
            </>
          )}

          {/* Document Scanning-specific selectors */}
          {modelType === 'document_scanning' && (
            <>
              <Selector
                value={selectedScanBackend}
                options={Object.entries(DOCUMENT_SCANNING_BACKEND_DISPLAY).map(([value, { label, description }]) => ({
                  value,
                  label,
                  description,
                }))}
                onChange={(v) => setSelectedScanBackend(v as DocumentScanningBackend)}
                label="Backend"
                className="min-w-[140px]"
              />
              <Selector
                value={selectedScanLanguage}
                options={DOCUMENT_SCANNING_LANGUAGES.map(lang => ({
                  value: lang.code,
                  label: lang.label,
                }))}
                onChange={setSelectedScanLanguage}
                label="Language"
                className="min-w-[100px]"
              />
            </>
          )}

          {/* Encoder-specific selectors */}
          {modelType === 'encoder' && (
            <>
              {/* Sub-mode toggle: Embedding vs Reranking */}
              <Selector
                value={encoderSubMode}
                options={[
                  { value: 'embedding', label: 'Embedding Similarity' },
                  { value: 'reranking', label: 'Document Reranking' },
                ]}
                onChange={(v) => setEncoderSubMode(v as EncoderSubMode)}
                label="Mode"
                className="min-w-[160px]"
              />

              {/* Model selector based on sub-mode */}
              {encoderSubMode === 'embedding' ? (
                <div className="flex items-end gap-2">
                  <Selector
                    value={selectedEmbeddingModel}
                    options={COMMON_EMBEDDING_MODELS.map(m => ({
                      value: m.value,
                      label: m.label,
                      description: m.description,
                    }))}
                    onChange={setSelectedEmbeddingModel}
                    disabled={createEmbeddingsMutation.isPending}
                    label="Embedding Model"
                    className="min-w-[180px]"
                  />
                  <button
                    onClick={() => {
                      const sample = EMBEDDING_SAMPLES[embeddingSampleIndex]
                      setEmbeddingTextA(sample.textA)
                      setEmbeddingTextB(sample.textB)
                      setEmbeddingResult(null)
                      setEncoderError(null)
                      setEmbeddingSampleIndex((embeddingSampleIndex + 1) % EMBEDDING_SAMPLES.length)
                    }}
                    disabled={createEmbeddingsMutation.isPending}
                    className="h-9 text-xs px-3 rounded-lg bg-secondary/60 hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                    title={`Sample ${embeddingSampleIndex + 1}/${EMBEDDING_SAMPLES.length}`}
                  >
                    Try Sample
                  </button>
                </div>
              ) : (
                <div className="flex items-end gap-2">
                  <Selector
                    value={selectedRerankingModel}
                    options={COMMON_RERANKING_MODELS.map(m => ({
                      value: m.value,
                      label: m.label,
                      description: m.description,
                    }))}
                    onChange={setSelectedRerankingModel}
                    disabled={rerankMutation.isPending}
                    label="Reranking Model"
                    className="min-w-[180px]"
                  />
                  <button
                    onClick={() => {
                      const sample = RERANKING_SAMPLES[rerankSampleIndex]
                      setRerankQuery(sample.query)
                      setRerankDocuments(sample.documents)
                      setRerankResult(null)
                      setEncoderError(null)
                      setRerankSampleIndex((rerankSampleIndex + 1) % RERANKING_SAMPLES.length)
                    }}
                    disabled={rerankMutation.isPending}
                    className="h-9 text-xs px-3 rounded-lg bg-secondary/60 hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                    title={`Sample ${rerankSampleIndex + 1}/${RERANKING_SAMPLES.length}`}
                  >
                    Try Sample
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {combinedError && (
        <div className="mx-4 mt-3 p-2 bg-red-100 border border-red-400 text-red-700 rounded text-xs">
          {combinedError}
        </div>
      )}

      {/* No active project warning - only for inference mode */}
      {modelType === 'inference' && !MOCK_MODE && !chatParams && (
        <div className="mx-4 mt-3 p-2 bg-amber-100 border border-amber-400 text-amber-700 rounded text-xs">
          No active project selected. Please select a project to use the chat
          feature.
        </div>
      )}

      {/* Main Content Area - mode conditional */}
      <div className="relative flex-1 min-h-0">
        {modelType === 'inference' ? (
          <>
            {/* Inference: Messages - wrapper with relative positioning for button (matching Chatbox) */}
            <div
              ref={listRef}
              onScroll={handleScroll}
              className="absolute inset-0 overflow-y-auto flex flex-col gap-4 p-3 md:p-4"
            >
              {unifiedModels.length === 0 && !modelsLoading ? (
                <InferenceNoModelsState
                  onAddModel={() => navigate('/chat/models/add')}
                />
              ) : !hasMessages ? (
                <EmptyState />
              ) : (
                messages.map((m: ChatboxMessage) => (
                  <TestChatMessage
                    key={m.id}
                    message={m}
                    showReferences={showReferences}
                    allowRanking={allowRanking}
                    showPrompts={showPrompts}
                    showThinking={showThinking}
                    lastUserInput={lastUserInputRef.current}
                    showGenSettings={showGenSettings}
                  />
                ))
              )}
              <div ref={endRef} />
            </div>
            {/* Jump to latest button - positioned outside scroll container */}
            {!isUserAtBottom && hasMessages && (
              <button
                onClick={handleJumpToLatest}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-3 py-2 rounded-full bg-primary/90 hover:bg-primary text-primary-foreground shadow-lg transition-all hover:shadow-xl"
                aria-label="Jump to latest message"
              >
                <span className="text-sm font-medium">Jump to latest</span>
                <svg
                  viewBox="0 0 24 24"
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="7 13 12 18 17 13" />
                  <polyline points="7 6 12 11 17 6" />
                </svg>
              </button>
            )}
          </>
        ) : modelType === 'anomaly' ? (
          /* Anomaly: Result display with optional history sidebar */
          <div className="absolute inset-0 flex overflow-hidden">
            {/* Main content area */}
            <div className="flex-1 overflow-y-auto p-3 md:p-4">
              {sortedAnomalyModels.length === 0 && !isLoadingAnomalyModels ? (
                <AnomalyEmptyState
                  hasModels={false}
                  onCreateModel={() => navigate('/chat/models/train/anomaly/new')}
                />
              ) : anomalyResult || anomalyError || scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending ? (
                <AnomalyResultDisplay
                  result={anomalyResult}
                  error={anomalyError}
                  isLoading={scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending}
                />
              ) : (
                <AnomalyEmptyState
                  hasModels={true}
                  onCreateModel={() => navigate('/chat/models/train/anomaly/new')}
                />
              )}
            </div>

            {/* History sidebar - right side, collapsible */}
            {anomalyHistory.length > 0 && (
              <div className={`flex-shrink-0 border-l border-border bg-muted/10 transition-all ${showAnomalyHistory ? 'w-[25%]' : 'w-8'}`}>
                {showAnomalyHistory ? (
                  <div className="flex flex-col h-full">
                    <div className="flex items-center justify-between px-3 py-1.5 border-b border-border">
                      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                        History
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setAnomalyHistory([])}
                          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Clear history"
                        >
                          <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => setShowAnomalyHistory(false)}
                          className="p-0.5 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Close history"
                        >
                          <FontIcon type="close" className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div
                      ref={anomalyHistoryScrollRef}
                      className="flex-1 overflow-y-auto p-2 space-y-2"
                    >
                      {anomalyHistory.map(item => (
                        <AnomalyHistoryItem
                          key={item.id}
                          item={item}
                          onRerun={(input) => setAnomalyInput(input)}
                        />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col h-full items-center pt-2">
                    <button
                      onClick={() => setShowAnomalyHistory(true)}
                      className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted/30 rounded"
                      title="Show history"
                    >
                      <FontIcon type="recently-viewed" className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : modelType === 'classifier' ? (
          /* Classifier: Result display with optional history sidebar */
          <div className="absolute inset-0 flex overflow-hidden">
            {/* Main content area */}
            <div className="flex-1 overflow-y-auto p-3 md:p-4">
              {sortedClassifierModels.length === 0 && !isLoadingClassifierModels ? (
                <ClassifierEmptyState
                  hasModels={false}
                  onCreateModel={() => navigate('/chat/models/train/classifier/new')}
                />
              ) : classifierResult || classifierError || predictClassifierMutation.isPending || loadClassifierMutation.isPending ? (
                <ClassifierResultDisplay
                  result={classifierResult}
                  error={classifierError}
                  isLoading={predictClassifierMutation.isPending || loadClassifierMutation.isPending}
                  inputText={lastClassifierInput}
                />
              ) : (
                <ClassifierEmptyState
                  hasModels={true}
                  onCreateModel={() => navigate('/chat/models/train/classifier/new')}
                />
              )}
            </div>

            {/* History sidebar - right side, collapsible */}
            {classifierHistory.length > 0 && (
              <div className={`flex-shrink-0 border-l border-border bg-muted/10 transition-all ${showClassifierHistory ? 'w-[25%]' : 'w-8'}`}>
                {showClassifierHistory ? (
                  <div className="flex flex-col h-full">
                    <div className="flex items-center justify-between px-3 py-1.5 border-b border-border">
                      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                        History
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setClassifierHistory([])}
                          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Clear history"
                        >
                          <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => setShowClassifierHistory(false)}
                          className="p-0.5 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Close history"
                        >
                          <FontIcon type="close" className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div
                      ref={classifierHistoryScrollRef}
                      className="flex-1 overflow-y-auto p-2 space-y-2"
                    >
                      {classifierHistory.map(item => (
                        <ClassifierHistoryItem
                          key={item.id}
                          item={item}
                          onRerun={(input) => setClassifierInput(input)}
                        />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col h-full items-center pt-2">
                    <button
                      onClick={() => setShowClassifierHistory(true)}
                      className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted/30 rounded"
                      title="Show history"
                    >
                      <FontIcon type="recently-viewed" className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : modelType === 'document_scanning' ? (
          /* Document Scanning: File upload and result display with optional history sidebar */
          <div className="absolute inset-0 flex overflow-hidden">
            {/* Main content area - always accepts file drops */}
            <DocumentScanningContainer
              onFileSelect={handleScanFileSelect}
              disabled={scanDocumentMutation.isPending}
              onInputRefReady={(trigger) => { triggerScanBrowseRef.current = trigger }}
            >
              {scanResults || scanError || scanDocumentMutation.isPending ? (
                <DocumentScanningResultDisplay
                  results={scanResults}
                  error={scanError}
                  isLoading={scanDocumentMutation.isPending}
                  fileName={scanFile?.name || ''}
                />
              ) : (
                <DocumentScanningEmptyState
                  onBrowseClick={() => triggerScanBrowseRef.current?.()}
                  disabled={scanDocumentMutation.isPending}
                />
              )}
            </DocumentScanningContainer>

            {/* History sidebar - right side, collapsible */}
            {scanHistory.length > 0 && (
              <div className={`flex-shrink-0 border-l border-border bg-muted/10 transition-all ${showScanHistory ? 'w-[25%]' : 'w-8'}`}>
                {showScanHistory ? (
                  <div className="flex flex-col h-full">
                    <div className="flex items-center justify-between px-3 py-1.5 border-b border-border">
                      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                        History
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setScanHistory([])}
                          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Clear history"
                        >
                          <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => setShowScanHistory(false)}
                          className="p-0.5 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                          title="Close history"
                        >
                          <FontIcon type="close" className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div
                      ref={scanHistoryScrollRef}
                      className="flex-1 overflow-y-auto p-2 space-y-2"
                    >
                      {scanHistory.map(item => (
                        <DocumentScanningHistoryItem
                          key={item.id}
                          item={item}
                          onSelect={() => handleScanHistorySelect(item)}
                        />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col h-full items-center pt-2">
                    <button
                      onClick={() => setShowScanHistory(true)}
                      className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted/30 rounded"
                      title="Show history"
                    >
                      <FontIcon type="recently-viewed" className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          /* Encoder: Embedding similarity or Reranking */
          <div className="absolute inset-0 flex overflow-hidden">
            {encoderSubMode === 'embedding' ? (
              /* Embedding mode - original layout */
              <>
                {/* Main content area */}
                <div className="flex-1 overflow-y-auto p-3 md:p-4">
                  {embeddingResult || encoderError || createEmbeddingsMutation.isPending ? (
                    <EmbeddingSimilarityDisplay
                      result={embeddingResult}
                      error={encoderError}
                      isLoading={createEmbeddingsMutation.isPending}
                      onCompareAnother={() => {
                        setEmbeddingResult(null)
                        setEncoderError(null)
                        setEmbeddingTextA('')
                        setEmbeddingTextB('')
                      }}
                    />
                  ) : (
                    <EncoderEmptyState subMode="embedding" />
                  )}
                </div>

                {/* History sidebar for embedding */}
                {encoderHistory.filter(h => h.mode === 'embedding').length > 0 && (
                  <div className={`flex-shrink-0 border-l border-border bg-muted/10 transition-all ${showEncoderHistory ? 'w-[25%]' : 'w-8'}`}>
                    {showEncoderHistory ? (
                      <div className="flex flex-col h-full">
                        <div className="flex items-center justify-between px-3 py-1.5 border-b border-border">
                          <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                            History
                          </span>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setEncoderHistory(prev => prev.filter(h => h.mode !== 'embedding'))}
                              className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                              title="Clear history"
                            >
                              <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => setShowEncoderHistory(false)}
                              className="p-0.5 text-muted-foreground hover:text-foreground rounded hover:bg-muted/50"
                              title="Close history"
                            >
                              <FontIcon type="close" className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                        <div
                          ref={encoderHistoryScrollRef}
                          className="flex-1 overflow-y-auto p-2 space-y-2"
                        >
                          {encoderHistory.filter(h => h.mode === 'embedding').map(item => (
                            <EncoderHistoryItem
                              key={item.id}
                              item={item}
                              onRerun={() => {
                                setEmbeddingTextA(item.texts?.[0] || '')
                                setEmbeddingTextB(item.texts?.[1] || '')
                                setEmbeddingResult(null)
                                setEncoderError(null)
                              }}
                            />
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col h-full items-center pt-2">
                        <button
                          onClick={() => setShowEncoderHistory(true)}
                          className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted/30 rounded"
                          title="Show history"
                        >
                          <FontIcon type="recently-viewed" className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              /* Reranking mode - split layout with inputs on right */
              <>
                {/* Left panel: Empty state / Results */}
                <div className="flex-1 overflow-y-auto p-3 md:p-4">
                  {rerankResult || encoderError || rerankMutation.isPending ? (
                    <RerankResultDisplay
                      result={rerankResult}
                      error={encoderError}
                      isLoading={rerankMutation.isPending}
                      query={rerankQuery}
                      onRankAgain={() => {
                        setRerankResult(null)
                        setEncoderError(null)
                        setRerankRightPanelTab('inputs')
                      }}
                    />
                  ) : (
                    <EncoderEmptyState subMode="reranking" />
                  )}
                </div>

                {/* Right panel: Inputs with History tab */}
                <div className="w-1/2 border-l border-border bg-muted/5 flex flex-col">
                  {/* Tabs */}
                  <div className="flex border-b border-border">
                    <button
                      onClick={() => setRerankRightPanelTab('inputs')}
                      className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                        rerankRightPanelTab === 'inputs'
                          ? 'text-primary border-b-2 border-primary bg-background'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      Inputs
                    </button>
                    <button
                      onClick={() => encoderHistory.filter(h => h.mode === 'reranking').length > 0 && setRerankRightPanelTab('history')}
                      disabled={encoderHistory.filter(h => h.mode === 'reranking').length === 0}
                      className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                        rerankRightPanelTab === 'history'
                          ? 'text-primary border-b-2 border-primary bg-background'
                          : encoderHistory.filter(h => h.mode === 'reranking').length === 0
                            ? 'text-muted-foreground/50 cursor-not-allowed'
                            : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      History {encoderHistory.filter(h => h.mode === 'reranking').length > 0 && `(${encoderHistory.filter(h => h.mode === 'reranking').length})`}
                    </button>
                  </div>

                  {/* Tab content */}
                  <div className="flex-1 flex flex-col min-h-0">
                    {rerankRightPanelTab === 'inputs' ? (
                      /* Inputs tab */
                      <div className="flex-1 flex flex-col min-h-0">
                        {/* Scrollable content area */}
                        <div ref={rerankDocumentsScrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
                          {encoderError && (
                            <div className="text-xs text-destructive">{encoderError}</div>
                          )}

                          {/* Query section */}
                          <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                            <label className="text-xs font-medium text-primary mb-1.5 block">Query</label>
                            <textarea
                              value={rerankQuery}
                              onChange={e => setRerankQuery(e.target.value)}
                              onKeyDown={e => {
                                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                                  e.preventDefault()
                                  handleRerank()
                                }
                              }}
                              disabled={rerankMutation.isPending}
                              placeholder="Enter the query to rank against..."
                              className="w-full px-3 py-2 rounded-lg border border-input bg-background text-sm resize-none"
                              rows={2}
                            />
                          </div>

                          {/* Documents section */}
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <label className="text-xs font-medium text-muted-foreground">
                                Documents to Rank (minimum 2)
                              </label>
                              {rerankDocuments.length > 10 && (
                                <span className="text-xs text-amber-500">
                                  Many documents may slow processing
                                </span>
                              )}
                            </div>

                            {/* Document cards */}
                            <div className="space-y-2">
                              {rerankDocuments.map((doc, index) => (
                                <div
                                  key={index}
                                  className="rounded-lg border border-border bg-card/50 p-2 flex items-start gap-2"
                                >
                                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-muted flex items-center justify-center text-xs font-medium text-muted-foreground">
                                    {index + 1}
                                  </div>
                                  <textarea
                                    value={doc}
                                    onChange={e => {
                                      const newDocs = [...rerankDocuments]
                                      newDocs[index] = e.target.value
                                      setRerankDocuments(newDocs)
                                    }}
                                    onKeyDown={e => {
                                      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                                        e.preventDefault()
                                        handleRerank()
                                      }
                                    }}
                                    disabled={rerankMutation.isPending}
                                    placeholder={`Document ${index + 1}...`}
                                    className="flex-1 px-2 py-1 rounded border border-input bg-background text-sm resize-none min-h-[50px]"
                                    rows={2}
                                  />
                                  <button
                                    onClick={() => {
                                      const newDocs = rerankDocuments.filter((_, i) => i !== index)
                                      setRerankDocuments(newDocs)
                                    }}
                                    disabled={rerankDocuments.length <= 2 || rerankMutation.isPending}
                                    className="flex-shrink-0 p-1 rounded hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
                                    title={rerankDocuments.length <= 2 ? 'Minimum 2 documents required' : 'Remove document'}
                                  >
                                    <FontIcon type="close" className="w-4 h-4 text-muted-foreground" />
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Sticky action buttons */}
                        <div className="flex-shrink-0 p-3 border-t border-border bg-background/95 backdrop-blur-sm">
                          <div className="flex gap-2">
                            <button
                              onClick={() => {
                                setRerankDocuments([...rerankDocuments, ''])
                                // Scroll to bottom after React updates the DOM
                                setTimeout(() => {
                                  rerankDocumentsScrollRef.current?.scrollTo({
                                    top: rerankDocumentsScrollRef.current.scrollHeight,
                                    behavior: 'smooth'
                                  })
                                }, 0)
                              }}
                              disabled={rerankMutation.isPending}
                              className="flex-1 py-2 rounded-lg border border-dashed border-border hover:border-primary hover:bg-primary/5 text-sm text-muted-foreground hover:text-primary transition-colors disabled:opacity-50"
                            >
                              + Add
                            </button>
                            <button
                              onClick={handleRerank}
                              disabled={!rerankQuery.trim() || rerankDocuments.filter(d => d.trim()).length < 2 || rerankMutation.isPending}
                              className="flex-1 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors"
                            >
                              {rerankMutation.isPending ? 'Ranking...' : 'Rank'}
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* History tab */
                      <div className="flex-1 overflow-y-auto p-3 space-y-2">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-muted-foreground">
                            {encoderHistory.filter(h => h.mode === 'reranking').length} items
                          </span>
                          <button
                            onClick={() => setEncoderHistory(prev => prev.filter(h => h.mode !== 'reranking'))}
                            className="text-xs text-muted-foreground hover:text-foreground"
                          >
                            Clear all
                          </button>
                        </div>
                        {encoderHistory.filter(h => h.mode === 'reranking').map(item => (
                          <EncoderHistoryItem
                            key={item.id}
                            item={item}
                            onRerun={() => {
                              setRerankQuery(item.query || '')
                              const docs = item.documents || []
                              setRerankDocuments(docs.length >= 2 ? docs : [...docs, '', ''].slice(0, Math.max(2, docs.length)))
                              setRerankResult(null)
                              setEncoderError(null)
                              setRerankRightPanelTab('inputs')
                            }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Input Area - mode conditional (hidden for encoder/reranking since inputs are in right panel) */}
      {!(modelType === 'encoder' && encoderSubMode === 'reranking') && (
      <div className={inputContainerClasses}>
        {modelType === 'inference' ? (
          /* Inference: Chat input */
          <>
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={e => updateInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={combinedIsSending || (!MOCK_MODE && !chatParams) || unifiedModels.length === 0}
              placeholder={
                unifiedModels.length === 0
                  ? 'Add an inference model to start chatting...'
                  : combinedIsSending
                    ? 'Waiting for response…'
                    : !MOCK_MODE && !chatParams
                      ? 'Select a project to start chatting…'
                      : 'Type a message and press Enter'
              }
              className={textareaClasses}
              aria-label="Message input"
            />
            <div className="flex items-center justify-between">
              {combinedIsSending && (
                <span className="text-xs text-muted-foreground">
                  {USE_PROJECT_CHAT ? 'Sending to project…' : 'Sending…'}
                </span>
              )}
              <FontIcon
                isButton
                type="arrow-filled"
                className={`w-8 h-8 self-end ${!combinedCanSend || (!MOCK_MODE && !chatParams) || unifiedModels.length === 0 ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
                handleOnClick={unifiedModels.length === 0 ? undefined : handleSend}
              />
            </div>
          </>
        ) : modelType === 'anomaly' ? (
          /* Anomaly: Data input */
          <>
            {anomalyError && (
              <div className="text-xs text-destructive mb-2">{anomalyError}</div>
            )}
            <textarea
              ref={anomalyInputRef}
              value={anomalyInput}
              onChange={e => setAnomalyInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleAnomalyDetect()
                }
              }}
              disabled={!selectedAnomalyModel || sortedAnomalyModels.length === 0 || scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending}
              placeholder={
                sortedAnomalyModels.length === 0
                  ? 'Create an anomaly model to start testing...'
                  : scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending
                    ? 'Analyzing...'
                    : 'Paste data row or enter comma-separated values...'
              }
              className={textareaClasses}
              aria-label="Anomaly data input"
            />
            <div className="flex items-center justify-between">
              {(scoreAnomalyMutation.isPending || loadAnomalyMutation.isPending) && (
                <span className="text-xs text-muted-foreground">
                  Analyzing data...
                </span>
              )}
              <FontIcon
                isButton
                type="arrow-filled"
                className={`w-8 h-8 self-end ${!selectedAnomalyModel || !anomalyInput.trim() || sortedAnomalyModels.length === 0 || scoreAnomalyMutation.isPending ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
                handleOnClick={handleAnomalyDetect}
              />
            </div>
          </>
        ) : modelType === 'classifier' ? (
          /* Classifier: Text input */
          <>
            {classifierError && (
              <div className="text-xs text-destructive mb-2">{classifierError}</div>
            )}
            <textarea
              ref={classifierInputRef}
              value={classifierInput}
              onChange={e => setClassifierInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleClassify()
                }
              }}
              disabled={!selectedClassifierModel || sortedClassifierModels.length === 0 || predictClassifierMutation.isPending || loadClassifierMutation.isPending}
              placeholder={
                sortedClassifierModels.length === 0
                  ? 'Create a classifier model to start testing...'
                  : predictClassifierMutation.isPending || loadClassifierMutation.isPending
                    ? 'Classifying...'
                    : 'Enter text to classify...'
              }
              className={textareaClasses}
              aria-label="Classifier text input"
            />
            <div className="flex items-center justify-between">
              {(predictClassifierMutation.isPending || loadClassifierMutation.isPending) && (
                <span className="text-xs text-muted-foreground">
                  Classifying...
                </span>
              )}
              <FontIcon
                isButton
                type="arrow-filled"
                className={`w-8 h-8 self-end ${!selectedClassifierModel || !classifierInput.trim() || sortedClassifierModels.length === 0 || predictClassifierMutation.isPending ? 'text-muted-foreground opacity-50' : 'text-primary'}`}
                handleOnClick={handleClassify}
              />
            </div>
          </>
        ) : modelType === 'document_scanning' ? (
          /* Document Scanning: Status info (auto-scans on drop) */
          <>
            {scanError && (
              <div className="text-xs text-destructive mb-2">{scanError}</div>
            )}
            <div className="flex items-center gap-3 min-h-[40px]">
              {scanDocumentMutation.isPending ? (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-4 h-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    <span>Scanning {scanFile?.name}...</span>
                  </div>
                  {!hasScannedBefore && (
                    <span className="text-xs text-muted-foreground/70">
                      First scan loads OCR models and may take a minute
                    </span>
                  )}
                </div>
              ) : scanFile ? (
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <FontIcon type="data" className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-sm truncate">{scanFile.name}</span>
                  <span className="text-xs text-muted-foreground flex-shrink-0">
                    ({(scanFile.size / 1024).toFixed(1)} KB)
                  </span>
                  <button
                    onClick={() => {
                      setScanFile(null)
                      setScanResults(null)
                      setScanError(null)
                    }}
                    className="text-muted-foreground hover:text-foreground p-1"
                    title="Remove file"
                  >
                    <FontIcon type="close" className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <span className="text-sm text-muted-foreground">
                  Drop an image or PDF above to scan
                </span>
              )}
            </div>
          </>
        ) : (
          /* Encoder: Mode-specific inputs */
          <>
            {encoderError && (
              <div className="text-xs text-destructive mb-2">{encoderError}</div>
            )}
            {encoderSubMode === 'embedding' ? (
              <>
                {/* Two-column layout for side-by-side text inputs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                  {/* Text A Panel */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Text A</label>
                    <textarea
                      value={embeddingTextA}
                      onChange={e => setEmbeddingTextA(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          e.preventDefault()
                          handleEmbedding()
                        }
                      }}
                      disabled={createEmbeddingsMutation.isPending}
                      placeholder="Paste or type text..."
                      className={`${textareaClasses} min-h-[100px]`}
                      aria-label="First text input"
                    />
                  </div>
                  {/* Text B Panel */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Text B</label>
                    <textarea
                      value={embeddingTextB}
                      onChange={e => setEmbeddingTextB(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          e.preventDefault()
                          handleEmbedding()
                        }
                      }}
                      disabled={createEmbeddingsMutation.isPending}
                      placeholder="Paste or type text..."
                      className={`${textareaClasses} min-h-[100px]`}
                      aria-label="Second text input"
                    />
                  </div>
                </div>
                {/* Centered submit button */}
                <div className="flex justify-center">
                  <button
                    onClick={handleEmbedding}
                    disabled={!embeddingTextA.trim() || !embeddingTextB.trim() || createEmbeddingsMutation.isPending}
                    className="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors"
                  >
                    {createEmbeddingsMutation.isPending ? 'Comparing...' : 'Compare Similarity'}
                  </button>
                </div>
              </>
            ) : null /* Reranking inputs are in the right panel above */}
          </>
        )}
      </div>
      )}
    </div>
  )
}

interface TestChatMessageProps {
  message: ChatboxMessage
  showReferences: boolean
  allowRanking: boolean
  showPrompts?: boolean
  showThinking?: boolean
  lastUserInput?: string
  showGenSettings?: boolean
}

export function TestChatMessage({
  message,
  showReferences,
  allowRanking,
  showPrompts,
  // showThinking - no longer used here; thinking is always shown if present in message
  lastUserInput,
  showGenSettings,
}: TestChatMessageProps) {
  const isUser = message.type === 'user'
  const isAssistant = message.type === 'assistant'
  const [thumb, setThumb] = useState<null | 'up' | 'down'>(null)
  const [showExpected, setShowExpected] = useState<boolean>(false)
  const [openPrompts, setOpenPrompts] = useState<boolean>(true)
  // Thinking UI state - collapsed by default, user can expand
  const [userExpandedThinking, setUserExpandedThinking] = useState<boolean>(false)

  // Extract optional <think> ... </think> section from assistant content
  // If there is no closing tag, assume thinking continues to end of content
  let thinkingFromTags = ''
  let contentWithoutThinking = message.content
  let isThinkingInProgress = false
  if (
    isAssistant &&
    typeof message.content === 'string' &&
    message.content.includes('<think>')
  ) {
    const start = message.content.indexOf('<think>') + 7
    const end = message.content.indexOf('</think>')
    if (end !== -1) {
      thinkingFromTags = message.content.slice(start, end).trim()
      contentWithoutThinking = (
        message.content.slice(0, message.content.indexOf('<think>')) +
        message.content.slice(end + 8)
      ).trim()
    } else {
      // No closing tag yet - thinking is in progress
      isThinkingInProgress = true
      thinkingFromTags = message.content.slice(start).trim()
      contentWithoutThinking = message.content
        .slice(0, message.content.indexOf('<think>'))
        .trim()
    }
  }

  // Ref for auto-scrolling thinking box
  const thinkingBoxRef = useRef<HTMLDivElement>(null)

  // Auto-scroll thinking box to bottom as content streams in
  useEffect(() => {
    if (isThinkingInProgress && thinkingBoxRef.current) {
      thinkingBoxRef.current.scrollTop = thinkingBoxRef.current.scrollHeight
    }
  }, [thinkingFromTags, isThinkingInProgress])

  // Remove raw XML tags from display
  if (isAssistant && typeof contentWithoutThinking === 'string') {
    // Remove complete tool_call blocks
    contentWithoutThinking = contentWithoutThinking.replace(
      /<tool_call>[\s\S]*?<\/tool_call>/g,
      ''
    )
    // Remove unclosed tool_call tags (streaming in progress)
    contentWithoutThinking = contentWithoutThinking.replace(
      /<tool_call>[\s\S]*$/g,
      ''
    )
    // Remove orphaned closing tags
    contentWithoutThinking = contentWithoutThinking.replace(
      /<\/tool_call>/g,
      ''
    )
    // Remove any remaining think tags (opening and closing)
    contentWithoutThinking = contentWithoutThinking.replace(
      /<think>[\s\S]*?<\/think>/g,
      ''
    )
    contentWithoutThinking = contentWithoutThinking.replace(/<\/?think>/g, '')
    contentWithoutThinking = contentWithoutThinking.trim()
  }

  // Load persisted thumb for this message
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const key = `lf_thumb_${message.id}`
      const saved = localStorage.getItem(key)
      if (saved === 'up' || saved === 'down') setThumb(saved)
    } catch {}
  }, [message.id])

  // Handle user toggle of thinking section
  const handleThinkingToggle = useCallback(() => {
    setUserExpandedThinking(prev => !prev)
  }, [])

  const onThumb = useCallback(
    (kind: 'up' | 'down') => {
      setThumb(prev => {
        const next = prev === kind ? null : kind
        try {
          const key = `lf_thumb_${message.id}`
          if (next) localStorage.setItem(key, next)
          else localStorage.removeItem(key)
        } catch {}
        return next
      })
    },
    [message.id]
  )

  return (
    <div
      className={`flex flex-col ${isUser ? 'self-end' : ''}`}
      style={{ maxWidth: isUser ? 'min(88%, 900px)' : 'min(92%, 900px)' }}
    >
      <div
        className={
          isUser
            ? 'px-4 py-2.5 rounded-lg bg-primary/10 text-foreground leading-snug'
            : isAssistant
              ? 'px-0 md:px-0 text-[15px] md:text-base leading-relaxed text-foreground/90'
              : 'px-4 py-3 rounded-lg bg-muted text-foreground'
        }
      >
        {message.isLoading && isAssistant ? (
          <TypingDots label="Loading model (may take a moment if downloading)" />
        ) : message.metadata?.isTest && isUser ? (
          <div className="whitespace-pre-wrap">
            <div className="mb-2">
              <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
                Test input
              </Badge>
            </div>
            {message.content}
          </div>
        ) : (
          <>
            {/* Thinking card - always show if message has thinking content (toggle only affects new messages) */}
            {isAssistant && thinkingFromTags && (
              <div className={`mb-2 rounded-md border border-border bg-card/40 overflow-hidden ${isThinkingInProgress ? 'animate-pulse' : ''}`}>
                <button
                  type="button"
                  onClick={isThinkingInProgress ? undefined : handleThinkingToggle}
                  className={`w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground ${isThinkingInProgress ? '' : 'hover:bg-accent/40 cursor-pointer'}`}
                  aria-expanded={isThinkingInProgress || userExpandedThinking}
                  disabled={isThinkingInProgress}
                >
                  <span className="font-medium flex items-center gap-2">
                    <span className="text-purple-400">💭</span>
                    {isThinkingInProgress ? 'Thinking...' : 'Thinking steps'}
                  </span>
                  {!isThinkingInProgress && (
                    <span className="text-[11px]">
                      {userExpandedThinking ? 'Hide' : 'Show'}
                    </span>
                  )}
                </button>
                {/* Content area - fixed height during streaming, collapsible when done */}
                <div
                  ref={thinkingBoxRef}
                  className={`border-t border-border overflow-hidden transition-all duration-300 ease-in-out ${
                    isThinkingInProgress
                      ? 'h-[150px] overflow-y-scroll scrollbar-none'
                      : ''
                  }`}
                  style={
                    isThinkingInProgress
                      ? { pointerEvents: 'none' }
                      : {
                          maxHeight: userExpandedThinking ? '2000px' : '0px',
                          opacity: userExpandedThinking ? 1 : 0,
                        }
                  }
                >
                  <div className="px-3 py-3 text-sm text-muted-foreground/70">
                    <div className="prose prose-sm max-w-none leading-relaxed prose-p:my-3 prose-li:my-1 prose-ul:my-2 prose-ol:my-2 prose-headings:my-3 prose-headings:font-medium prose-pre:my-2 [&_*]:text-muted-foreground/70 space-y-3">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {thinkingFromTags
                          .replace(/\n{3,}/g, '\n\n')
                          .replace(/([.!?])\n(?=[A-Z])/g, '$1\n\n')}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Final answer content (without <think> … </think>) */}
            <div className={
              isUser
                ? 'prose prose-sm dark:prose-invert max-w-none leading-snug prose-p:my-0 [&>*]:mb-0 [&>*:last-child]:mb-0'
                : 'prose prose-sm dark:prose-invert max-w-none leading-relaxed prose-p:my-4 prose-li:my-1 prose-ul:my-4 prose-ol:my-4 prose-headings:my-4 prose-pre:my-3 [&>*]:mb-4'
            }>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {contentWithoutThinking.replace(/\n{3,}/g, '\n\n')}
              </ReactMarkdown>
            </div>
          </>
        )}
      </div>

      {/* Assistant footer actions - hidden during streaming */}
      {isAssistant && !message.isStreaming && !message.isLoading && (
        <div className="mt-2 flex items-center gap-3 text-muted-foreground">
          {allowRanking && (
            <>
              <ThumbButton
                kind="up"
                active={thumb === 'up'}
                onClick={() => onThumb('up')}
              />
              <ThumbButton
                kind="down"
                active={thumb === 'down'}
                onClick={() => {
                  const next = thumb === 'down' ? null : 'down'
                  onThumb('down')
                  // Show a subtle troubleshoot nudge when giving thumbs down
                  if (next === 'down') {
                    try {
                      // Lightweight toast via alert-style for now: inline nudge button below
                    } catch {}
                  }
                }}
              />
            </>
          )}
          <ActionLink
            label="Diagnose"
            className={
              thumb === 'down'
                ? 'text-xs text-teal-500 hover:text-teal-400 hover:underline font-medium'
                : undefined
            }
            onClick={() => {
              // brief local visual loading cue by dimming text
              const el = document.activeElement as HTMLElement | null
              if (el) el.blur()
              window.dispatchEvent(
                new CustomEvent('lf-diagnose', {
                  detail: {
                    source: 'message_action',
                    responseText: message.content,
                  },
                })
              )
            }}
          />
          <ActionLink
            label="Retry"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent('lf-chat-retry', { detail: { id: message.id } })
              )
            }
          />
          <ActionLink
            label="Use as prompt"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent('lf-chat-use-as-prompt', {
                  detail: { content: message.content },
                })
              )
            }
          />
        </div>
      )}

      {/* Generation settings, compact */}
      {isAssistant &&
        showGenSettings &&
        message.metadata &&
        (() => {
          const gen = (message.metadata as any)?.generation || null
          if (!gen) return null
          return (
            <div className="mt-1 text-[11px] text-muted-foreground">
              T={gen?.temperature ?? '—'} • top‑p={gen?.topP ?? '—'} • max=
              {gen?.maxTokens ?? '—'}
              {typeof gen?.seed !== 'undefined' ? (
                <> • seed={String(gen?.seed)}</>
              ) : null}
            </div>
          )
        })()}

      {/* References */}
      {showReferences &&
        isAssistant &&
        Array.isArray(message.sources) &&
        message.sources.length > 0 && <References sources={message.sources} />}

      {/* Test result block */}
      {isAssistant && message.metadata?.testResult && (
        <div className="mt-3 rounded-md border border-border bg-card/40 p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
                Test result
              </Badge>
              <span
                className="text-[11px] text-muted-foreground"
                title="This is a simple lexical overlap metric and may not reflect semantic correctness."
              >
                experimental
              </span>
              <span
                className={`px-2 py-0.5 rounded-2xl text-xs ${
                  (message.metadata.testResult.score ?? 0) >= 95
                    ? 'bg-teal-300 text-black'
                    : (message.metadata.testResult.score ?? 0) >= 75
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-amber-300 text-black'
                }`}
              >
                {message.metadata.testResult.score}% match
              </span>
            </div>
            <button
              className="text-xs underline text-muted-foreground"
              onClick={() => setShowExpected(s => !s)}
            >
              {showExpected ? 'Hide expected' : 'View expected'}
            </button>
          </div>
          {typeof message.metadata.testResult.score === 'number' &&
            message.metadata.testResult.score < 80 && (
              <div className="mt-2 text-xs">
                <button
                  type="button"
                  className="px-2 py-0.5 rounded border border-teal-500/50 text-teal-700 hover:bg-teal-500/10 dark:text-teal-300"
                  onClick={() =>
                    window.dispatchEvent(
                      new CustomEvent('lf-diagnose', {
                        detail: {
                          source: 'low_score',
                          testId: message.metadata?.testId,
                          testName: message.metadata?.testName,
                          input: lastUserInput || '',
                          expected:
                            message.metadata?.testResult?.expected || '',
                          matchScore: message.metadata?.testResult?.score,
                        },
                      })
                    )
                  }
                >
                  Diagnose
                </button>
              </div>
            )}
          <div className="mt-2 text-xs text-muted-foreground flex items-center gap-4">
            <div>{message.metadata.testResult.latencyMs}ms result</div>
            <div>
              {message.metadata.testResult.tokenUsage.total} tokens
              <span className="opacity-60">
                {' '}
                (p {message.metadata.testResult.tokenUsage.prompt} / c{' '}
                {message.metadata.testResult.tokenUsage.completion})
              </span>
            </div>
          </div>
          {showExpected && message.metadata.testResult.expected && (
            <div className="mt-2 p-2 rounded bg-muted text-xs whitespace-pre-wrap">
              {message.metadata.testResult.expected}
            </div>
          )}
        </div>
      )}

      {/* Optional helper cards */}
      {isAssistant &&
        showPrompts &&
        Array.isArray(message.metadata?.prompts) && (
          <div className="mt-2 rounded-md border border-border bg-card/40">
            <button
              type="button"
              onClick={() => setOpenPrompts(o => !o)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground rounded-t-md hover:bg-accent/40"
              aria-expanded={openPrompts}
            >
              <span className="font-medium">
                Prompts sent ({message.metadata.prompts.length})
              </span>
              <span className="text-[11px]">
                {openPrompts ? 'Hide' : 'Show'}
              </span>
            </button>
            {openPrompts && (
              <div className="divide-y divide-border">
                {message.metadata.prompts.map((p: string, i: number) => (
                  <div
                    key={i}
                    className="px-3 py-2 text-sm whitespace-pre-wrap"
                  >
                    {p}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      {/* Metadata thinking - always show if exists (toggle only affects new messages) */}
      {isAssistant &&
        Array.isArray(message.metadata?.thinking) && (
          <div className="mt-2 rounded-md border border-border bg-card/40 overflow-hidden">
            <button
              type="button"
              onClick={handleThinkingToggle}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground rounded-t-md hover:bg-accent/40"
              aria-expanded={userExpandedThinking}
            >
              <span className="font-medium flex items-center gap-2">
                <span className="text-purple-400">💭</span>
                Thinking steps
              </span>
              <span className="text-[11px]">
                {userExpandedThinking ? 'Hide' : 'Show'}
              </span>
            </button>
            <div
              className="transition-all duration-300 ease-in-out overflow-hidden"
              style={{
                maxHeight: userExpandedThinking ? '2000px' : '0px',
                opacity: userExpandedThinking ? 1 : 0,
              }}
            >
              <ol className="px-5 py-3 text-sm list-decimal marker:text-muted-foreground/70 space-y-2">
                {message.metadata.thinking.map((step: string, i: number) => (
                  <li key={i} className="leading-relaxed">
                    {step}
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
    </div>
  )
}

function ThumbButton({
  kind,
  active,
  onClick,
}: {
  kind: 'up' | 'down'
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 group cursor-pointer rounded-sm hover:opacity-80"
    >
      <FontIcon
        type={
          kind === 'up'
            ? active
              ? 'thumbs-up-filled'
              : 'thumbs-up'
            : active
              ? 'thumbs-down-filled'
              : 'thumbs-down'
        }
        className={`w-5 h-5 ${active ? 'text-teal-500' : 'text-muted-foreground group-hover:text-foreground'}`}
      />
    </button>
  )
}

// Copy button removed

function ActionLink({
  label,
  onClick,
  className,
}: {
  label: string
  onClick: () => void
  className?: string
}) {
  return (
    <button
      onClick={onClick}
      className={className || 'text-xs hover:underline'}
    >
      {label}
    </button>
  )
}

function References({ sources }: { sources: any[] }) {
  const [open, setOpen] = useState<boolean>(true)
  const count = sources.length
  return (
    <div className="mt-2 rounded-md border border-border bg-card/40">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:bg-accent/40 rounded-t-md focus:outline-none focus:ring-2 focus:ring-primary/60"
        aria-expanded={open}
        aria-controls={`references-panel`}
      >
        <span className="font-medium">References ({count})</span>
        <span className="text-[11px]">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <div id="references-panel" className="divide-y divide-border">
          {sources.map((s, idx) => (
            <div key={idx} className="px-3 py-2">
              {s.content && (
                <div className="text-sm text-foreground whitespace-pre-wrap line-clamp-2">
                  {s.content}
                </div>
              )}
              <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                <div className="truncate">
                  {s.source || s.metadata?.source || 'source'}
                </div>
                {typeof s.score === 'number' && (
                  <span className="ml-2 text-[11px]">
                    {(s.score * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TypingDots({ label = 'Loading model' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 opacity-80">
      <span>{label}</span>
      <span className="animate-pulse">.</span>
      <span className="animate-pulse [animation-delay:150ms]">.</span>
      <span className="animate-pulse [animation-delay:300ms]">.</span>
    </span>
  )
}
