/**
 * Demo Project Creation Modal
 * Beautiful, educational workflow showing API calls in real-time
 */

import React, { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '../ui/dialog'
import { getFileBasedDemos, type FileBasedDemo } from '../../config/demos'
import { useDemoWorkflow, DemoStep, ApiCall, ProcessingResult } from '../../hooks/useDemoWorkflow'
import { CheckCircle2, Loader2, XCircle, ChevronDown, ChevronRight, Copy, Check } from 'lucide-react'

interface DemoModalProps {
  isOpen: boolean
  onClose: () => void
  namespace: string
}

function DemoSelector({ onSelect }: { onSelect: (demo: FileBasedDemo) => void }) {
  const fileBasedDemos = getFileBasedDemos()
  return (
    <div className="space-y-4">

      <div className="grid gap-3">
        {fileBasedDemos.map(demo => (
          <button
            key={demo.id}
            onClick={() => onSelect(demo)}
            className="group relative flex items-start gap-4 rounded-lg border border-input bg-card p-4 text-left transition-all hover:border-primary hover:bg-accent/50"
          >
            <div className="text-4xl">{demo.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2 mb-1">
                <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
                  {demo.displayName}
                </h3>
                <span className="text-xs text-muted-foreground whitespace-nowrap">
                  {demo.estimatedTime}
                </span>
              </div>
              <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                {demo.description}
              </p>
              <div className="flex items-center gap-2 text-xs">
                <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                  {demo.category}
                </span>
                <span className="text-muted-foreground">
                  {demo.files.length} file{demo.files.length !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function ApiCallItem({ call }: { call: ApiCall }) {
  const [isExpanded, setIsExpanded] = useState(false)

  const methodColors: Record<string, string> = {
    GET: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
    POST: 'bg-green-500/10 text-green-600 border-green-500/20',
    PUT: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20',
    DELETE: 'bg-red-500/10 text-red-600 border-red-500/20'
  }

  const statusIcons = {
    pending: <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />,
    success: <CheckCircle2 className="w-3 h-3 text-green-500" />,
    error: <XCircle className="w-3 h-3 text-destructive" />
  }

  return (
    <div className="border-l-2 border-muted pl-4 py-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-start gap-3 text-left group"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`px-2 py-0.5 text-xs font-mono font-semibold rounded border ${
                methodColors[call.method] || 'bg-gray-500/10 text-gray-600'
              }`}
            >
              {call.method}
            </span>
            {statusIcons[call.status]}
            {call.duration && (
              <span className="text-xs text-muted-foreground">
                {call.duration}ms
              </span>
            )}
          </div>
          <p className="text-sm text-foreground group-hover:text-primary transition-colors">
            {call.description}
          </p>
        </div>
      </button>

      {isExpanded && (
        <div className="mt-2 ml-7 p-2 rounded bg-muted/50 text-xs font-mono">
          <div className="flex items-center justify-between mb-1">
            <span className="text-muted-foreground">Endpoint:</span>
            <span className="text-foreground">{call.endpoint}</span>
          </div>
          {call.statusCode && (
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Status:</span>
              <span className="text-foreground">{call.statusCode}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function CopyableQuestion({ question, index }: { question: string; index: number }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(question)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <li>
      <button
        onClick={handleCopy}
        className="w-full group flex items-center gap-3 p-3 rounded-lg border border-border bg-card hover:border-primary hover:bg-accent/50 transition-all text-left"
      >
        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold shrink-0">
          {index}
        </span>
        <span className="flex-1 text-sm text-foreground">{question}</span>
        {copied ? (
          <Check className="w-4 h-4 text-green-600 shrink-0" />
        ) : (
          <Copy className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
        )}
      </button>
    </li>
  )
}

function WorkflowProgress({
  demo,
  currentStep,
  progress,
  error,
  apiCalls,
  projectName,
  processingResult,
  onStartChat,
  onRetry
}: {
  demo: FileBasedDemo
  currentStep: DemoStep
  progress: number
  error: string | null
  apiCalls: ApiCall[]
  projectName: string | null
  processingResult: ProcessingResult | null
  onStartChat: () => void
  onRetry: () => void
}) {
  const [showApiCalls, setShowApiCalls] = useState(false)

  const stepLabels: Record<DemoStep, string> = {
    idle: 'Ready',
    fetching_config: 'Fetching Configuration',
    downloading_model: 'Downloading AI Model',
    creating_project: 'Creating Project',
    uploading_files: 'Uploading Files',
    processing_dataset: 'Processing Dataset',
    completed: 'Completed',
    error: 'Error'
  }

  const isCompleted = currentStep === 'completed'
  const isError = currentStep === 'error'

  // Failure state
  if (isError) {
    // Determine if this is a dataset processing error and provide helpful context
    const isDatasetError = error?.toLowerCase().includes('dataset') || 
                          error?.toLowerCase().includes('processing')
    
    const isProjectDeletedError = error?.toLowerCase().includes('not found') ||
                                  error?.toLowerCase().includes('deleted') ||
                                  error?.toLowerCase().includes('stale')
    
    let helpText = null
    if (isProjectDeletedError) {
      helpText = "This usually happens when there are background tasks from recently deleted projects. Simply click 'Try Again' to create a fresh demo project."
    } else if (isDatasetError) {
      helpText = "Check if the RAG worker is running properly. You can also try refreshing the page."
    }

    return (
      <div className="space-y-4">
        {/* Error header */}
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
          <div className="flex items-start gap-3">
            <XCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-destructive mb-1">Creation Failed</p>
              <p className="text-sm text-destructive/90 mb-2">{error}</p>
              {helpText && (
                <div className="mt-3 pt-3 border-t border-destructive/20">
                  <p className="text-xs text-muted-foreground">
                    💡 <span className="font-medium">What to do:</span> {helpText}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-2">
          <button
            onClick={onRetry}
            className="w-full px-6 py-3 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-medium"
          >
            Try Again
          </button>
          <button
            onClick={onStartChat}
            className="w-full px-6 py-3 rounded-lg border border-border bg-card hover:bg-accent transition-colors font-medium"
          >
            Go to Home
          </button>
        </div>
      </div>
    )
  }

  // Success state
  if (isCompleted) {
    return (
      <div className="space-y-3">
        {/* Success header - compact */}
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-green-500/10 border border-green-500/20">
          <CheckCircle2 className="w-5 h-5 text-green-600 shrink-0" />
          <p className="text-sm font-medium text-green-600">
            Your project is ready!
          </p>
        </div>

        {/* Processing stats - compact */}
        {processingResult && (
          <div className="flex items-center gap-4 px-4 py-2 rounded-lg bg-accent/30 border border-accent text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Files Processed:</span>
              <span className="font-semibold text-foreground">{processingResult.totalFiles}</span>
            </div>
            <div className="h-4 w-px bg-border" />
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Strategy:</span>
              <span className="font-medium text-foreground">{processingResult.parsers[0] || 'default'}</span>
            </div>
          </div>
        )}

        {/* Sample questions - copyable */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-foreground">
            Try these sample questions to see professional answers:
          </p>
          <ul className="space-y-1.5">
            {demo.sampleQuestions.slice(0, 4).map((q, i) => (
              <CopyableQuestion key={i} question={q} index={i + 1} />
            ))}
          </ul>
        </div>

        {/* Start Chatting button - full width */}
        <button
          onClick={onStartChat}
          className="w-full px-6 py-3 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-medium"
        >
          Start Chatting
        </button>
      </div>
    )
  }

  // Loading state
  return (
    <div className="space-y-4">
      {/* Header with demo info */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-accent/50 border border-accent">
        <div className="text-3xl">{demo.icon}</div>
        <div className="flex-1">
          <h4 className="font-semibold text-foreground">{demo.displayName}</h4>
          {projectName && (
            <p className="text-sm text-muted-foreground">
              Creating: <span className="font-mono text-primary">{projectName}</span>
            </p>
          )}
        </div>
      </div>

      {/* Current step with progress */}
      <div className="space-y-3">
        {/* Current step indicator */}
        <div className="flex items-center gap-3">
          {error ? (
            <XCircle className="w-5 h-5 text-destructive" />
          ) : (
            <Loader2 className="w-5 h-5 text-primary animate-spin" />
          )}
          <span className={`text-sm font-medium ${error ? 'text-destructive' : 'text-primary'}`}>
            {error ? 'Error' : stepLabels[currentStep]}
          </span>
          <span className="text-sm text-muted-foreground ml-auto">
            {Math.round(progress)}%
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ease-out ${error ? 'bg-destructive' : 'bg-primary'}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
          <div className="flex items-start gap-2">
            <XCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-destructive mb-1">Error</p>
              <p className="text-sm text-destructive/90">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Collapsed API Calls */}
      {apiCalls.length > 0 && (
        <div className="border border-accent rounded-lg overflow-hidden">
          <button
            onClick={() => setShowApiCalls(!showApiCalls)}
            className="w-full flex items-center justify-between p-3 bg-accent/30 hover:bg-accent/50 transition-colors"
          >
            <span className="text-sm font-medium text-foreground">
              API Calls <span className="text-muted-foreground">({apiCalls.length})</span>
            </span>
            {showApiCalls ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
          </button>
          
          {showApiCalls && (
            <div className="p-3 space-y-1 max-h-64 overflow-y-auto">
              {apiCalls.map(call => (
                <ApiCallItem key={call.id} call={call} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function DemoModal({ isOpen, onClose, namespace, autoStartDemoId }: DemoModalProps & { autoStartDemoId?: string | null }) {
  const [selectedDemo, setSelectedDemo] = useState<FileBasedDemo | null>(null)
  const { currentStep, progress, error, apiCalls, projectName, processingResult, startDemo, reset } =
    useDemoWorkflow()

  // Auto-start demo if provided
  React.useEffect(() => {
    if (isOpen && autoStartDemoId && !selectedDemo && currentStep === 'idle') {
      const fileBasedDemos = getFileBasedDemos()
      const demo = fileBasedDemos.find(d => d.id === autoStartDemoId)
      if (demo) {
        handleSelectDemo(demo)
      }
    }
  }, [isOpen, autoStartDemoId, selectedDemo, currentStep])


  const handleSelectDemo = (demo: FileBasedDemo) => {
    setSelectedDemo(demo)
    startDemo(demo, namespace)
  }

  const handleClose = () => {
    if (currentStep === 'completed' || currentStep === 'error') {
      reset()
      setSelectedDemo(null)
    }
    onClose()
  }

  const handleRetry = () => {
    if (selectedDemo) {
      reset()
      startDemo(selectedDemo, namespace)
    }
  }

  const canClose = currentStep === 'idle' || currentStep === 'completed' || currentStep === 'error'

  // Dynamic title and description based on state
  const getModalContent = () => {
    if (!selectedDemo) {
      return {
        title: 'Create Demo Project',
        description: 'Choose a demo to instantly create a fully-configured AI assistant with knowledge base'
      }
    }
    if (currentStep === 'completed') {
      return {
        title: 'Demo project created!',
        description: 'Your AI assistant is ready to use with the demo knowledge base'
      }
    }
    if (currentStep === 'error') {
      return {
        title: 'Project creation failed',
        description: 'Something went wrong while creating your demo project'
      }
    }
    return {
      title: 'Creating demo project...',
      description: 'Watch your demo project being created in real-time'
    }
  }

  const { title, description } = getModalContent()


  return (
    <Dialog open={isOpen} onOpenChange={canClose ? handleClose : undefined}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto" hideCloseButton={!canClose}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div>
          {!selectedDemo ? (
            <DemoSelector onSelect={handleSelectDemo} />
          ) : (
            <WorkflowProgress
              demo={selectedDemo}
              currentStep={currentStep}
              progress={progress}
              error={error}
              apiCalls={apiCalls}
              projectName={projectName}
              processingResult={processingResult}
              onStartChat={handleClose}
              onRetry={handleRetry}
            />
          )}
        </div>

        {/* Only show close button when not completed or error (those states have their own buttons) */}
        {canClose && currentStep !== 'completed' && currentStep !== 'error' && (
          <div className="flex justify-end pt-4 border-t">
            <button
              onClick={handleClose}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
            >
              Close
            </button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
