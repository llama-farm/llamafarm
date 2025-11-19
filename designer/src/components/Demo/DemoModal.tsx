/**
 * Demo Project Creation Modal
 * Beautiful, educational workflow showing API calls in real-time
 */

import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '../ui/dialog'
import { AVAILABLE_DEMOS, DemoConfig } from '../../config/demos'
import { useDemoWorkflow, DemoStep, ApiCall, ProcessingResult } from '../../hooks/useDemoWorkflow'
import { CheckCircle2, Circle, Loader2, XCircle, ChevronDown, ChevronRight } from 'lucide-react'

interface DemoModalProps {
  isOpen: boolean
  onClose: () => void
  namespace: string
}

function DemoSelector({ onSelect }: { onSelect: (demo: DemoConfig) => void }) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Choose a demo to instantly create a fully-configured AI assistant with knowledge base:
      </p>

      <div className="grid gap-3">
        {AVAILABLE_DEMOS.map(demo => (
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

function StepIndicator({
  step,
  currentStep,
  lastValidStep
}: {
  step: DemoStep
  currentStep: DemoStep
  lastValidStep: DemoStep
}) {
  const steps: DemoStep[] = [
    'fetching_config',
    'creating_project',
    'uploading_files',
    'processing_dataset',
    'completed'
  ]

  const stepLabels: Record<DemoStep, string> = {
    idle: 'Ready',
    fetching_config: 'Fetching Configuration',
    creating_project: 'Creating Project',
    uploading_files: 'Uploading Files',
    processing_dataset: 'Processing Dataset',
    completed: 'Completed',
    error: 'Error'
  }

  // When error occurs, use lastValidStep to determine progress
  const effectiveCurrentStep = currentStep === 'error' ? lastValidStep : currentStep
  const currentIndex = steps.indexOf(effectiveCurrentStep)
  const stepIndex = steps.indexOf(step)

  const isActive = step === effectiveCurrentStep
  const isCompleted = stepIndex < currentIndex || currentStep === 'completed'
  const isError = currentStep === 'error' && isActive

  return (
    <div className="flex items-center gap-2">
      {isCompleted ? (
        <CheckCircle2 className="w-4 h-4 text-green-500" />
      ) : isError ? (
        <XCircle className="w-4 h-4 text-destructive" />
      ) : isActive ? (
        <Loader2 className="w-4 h-4 text-primary animate-spin" />
      ) : (
        <Circle className="w-4 h-4 text-muted-foreground/50" />
      )}
      <span
        className={`text-sm ${
          isActive
            ? 'text-primary font-medium'
            : isCompleted
            ? 'text-green-600'
            : 'text-muted-foreground'
        }`}
      >
        {stepLabels[step]}
      </span>
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

function WorkflowProgress({
  demo,
  currentStep,
  lastValidStep,
  progress,
  error,
  apiCalls,
  projectName,
  processingResult
}: {
  demo: DemoConfig
  currentStep: DemoStep
  lastValidStep: DemoStep
  progress: number
  error: string | null
  apiCalls: ApiCall[]
  projectName: string | null
  processingResult: ProcessingResult | null
}) {
  const steps: DemoStep[] = [
    'fetching_config',
    'creating_project',
    'uploading_files',
    'processing_dataset',
    'completed'
  ]

  return (
    <div className="space-y-6">
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

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Overall Progress</span>
          <span className="font-medium text-foreground">{Math.round(progress)}%</span>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        <h5 className="text-sm font-medium text-foreground">Steps</h5>
        <div className="space-y-1">
          {steps.map(step => (
            <StepIndicator key={step} step={step} currentStep={currentStep} lastValidStep={lastValidStep} />
          ))}
        </div>
      </div>

      {/* API Calls */}
      {apiCalls.length > 0 && (
        <div className="space-y-2">
          <h5 className="text-sm font-medium text-foreground">
            API Calls <span className="text-muted-foreground">({apiCalls.length})</span>
          </h5>
          <div className="max-h-64 overflow-y-auto space-y-1 pr-2">
            {apiCalls.map(call => (
              <ApiCallItem key={call.id} call={call} />
            ))}
          </div>
        </div>
      )}

      {/* Error */}
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

      {/* Completion message */}
      {currentStep === 'completed' && (
        <div className="space-y-4">
          {/* Success header */}
          <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-6 h-6 text-green-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-lg font-semibold text-green-600 mb-1">
                  Your project is ready!
                </p>
                <p className="text-sm text-muted-foreground">
                  Test it by chatting with the model and the {demo.displayName.toLowerCase()} in RAG.
                </p>
              </div>
            </div>
          </div>

          {/* Processing stats */}
          {processingResult && (
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-accent/50 border border-accent">
                <p className="text-xs text-muted-foreground mb-1">Files Processed</p>
                <p className="text-2xl font-semibold text-foreground">
                  {processingResult.totalFiles}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-accent/50 border border-accent">
                <p className="text-xs text-muted-foreground mb-1">Strategy</p>
                <p className="text-sm font-medium text-foreground truncate" title={processingResult.parsers[0]}>
                  {processingResult.parsers[0] || 'default'}
                </p>
              </div>
            </div>
          )}

          {/* Sample questions */}
          <div className="p-4 rounded-lg bg-primary/5 border border-primary/10">
            <p className="text-sm font-medium text-foreground mb-3">
              Try these sample questions to see professional answers:
            </p>
            <ul className="space-y-2">
              {demo.sampleQuestions.slice(0, 4).map((q, i) => (
                <li key={i} className="text-sm text-foreground flex items-start gap-2">
                  <span className="text-primary font-semibold mt-0.5 shrink-0">{i + 1}.</span>
                  <span className="italic text-muted-foreground">&ldquo;{q}&rdquo;</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Close to start chatting */}
          <div className="text-center p-3 rounded-lg bg-accent/30 border border-accent/50">
            <p className="text-sm text-muted-foreground">
              👆 Close this modal to start chatting
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export function DemoModal({ isOpen, onClose, namespace }: DemoModalProps) {
  const [selectedDemo, setSelectedDemo] = useState<DemoConfig | null>(null)
  const { currentStep, lastValidStep, progress, error, apiCalls, projectName, processingResult, startDemo, reset } =
    useDemoWorkflow()

  const handleSelectDemo = (demo: DemoConfig) => {
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

  const canClose = currentStep === 'idle' || currentStep === 'completed' || currentStep === 'error'

  return (
    <Dialog open={isOpen} onOpenChange={canClose ? handleClose : undefined}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create Demo Project</DialogTitle>
          <DialogDescription>
            {!selectedDemo
              ? 'Choose a pre-configured demo to explore LlamaFarm capabilities'
              : 'Watch your demo project being created in real-time'}
          </DialogDescription>
        </DialogHeader>

        <div className="mt-4">
          {!selectedDemo ? (
            <DemoSelector onSelect={handleSelectDemo} />
          ) : (
            <WorkflowProgress
              demo={selectedDemo}
              currentStep={currentStep}
              lastValidStep={lastValidStep}
              progress={progress}
              error={error}
              apiCalls={apiCalls}
              projectName={projectName}
              processingResult={processingResult}
            />
          )}
        </div>

        {canClose && (
          <div className="flex justify-end pt-4 border-t">
            <button
              onClick={handleClose}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
            >
              {currentStep === 'completed' ? 'Start Chatting' : 'Close'}
            </button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
