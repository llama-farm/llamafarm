/**
 * Onboarding wizard component
 * Takes over the entire dashboard content area
 * Guides users through setup questions and generates a personalized checklist
 */

import { useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Loader2, ArrowLeft, ArrowRight, X } from 'lucide-react'
import { useOnboardingContext } from '../../contexts/OnboardingContext'
import { WizardProgress } from './WizardProgress'
import { ProjectTypeSelector } from './ProjectTypeSelector'
import { DataStatusSelector } from './DataStatusSelector'
import { DeployTargetSelector } from './DeployTargetSelector'
import { ExperienceSelector } from './ExperienceSelector'

interface OnboardingWizardProps {
  className?: string
}

export function OnboardingWizard({ className }: OnboardingWizardProps) {
  const {
    state,
    canProceed,
    nextStep,
    prevStep,
    skipWizard,
    completeWizard,
    setProjectType,
    setDataStatus,
    setSelectedSampleDataset,
    setDeployTarget,
    setExperienceLevel,
    setUploadedFiles,
    setDatasetName,
    addActualFiles,
    removeActualFile,
  } = useOnboardingContext()

  const { currentStep, answers } = state

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't handle Enter if user is typing in an input, textarea, or contenteditable
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return
      }
      if (e.key === 'Escape') {
        skipWizard()
      } else if (e.key === 'Enter' && canProceed) {
        if (currentStep === 4) {
          nextStep() // Go to transition
        } else if (typeof currentStep === 'number' && currentStep < 4) {
          nextStep()
        }
      }
    },
    [canProceed, currentStep, nextStep, skipWizard]
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Auto-advance from transition screen
  useEffect(() => {
    if (currentStep === 'transition') {
      const timer = setTimeout(() => {
        completeWizard()
      }, 1500)
      return () => clearTimeout(timer)
    }
  }, [currentStep, completeWizard])

  // Handle final step action
  const handleBuildGuide = () => {
    nextStep() // Go to transition screen
  }

  // Welcome screen (Step 0)
  if (currentStep === 0) {
    return (
      <div className={cn('flex flex-col', className)}>
        <div className="flex-1 flex items-center justify-center">
          <div className="max-w-lg w-full text-center space-y-8 p-8">
            {/* Llama illustration placeholder */}
            <div className="text-8xl">🦙</div>

            <div className="space-y-3">
              <h2 className="text-3xl font-semibold text-foreground">
                Let's get you set up
              </h2>
              <p className="text-lg text-muted-foreground">
                A few quick questions and we'll build you a personalized
                getting-started guide.
              </p>
            </div>

            <div className="space-y-3 pt-4">
              <Button
                size="lg"
                className="w-full max-w-xs"
                onClick={nextStep}
              >
                Let's do it
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>

              <div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground"
                  onClick={skipWizard}
                >
                  Skip, I'll figure it out myself
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Transition screen
  if (currentStep === 'transition') {
    return (
      <div className={cn('flex flex-col', className)}>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-6">
            <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
            <p className="text-xl text-muted-foreground">
              Building your guide...
            </p>
          </div>
        </div>
      </div>
    )
  }

  // Question screens (Steps 1-4)
  if (typeof currentStep === 'number' && currentStep >= 1 && currentStep <= 4) {
    return (
      <div className={cn('flex flex-col', className)}>
        {/* Header with progress and skip */}
        <div className="flex-shrink-0 flex items-center justify-between border-b border-border px-6 py-4">
          <WizardProgress currentStep={currentStep} totalSteps={4} />
          <Button
            variant="ghost"
            size="sm"
            onClick={skipWizard}
            className="text-muted-foreground gap-1.5"
          >
            <X className="h-4 w-4" />
            Skip
          </Button>
        </div>

        {/* Content area - scrollable */}
        <div className="flex-1 overflow-y-auto">
          {/* Wider container for project type selector (5 cards), narrower for other steps */}
          <div className={cn(
            'mx-auto px-6 py-8',
            currentStep === 1 ? 'max-w-4xl' : 'max-w-2xl'
          )}>
            {currentStep === 1 && (
              <ProjectTypeSelector
                selected={answers.projectType}
                onSelect={setProjectType}
              />
            )}
            {currentStep === 2 && (
              <DataStatusSelector
                selected={answers.dataStatus}
                onSelect={setDataStatus}
                selectedSampleDataset={answers.selectedSampleDataset}
                onSelectSampleDataset={setSelectedSampleDataset}
                projectType={answers.projectType}
                uploadedFiles={answers.uploadedFiles}
                onUploadedFilesChange={setUploadedFiles}
                datasetName={answers.datasetName}
                onDatasetNameChange={setDatasetName}
                onAddActualFiles={addActualFiles}
                onRemoveActualFile={removeActualFile}
              />
            )}
            {currentStep === 3 && (
              <DeployTargetSelector
                selected={answers.deployTarget}
                onSelect={setDeployTarget}
              />
            )}
            {currentStep === 4 && (
              <ExperienceSelector
                selected={answers.experienceLevel}
                onSelect={setExperienceLevel}
              />
            )}
          </div>
        </div>

        {/* Footer with navigation */}
        <div className="flex-shrink-0 flex items-center justify-between border-t border-border px-6 py-4 bg-muted/30">
          <Button
            variant="outline"
            size="lg"
            onClick={prevStep}
            className="gap-2 px-6"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>

          {currentStep === 4 ? (
            <Button
              size="lg"
              onClick={handleBuildGuide}
              disabled={!canProceed}
              className="gap-2 px-8 font-semibold"
            >
              Build my guide
              <ArrowRight className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="lg"
              onClick={nextStep}
              disabled={!canProceed}
              className="gap-2 px-8 font-semibold"
            >
              Next
              <ArrowRight className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    )
  }

  // Fallback - shouldn't reach here
  return null
}
