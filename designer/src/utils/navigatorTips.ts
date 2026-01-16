/**
 * Generates contextual tips for the ChecklistNavigator
 * Tips are personalized based on user's onboarding journey
 */

import type { OnboardingAnswers, ChecklistStep } from '../types/onboarding'
import { getDemoById } from '../config/demos'

export interface NavigatorTip {
  text: string
}

/**
 * Generate a contextual tip based on the current checklist step and user's answers
 */
export function generateNavigatorTip(
  currentStep: ChecklistStep | null,
  answers: OnboardingAnswers,
  _currentPath: string
): NavigatorTip | null {
  if (!currentStep) return null

  const { projectType, dataStatus, uploadedFiles, datasetName, selectedSampleDataset } = answers
  const fileCount = uploadedFiles?.length || 0

  // Get demo name if sample data was selected
  const demoName = selectedSampleDataset ? getDemoById(selectedSampleDataset)?.displayName ?? null : null

  // Determine which page we're on by step ID patterns
  const isDataStep = currentStep.id.includes('data') || currentStep.id.includes('sample')
  const isPromptStep = currentStep.id.includes('prompt')
  const isTestStep = currentStep.id.includes('test')
  const isTrainStep = currentStep.id.includes('train')
  const isShipStep = currentStep.id.includes('ship') || currentStep.id.includes('build') || currentStep.id.includes('decide')
  const isCreateStep = currentStep.id.includes('create')

  // Data page tips (step 1)
  if (isDataStep) {
    return getDataPageTip(projectType, dataStatus, fileCount, datasetName, demoName)
  }

  // Create model step (classifier/anomaly step 2)
  if (isCreateStep) {
    return getCreateStepTip(projectType)
  }

  // Train & test step (classifier/anomaly need-data flow step 2)
  if (isTrainStep) {
    return getTrainStepTip(projectType)
  }

  // Prompt page tips (step 2 for doc-qa)
  if (isPromptStep) {
    return getPromptPageTip(projectType, dataStatus, fileCount, demoName)
  }

  // Test page tips (step 3)
  if (isTestStep) {
    return getTestPageTip(projectType, dataStatus, demoName)
  }

  // Ship/package tips (step 4)
  if (isShipStep) {
    return getShipPageTip(projectType)
  }

  return null
}

function getDataPageTip(
  projectType: string | null,
  dataStatus: string | null,
  fileCount: number,
  datasetName: string | null,
  demoName: string | null
): NavigatorTip | null {
  switch (projectType) {
    case 'doc-qa':
      if (dataStatus === 'has-data') {
        if (fileCount > 0) {
          const dsName = datasetName || 'my-data'
          return { text: `Your ${fileCount} file${fileCount === 1 ? '' : 's'} ${fileCount === 1 ? 'is' : 'are'} in "${dsName}". Hit Process to prepare them for your AI.` }
        }
        return { text: 'Upload documents you want your AI to answer questions about.' }
      }
      if (dataStatus === 'sample-data' && demoName) {
        return { text: `${demoName} data is loading. Once processed, you can ask questions about it.` }
      }
      return { text: 'Add documents to create a knowledge base for your AI assistant.' }

    case 'classifier':
      if (dataStatus === 'sample-data') {
        return { text: 'Sample classifier is training! Check progress, then test it with new content.' }
      }
      if (dataStatus === 'need-data') {
        return { text: 'Add training data or use sample data, then tap Train to create your classifier.' }
      }
      return { text: 'Upload labeled examples so your classifier can learn to categorize content.' }

    case 'anomaly':
      if (dataStatus === 'sample-data') {
        return { text: 'Sample detector is training! Once ready, test it with normal and unusual inputs.' }
      }
      if (dataStatus === 'need-data') {
        return { text: 'Add baseline data or use sample data, then tap Train to create your detector.' }
      }
      return { text: "Upload examples of 'normal' so your detector can learn what's unusual." }

    case 'doc-scan':
      return { text: 'Upload documents you want to extract structured information from.' }

    case 'exploring':
      return { text: 'Try importing a sample dataset to see how LlamaFarm works.' }

    default:
      return null
  }
}

function getCreateStepTip(projectType: string | null): NavigatorTip | null {
  switch (projectType) {
    case 'classifier':
      return { text: 'Add training data or use sample data, then hit Train to create your classifier.' }
    case 'anomaly':
      return { text: 'Add baseline data or use sample data, then hit Train to create your detector.' }
    default:
      return null
  }
}

function getTrainStepTip(projectType: string | null): NavigatorTip | null {
  switch (projectType) {
    case 'classifier':
      return { text: 'Tap Train to build your classifier, then test it with a few examples.' }
    case 'anomaly':
      return { text: 'Tap Train to build your detector, then test it with normal and unusual inputs.' }
    default:
      return null
  }
}

function getPromptPageTip(
  projectType: string | null,
  dataStatus: string | null,
  fileCount: number,
  demoName: string | null
): NavigatorTip | null {
  switch (projectType) {
    case 'doc-qa':
      if (dataStatus === 'has-data' && fileCount > 0) {
        return { text: `Customize how your AI responds. It will use your ${fileCount} uploaded doc${fileCount === 1 ? '' : 's'} to answer questions.` }
      }
      if (dataStatus === 'sample-data' && demoName) {
        return { text: `See how we configured ${demoName}. Edit the prompt to change the AI's personality.` }
      }
      return { text: 'The system prompt shapes how your AI responds. Be specific about its role.' }

    case 'classifier':
      return { text: 'Define your categories clearly. The prompt guides how content gets labeled.' }

    case 'anomaly':
      return { text: "Describe what 'normal' looks like. This helps the detector know what to flag." }

    case 'doc-scan':
      return { text: 'Specify what information to extract. Be explicit about the fields you need.' }

    default:
      return null
  }
}

function getTestPageTip(
  projectType: string | null,
  dataStatus: string | null,
  demoName: string | null
): NavigatorTip | null {
  switch (projectType) {
    case 'doc-qa':
      if (dataStatus === 'sample-data' && demoName) {
        return { text: 'Try the sample questions, or ask your own. See how the AI uses the docs.' }
      }
      return { text: 'Ask questions about your docs. Check if answers cite the right sources.' }

    case 'classifier':
      return { text: 'Test with new content. Check if the labels make sense for your use case.' }

    case 'anomaly':
      return { text: 'Try both normal and unusual inputs. See if the detector catches the weird stuff.' }

    case 'doc-scan':
      return { text: 'Test extraction on your documents. Verify the output matches what you need.' }

    case 'exploring':
      return { text: 'Explore different modes: Text Generation, Classifier, Anomaly Detection.' }

    default:
      return null
  }
}

function getShipPageTip(projectType: string | null): NavigatorTip | null {
  switch (projectType) {
    case 'doc-qa':
      return { text: 'Package creates everything needed to deploy your document Q&A assistant.' }

    case 'classifier':
      return { text: 'Your trained classifier will be included in the deployment package.' }

    case 'anomaly':
      return { text: 'Your anomaly detector will be bundled for production deployment.' }

    case 'doc-scan':
      return { text: 'Package your document scanner for production use.' }

    case 'exploring':
      return { text: 'Ready to build something real? Go back and pick a specific project type.' }

    default:
      return null
  }
}
