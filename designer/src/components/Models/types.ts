export type ModelStatus = 'ready' | 'downloading'

export interface InferenceModel {
  id: string
  name: string
  modelIdentifier?: string
  meta: string
  badges: string[]
  isDefault?: boolean
  status?: ModelStatus
}

