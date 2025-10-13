import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import apiClient, { examplesApi } from '../api/client'
import { datasetKeys } from './useDatasets'
import { projectKeys } from './useProjects'

type ExampleSummary = {
  id: string
  slug?: string
  title: string
  description?: string
  primaryModel?: string
  tags: string[]
}

export function useExamples() {
  return useQuery<{ examples: ExampleSummary[] }>({
    queryKey: ['examples'],
    queryFn: async () => {
      const { data } = await apiClient.get('/examples')
      return data
    },
  })
}

export function useExampleDatasets() {
  return useQuery<{ datasets: any[] }>({
    queryKey: ['examples', 'datasets'],
    queryFn: async () => examplesApi.listAllDatasets(),
  })
}

export function useImportExampleProject() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (payload: {
      exampleId: string
      namespace: string
      name: string
      process?: boolean
    }) => {
      const { exampleId, ...body } = payload
      const { data } = await apiClient.post(
        `/examples/${exampleId}/import-project`,
        body
      )
      return data as {
        project: string
        namespace: string
        datasets: string[]
        task_ids: string[]
      }
    },
    onSuccess: (_data, variables) => {
      // New project created – refresh projects list and seed its dataset list
      queryClient.invalidateQueries({
        queryKey: projectKeys.list(variables.namespace),
      })
      if (_data?.project) {
        queryClient.invalidateQueries({
          queryKey: datasetKeys.list(variables.namespace, _data.project),
        })
      }
    },
  })
}

export function useImportExampleData() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (payload: {
      exampleId: string
      namespace: string
      project: string
      include_strategies?: boolean
      process?: boolean
    }) => {
      const { exampleId, ...body } = payload
      const { data } = await apiClient.post(
        `/examples/${exampleId}/import-data`,
        body
      )
      return data as {
        project: string
        namespace: string
        datasets: string[]
        task_ids: string[]
      }
    },
    onSuccess: (_data, variables) => {
      // Force-refresh the datasets list for the target project
      const ns = variables.namespace
      const pid = variables.project
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(ns, pid),
      })
    },
  })
}

export function useImportExampleDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (payload: {
      exampleId: string
      namespace: string
      project: string
      dataset: string
      target_dataset?: string
      include_strategies?: boolean
      process?: boolean
    }) => {
      const { exampleId, ...body } = payload
      return examplesApi.importExampleDataset(exampleId, body)
    },
    onSuccess: (_data, variables) => {
      // Refresh datasets for the project after import
      const { datasetKeys } = require('./useDatasets')
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}
