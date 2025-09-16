import React, { createContext, useContext, useMemo } from 'react'
import { useProjectModal } from '../hooks/useProjectModal'
import { getCurrentNamespace } from '../utils/namespaceUtils'
import { useProjects } from '../hooks/useProjects'
import { getProjectsList } from '../utils/projectConstants'

export const ProjectModalContext = createContext<ReturnType<
  typeof useProjectModal
> | null>(null)

export function useProjectModalContext() {
  const ctx = useContext(ProjectModalContext)
  if (!ctx)
    throw new Error(
      'useProjectModalContext must be used within ProjectModalProvider'
    )
  return ctx
}

export function ProjectModalProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const namespace = getCurrentNamespace()
  const { data: projectsResponse } = useProjects(namespace)
  const existingProjects = useMemo(
    () => getProjectsList(projectsResponse),
    [projectsResponse]
  )
  const modal = useProjectModal({ namespace, existingProjects })

  return (
    <ProjectModalContext.Provider value={modal}>
      {children}
    </ProjectModalContext.Provider>
  )
}
