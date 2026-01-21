# TanStack Query Patterns

Best practices for TanStack Query v5 in the Designer subsystem.

## Query Key Factories

Always use query key factories for consistent cache management:

```typescript
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (namespace: string) => [...projectKeys.lists(), namespace] as const,
  details: () => [...projectKeys.all, 'detail'] as const,
  detail: (namespace: string, id: string) =>
    [...projectKeys.details(), namespace, id] as const,
}
```

## Checklist Items

### 1. Query Keys Use Factory Pattern

| Attribute | Value |
|-----------|-------|
| Description | Query keys must use factory pattern for cache consistency |
| Search Pattern | `grep -E "queryKey:\s*\[" hooks/*.ts` |
| Pass Criteria | Parameterized queries reference a keys factory; static queries may use inline arrays |
| Severity | High |
| Recommendation | Create `{resource}Keys` factory object with hierarchical key builders |

**Note**: Inline query keys are acceptable for simple, non-parameterized queries (e.g., `['examples']`, `['github', 'stars']`) where cache invalidation complexity is low. Use factories for any query that takes parameters or participates in cache invalidation hierarchies.

### 2. Queries Have Proper Enabled Conditions

| Attribute | Value |
|-----------|-------|
| Description | Queries with dependencies must use `enabled` option |
| Search Pattern | `grep -B5 -A10 "useQuery\({" hooks/*.ts` |
| Pass Criteria | Queries depending on variables have `enabled: !!variable` |
| Severity | Critical |
| Recommendation | Add `enabled: !!namespace && !!projectId` for dependent queries |

### 3. Mutations Invalidate Related Queries

| Attribute | Value |
|-----------|-------|
| Description | Mutations must invalidate affected queries in onSuccess |
| Search Pattern | `grep -A20 "useMutation\({" hooks/*.ts` |
| Pass Criteria | All mutations have onSuccess with invalidateQueries or setQueryData |
| Severity | Critical |
| Recommendation | Use `queryClient.invalidateQueries({ queryKey: resourceKeys.list(...) })` |

### 4. StaleTime Configured Appropriately

| Attribute | Value |
|-----------|-------|
| Description | Queries should have appropriate staleTime for their data |
| Search Pattern | `grep -E "staleTime:" hooks/*.ts` |
| Pass Criteria | Static data: 5-10 min, dynamic data: 30s-2min, real-time: 0 |
| Severity | High |
| Recommendation | Set `staleTime: 5 * 60 * 1000` for infrequently changing data |

### 5. Error Handling in Mutations

| Attribute | Value |
|-----------|-------|
| Description | Mutations should handle errors appropriately |
| Search Pattern | `grep -A25 "useMutation\({" hooks/*.ts` |
| Pass Criteria | Mutations have onError callback or error is handled by caller |
| Severity | High |
| Recommendation | Add `onError: (error) => console.error('Operation failed:', error)` |

### 6. Optimistic Updates Where Appropriate

| Attribute | Value |
|-----------|-------|
| Description | User-facing mutations should use optimistic updates |
| Search Pattern | `grep -E "onMutate:" hooks/*.ts` |
| Pass Criteria | Toggle/delete operations use optimistic updates with rollback |
| Severity | High |
| Recommendation | Use onMutate to update cache, return context for rollback in onError |

### 7. Query Client Used from Context

| Attribute | Value |
|-----------|-------|
| Description | Mutations must get queryClient from useQueryClient hook |
| Search Pattern | `grep -E "const queryClient = useQueryClient\(\)" hooks/*.ts` |
| Pass Criteria | Never import queryClient directly in hooks |
| Severity | High |
| Recommendation | Use `const queryClient = useQueryClient()` inside hook function |

### 8. Retry Configuration

| Attribute | Value |
|-----------|-------|
| Description | Queries and mutations should have appropriate retry config |
| Search Pattern | `grep -E "retry:" hooks/*.ts` |
| Pass Criteria | Network requests: retry 1-2, idempotent: retry 2, mutations: retry 1 |
| Severity | Medium |
| Recommendation | Add `retry: 1` for mutations, `retry: 2` for queries |

### 9. Polling Uses refetchInterval

| Attribute | Value |
|-----------|-------|
| Description | Polling should use refetchInterval, not manual timers |
| Search Pattern | `grep -E "refetchInterval:" hooks/*.ts` |
| Pass Criteria | No setInterval for query refetching |
| Severity | Medium |
| Recommendation | Use `refetchInterval: (query) => condition ? 2000 : false` |

### 10. GcTime Configured for Important Data

| Attribute | Value |
|-----------|-------|
| Description | Important data should have appropriate gcTime (garbage collection) |
| Search Pattern | `grep -E "gcTime:" hooks/*.ts` |
| Pass Criteria | Data that should persist has gcTime > staleTime |
| Severity | Medium |
| Recommendation | Set `gcTime: 5 * 60 * 1000` for data that should remain cached |

### 11. Dependent Queries Chain Properly

| Attribute | Value |
|-----------|-------|
| Description | Queries depending on other query data must wait for it |
| Search Pattern | `grep -B10 "enabled:.*data\." hooks/*.ts` |
| Pass Criteria | Dependent queries use `enabled: !!parentQuery.data` |
| Severity | Medium |
| Recommendation | Pass parent query result to enabled condition |

### 12. Select for Data Transformation

| Attribute | Value |
|-----------|-------|
| Description | Transform query data with select, not in component |
| Search Pattern | `grep -E "select:" hooks/*.ts` |
| Pass Criteria | Complex transformations use select option |
| Severity | Low |
| Recommendation | Use `select: (data) => data.items.filter(...)` for transformations |

## Patterns

### Query Hook

```typescript
export function useProjects(namespace: string) {
  return useQuery({
    queryKey: projectKeys.list(namespace),
    queryFn: () => projectService.listProjects(namespace),
    enabled: !!namespace,
    staleTime: 5 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
  })
}
```

### Mutation Hook with Cache Update

```typescript
export function useCreateProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ namespace, request }) =>
      projectService.createProject(namespace, request),
    onSuccess: (data, variables) => {
      // Invalidate list to refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.list(variables.namespace)
      })
      // Optionally seed the detail cache
      queryClient.setQueryData(
        projectKeys.detail(variables.namespace, data.project.name),
        { project: data.project }
      )
    },
    onError: (error) => {
      console.error('Failed to create project:', error)
    },
  })
}
```

### Polling Query

```typescript
export function useTaskStatus(taskId: string | null) {
  return useQuery({
    queryKey: ['task-status', taskId],
    queryFn: () => taskService.getStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.state === 'SUCCESS' || data?.state === 'FAILURE') {
        return false // Stop polling
      }
      return 2000 // Poll every 2 seconds
    },
    refetchIntervalInBackground: true,
    staleTime: 0,
  })
}
```

### Optimistic Update Pattern

```typescript
export function useToggleFavorite() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (projectId: string) =>
      projectService.toggleFavorite(projectId),
    onMutate: async (projectId) => {
      await queryClient.cancelQueries({ queryKey: projectKeys.all })

      const previousData = queryClient.getQueryData(projectKeys.lists())

      queryClient.setQueryData(projectKeys.lists(), (old) =>
        old?.map(p =>
          p.id === projectId
            ? { ...p, isFavorite: !p.isFavorite }
            : p
        )
      )

      return { previousData }
    },
    onError: (err, projectId, context) => {
      queryClient.setQueryData(projectKeys.lists(), context?.previousData)
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.all })
    },
  })
}
```

### Infinite Query Pattern

```typescript
export function useInfiniteMessages(chatId: string) {
  return useInfiniteQuery({
    queryKey: ['messages', chatId],
    queryFn: ({ pageParam }) =>
      chatService.getMessages(chatId, { cursor: pageParam }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    enabled: !!chatId,
  })
}
```

## Anti-Patterns

### Avoid: Inline Query Keys for Parameterized Queries

```typescript
// Bad - inline array with parameters (hard to invalidate consistently)
useQuery({ queryKey: ['projects', namespace], ... })

// Good - factory for parameterized queries
useQuery({ queryKey: projectKeys.list(namespace), ... })

// Acceptable - inline for static, non-parameterized queries
useQuery({ queryKey: ['examples'], ... })
```

### Avoid: Missing Enabled Condition

```typescript
// Bad - fires with undefined id
useQuery({ queryKey: projectKeys.detail(id), queryFn: () => fetch(id) })

// Good - waits for id
useQuery({
  queryKey: projectKeys.detail(id),
  queryFn: () => fetch(id),
  enabled: !!id,
})
```

### Avoid: Manual Refetching with setInterval

```typescript
// Bad - manual polling
useEffect(() => {
  const interval = setInterval(() => refetch(), 2000)
  return () => clearInterval(interval)
}, [])

// Good - built-in polling
useQuery({
  queryKey: ['status'],
  queryFn: fetchStatus,
  refetchInterval: 2000,
})
```
