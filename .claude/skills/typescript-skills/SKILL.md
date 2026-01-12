---
name: typescript-skills
description: Shared TypeScript best practices for Designer and Electron subsystems.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# TypeScript Skills for LlamaFarm

Shared TypeScript best practices for Designer (React) and Electron App subsystems.

## Overview

This skill covers idiomatic TypeScript patterns for LlamaFarm's frontend applications:
- **designer/**: React 18 + TanStack Query + TailwindCSS + Radix UI
- **electron-app/**: Electron 28 + Electron Vite

## Tech Stack

| Subsystem | Framework | Build | Key Libraries |
|-----------|-----------|-------|---------------|
| designer | React 18 | Vite | TanStack Query, Radix UI, axios, react-router-dom |
| electron-app | Electron 28 | electron-vite | electron-updater, axios |

## Configuration

Both projects use strict TypeScript:
```json
{
  "strict": true,
  "noUnusedLocals": true,
  "noUnusedParameters": true,
  "noFallthroughCasesInSwitch": true
}
```

## Core Principles

1. **Strict mode always** - Never use `any` without explicit justification
2. **Prefer interfaces** - Use `interface` for object shapes, `type` for unions/intersections
3. **Explicit return types** - Always type public function returns
4. **Immutability** - Use `readonly` and `as const` where applicable
5. **Null safety** - Handle null/undefined explicitly, avoid non-null assertions

## Related Documents

- [patterns.md](./patterns.md) - Idiomatic TypeScript patterns
- [typing.md](./typing.md) - Strict typing, generics, utility types
- [testing.md](./testing.md) - Vitest and testing patterns
- [security.md](./security.md) - XSS prevention, input validation

## Quick Reference

### React Component Pattern
```tsx
interface Props {
  readonly title: string
  readonly onAction?: () => void
}

function MyComponent({ title, onAction }: Props): JSX.Element {
  return <button onClick={onAction}>{title}</button>
}
```

### TanStack Query Hook Pattern
```typescript
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  detail: (id: string) => [...projectKeys.all, 'detail', id] as const,
}

export function useProject(id: string) {
  return useQuery({
    queryKey: projectKeys.detail(id),
    queryFn: () => fetchProject(id),
    enabled: !!id,
  })
}
```

### Error Class Pattern
```typescript
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly response?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}
```

## Checklist Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Typing | 3 | 4 | 2 | 1 |
| Patterns | 2 | 3 | 3 | 2 |
| Testing | 2 | 3 | 2 | 1 |
| Security | 4 | 2 | 1 | 0 |
