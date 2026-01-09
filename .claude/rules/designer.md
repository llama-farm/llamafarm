# Designer

## Overview
- Browser-based project workbench for building AI applications
- YAML config editing with live validation
- Interactive chat testing
- Dataset management (import/export/annotation)
- RAG strategy configuration
- Model selection and dashboards

## Architecture

### Entry Points
- `designer/src/main.tsx` - React application entry
- `designer/src/App.tsx` - Root component and routing

### Directory Structure
- **api/** - Server API clients
  - `client.ts` - Base HTTP client configuration
  - `projectService.ts` - Project CRUD operations
  - `chatService.ts` - Chat session management
  - `chatCompletionsService.ts` - Inference requests
  - `modelService.ts` - Model listing
  - `datasets.ts` - Dataset operations
  - `healthService.ts` - Service health checks
- **components/** - React components organized by feature
  - `Chatbox/` - Chat interface components
  - `ConfigEditor/` - YAML configuration editor
  - `Dashboard/` - Project dashboard views
  - `Data/` - Dataset management UI
  - `Models/` - Model selection and configuration
  - `Project/` - Project modals and management
  - `Prompt/` - Prompt engineering interface
  - `Rag/` - RAG strategy configuration
  - `Samples/` - Sample project browser
  - `ui/` - Shared UI primitives (shadcn/ui based)
  - `common/` - Shared component utilities
- **contexts/** - React context providers
  - `ThemeContext.tsx` - Dark/light mode
  - `ProjectModalContext.tsx` - Project modal state
  - `UnsavedChangesContext.tsx` - Dirty state tracking
- **hooks/** - Custom React hooks
  - `useProjects.ts` - Project data fetching
  - `useChatSession.ts` - Chat state management
  - `useActiveProject.ts` - Current project context
- **utils/** - Utility functions
- **types/** - TypeScript type definitions
- **constants/** - Application constants

### State Management
- TanStack Query for server state
- React Context for UI state
- Local storage for preferences

## Development

### Running
- `nx dev designer` or `cd designer && pnpm dev`
- Default port: 5173
- Hot module replacement enabled

### Testing
- `cd designer && pnpm test`
- Uses Vitest
- Test utilities in `src/test/`

### Building
- `nx build designer` produces static assets in `designer/dist/`
- Production build served by Server
