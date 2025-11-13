# Designer Development Guide

This guide covers how to run, build, and customize the LlamaFarm Designer for local development or contribution.

## Prerequisites

Before you begin, ensure you have:

- **Node.js 20+**: Download from [nodejs.org](https://nodejs.org/) or use `nvm`
- **npm**: Comes with Node.js
- **LlamaFarm Server**: The Designer requires a running backend server

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm/designer
```

### 2. Install Dependencies

The Designer uses `--legacy-peer-deps` due to some dependency resolution quirks:

```bash
npm install --legacy-peer-deps
```

This installs all required packages including:
- React 18
- Vite (build tool)
- TanStack Query (data fetching)
- Radix UI (component library)
- CodeMirror (YAML editor)
- Axios (HTTP client)

### 3. Start the Backend Server

The Designer needs the LlamaFarm server running. In a separate terminal:

```bash
# Option 1: Using the CLI
lf start

# Option 2: Manual server start
cd ../server
uv sync
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the server is running:
```bash
curl http://localhost:8000/health/liveness
```

### 4. Start the Development Server

Back in the designer directory:

```bash
npm run dev
```

The Designer will start on `http://localhost:5173` (Vite's default port).

Vite will automatically:
- Proxy API requests to `http://localhost:8000`
- Hot-reload when you edit source files
- Display build errors in the browser

### 5. Open in Browser

Navigate to `http://localhost:5173` to see the Designer.

## Development Workflow

### Project Structure

```
designer/
├── src/
│   ├── api/              # API client and service functions
│   ├── components/       # React components (organized by feature)
│   ├── hooks/            # Custom React hooks
│   ├── contexts/         # React context providers
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # Helper functions
│   ├── App.tsx           # Main app component with routing
│   └── main.tsx          # Application entry point
├── public/               # Static assets
├── nginx/                # Nginx config for production
├── package.json          # Dependencies and scripts
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
├── Dockerfile            # Multi-stage build for production
└── docker-entrypoint.sh  # Environment variable injection
```

### API Proxy Configuration

During development, Vite proxies API requests to avoid CORS issues. This is configured in `vite.config.ts`:

```typescript
server: {
  proxy: {
    '/api': {
      target: process.env.API_URL || 'http://localhost:8000',
      changeOrigin: true,
      rewrite: path => path.replace(/^\/api/, ''),
    },
  },
}
```

The client code (`src/api/client.ts`) automatically detects localhost and connects directly to `http://localhost:8000`.

### Environment Variables

You can customize the API URL using environment variables:

```bash
# .env.local
VITE_APP_API_URL=http://localhost:8000
```

Available variables:
- `VITE_APP_API_URL`: Backend API base URL
- `VITE_API_VERSION`: API version (default: `v1`)
- `VITE_APP_ENV`: Environment name (development/production)

### Making Changes

1. **Edit source files** in `src/`
2. **See changes instantly** via hot module replacement
3. **Check the browser console** (F12) for errors
4. **Test API calls** using the Network tab

### Code Style

The project uses ESLint and Prettier:

```bash
# Check for linting errors
npm run lint

# Auto-fix linting errors
npm run lint:fix
```

Follow these conventions:
- Use functional components with hooks
- Prefer TypeScript interfaces over types
- Keep components focused and small
- Use TanStack Query for all API calls
- Handle loading and error states

## Building for Production

### Local Build

To create a production build locally:

```bash
npm run build
```

This:
1. Runs TypeScript compiler (`tsc`) to check types
2. Bundles and minifies with Vite
3. Outputs to `dist/` directory

Preview the production build:

```bash
npm run preview
```

### Build Output

The production build includes:
- Minified JavaScript bundles
- Optimized CSS
- Static assets
- `index.html` entry point

Build artifacts are code-split for efficient loading:
- `codemirror-core`: Editor core functionality
- `codemirror-features`: Language support and themes
- `react-vendor`: React and React Router
- `ui-vendor`: Radix UI components
- `utils-vendor`: TanStack Query and Axios

## Docker Build

The Designer uses a multi-stage Docker build for production deployment.

### Building the Image

```bash
cd designer
docker build -t llamafarm-designer .
```

The Dockerfile:
1. **Build stage** (Node 20 Alpine): Installs deps and builds
2. **Production stage** (Nginx Alpine): Serves static files

### Running the Container

```bash
docker run -d \
  --name llamafarm-designer \
  -p 3123:80 \
  -e VITE_APP_API_URL=http://localhost:8000 \
  llamafarm-designer
```

### Environment Variable Injection

The container uses a custom entrypoint (`docker-entrypoint.sh`) that:
1. Captures `VITE_APP_*` environment variables
2. Injects them into the built `index.html` as a global `ENV` object
3. Starts Nginx

This allows runtime configuration without rebuilding the image.

## Architecture Overview

### Tech Stack

- **React 18**: UI framework with hooks and concurrent features
- **TypeScript**: Type safety and better DX
- **Vite**: Fast build tool with HMR
- **TanStack Query**: Server state management and caching
- **React Router**: Client-side routing
- **Radix UI**: Accessible component primitives
- **Tailwind CSS**: Utility-first styling
- **CodeMirror 6**: Advanced code editor
- **Axios**: HTTP client with interceptors

### State Management

The Designer uses multiple state management strategies:

1. **Server State** (TanStack Query):
   - Projects, datasets, models
   - Cached with automatic revalidation
   - See `src/hooks/` for query definitions

2. **Local State** (React hooks):
   - UI state (modals, panels, forms)
   - Transient data not persisted

3. **Context** (React Context):
   - Theme (light/dark)
   - Active project
   - Modal state

### Data Flow

1. User interacts with component
2. Component calls hook (e.g., `useCreateDataset`)
3. Hook triggers TanStack Query mutation
4. Query sends request via `src/api/` service
5. Server responds
6. Query updates cache and UI re-renders

### Key Components

- **Header**: Global navigation and project switcher
- **Dashboard**: Project overview and quick stats
- **Data**: Dataset and strategy management
- **Models**: Runtime configuration
- **Databases**: RAG setup
- **Prompts**: Prompt engineering interface
- **Chat**: Interactive testing with chatbox
- **ConfigEditor**: YAML editor with validation

### Routing

Routes are defined in `src/App.tsx`:

```
/                          → Home (project selection)
/samples                   → Sample projects
/chat
  /dashboard              → Project dashboard
  /data                   → Dataset management
  /data/:datasetId        → Dataset details
  /models                 → Model configuration
  /databases              → RAG configuration
  /prompt                 → Prompt management
  /test                   → Testing interface
```

## Testing

### Manual Testing

1. Start the dev server
2. Test each section:
   - Create/edit projects
   - Upload datasets
   - Configure models
   - Test RAG queries
   - Edit configuration

### API Testing

Use the browser Network tab to inspect:
- Request/response payloads
- Status codes
- Response times

### Console Logging

The Designer logs proxy requests during development:
```
[PROXY] GET /api/v1/projects/default/my-project -> http://localhost:8000/v1/projects/default/my-project
```

## Common Development Tasks

### Adding a New Component

1. Create component file in `src/components/`
2. Define props interface
3. Implement component
4. Export from `src/components/index.ts` (if needed)

### Adding a New API Endpoint

1. Add function to appropriate `src/api/*.ts` file
2. Define request/response types in `src/types/`
3. Create React Query hook in `src/hooks/`
4. Use hook in component

### Adding a New Route

1. Add route to `src/App.tsx`
2. Create page component
3. Update navigation links if needed

### Updating Types from Schema

When the backend schema changes:

```bash
cd ../config
uv run python generate_types.py
```

This regenerates both backend and frontend types to keep them in sync.

## Troubleshooting

### Port Already in Use

If port 5173 is occupied:
```bash
# Kill the process using the port
lsof -ti:5173 | xargs kill -9

# Or specify a different port
npm run dev -- --port 3000
```

### API Connection Errors

- Verify server is running: `curl http://localhost:8000/health/liveness`
- Check `VITE_APP_API_URL` in environment
- Look at browser console for CORS errors

### Build Failures

- Clear node_modules and reinstall: `rm -rf node_modules && npm install --legacy-peer-deps`
- Check TypeScript errors: `npx tsc --noEmit`
- Verify all imports resolve correctly

### Hot Reload Not Working

- Check file watchers limit on Linux: `echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p`
- Restart the dev server
- Clear browser cache

## Contributing

### Before Submitting a PR

1. Run linter: `npm run lint:fix`
2. Test your changes manually
3. Build successfully: `npm run build`
4. Update types if schema changed
5. Document new features in this guide

### Code Review Checklist

- [ ] TypeScript types are correct
- [ ] Loading and error states handled
- [ ] Responsive design works on mobile
- [ ] Dark mode looks correct
- [ ] No console errors or warnings
- [ ] API calls use TanStack Query
- [ ] Components are accessible (keyboard navigation, ARIA labels)

## Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [TanStack Query Documentation](https://tanstack.com/query/latest)
- [Radix UI Documentation](https://www.radix-ui.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [CodeMirror 6 Documentation](https://codemirror.net/)

## Next Steps

- Return to the [Designer Overview](./index.md)
- Explore [Designer Features](./features.md)
- Check the [Contributing Guide](../contributing/index.md) for general contribution guidelines
