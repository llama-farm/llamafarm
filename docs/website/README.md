# LlamaFarm Documentation Website

This is the official documentation website for LlamaFarm - a config-based AI framework for local-first deployment. Built with modern web technologies for the AI community.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** version 18.0 or higher
- **Yarn** package manager

## Setup

### 1. Install Dependencies

Navigate to the website directory and install all required dependencies:

```bash
cd docs/website
yarn install
```

This will install:
- Docusaurus and its dependencies
- Testing frameworks (Jest, Puppeteer)
- All other required packages

### 2. Start Development Server

#### Quick Start
```bash
./scripts/start-docusaurus.sh
```

#### Manual Start
```bash
yarn start
```

The development server will:
- Start on `http://localhost:3000`
- Automatically open in your browser
- Enable hot reloading for instant updates

### 3. Build for Production

```bash
yarn build
```

This generates static content in the `build` directory, optimized for production deployment.

## Testing

### Run Complete Test Suite

For a comprehensive test including setup verification:

```bash
./scripts/setup-and-test.sh
```

This script will:
- ✅ Verify Node.js version (18+)
- ✅ Install dependencies
- ✅ Run TypeScript type checking
- ✅ Build the project
- ✅ Start server and run health checks
- ✅ Verify all endpoints and assets

### Jest Tests

Run specific test suites:

```bash
# Run all tests
yarn test

# Run UI tests only (uses Puppeteer)
yarn test:ui

# Run backend/API tests only
yarn test:backend

# Run tests in watch mode
yarn test:watch

# Generate coverage report
yarn test:coverage
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `yarn start` | Start development server |
| `yarn build` | Build for production |
| `yarn serve` | Serve production build locally |
| `yarn clear` | Clear Docusaurus cache |
| `yarn typecheck` | Run TypeScript type checking |
| `yarn test` | Run all tests |
| `yarn deploy` | Deploy to GitHub Pages |

## Project Structure

```
docs/website/
├── __tests__/         # Test files
│   ├── ui.test.js     # UI/Frontend tests
│   └── backend.test.js # Backend/API tests
├── blog/              # Blog posts
├── docs/              # Documentation pages
│   ├── intro.md       # Getting started
│   ├── models.md      # Models documentation
│   ├── prompts.md     # Prompts documentation
│   └── rag.md         # RAG documentation
├── scripts/           # Utility scripts
│   ├── setup-and-test.sh
│   └── start-docusaurus.sh
├── src/               # Source code
│   ├── components/    # React components
│   ├── css/          # Global styles
│   └── pages/        # Custom pages
├── static/           # Static assets
│   └── img/          # Images
├── docusaurus.config.ts  # Main configuration
├── sidebars.ts       # Sidebar configuration
├── package.json      # Dependencies
└── README.md         # This file
```

## Troubleshooting

### Port Already in Use

If port 3000 is already in use:

```bash
yarn start --port 3001
```

### Clear Cache

If you encounter build issues:

```bash
yarn clear
rm -rf .docusaurus build node_modules/.cache
```

### Dependency Issues

Remove and reinstall dependencies:

```bash
rm -rf node_modules yarn.lock
yarn install
```

### Test Failures

If tests fail due to server issues:
1. Ensure no other process is using port 3000
2. Clear cache and rebuild
3. Check Node.js version is 18+

## Deployment

### GitHub Pages

```bash
# Using SSH
USE_SSH=true yarn deploy

# Using HTTPS
GIT_USER=<Your GitHub username> yarn deploy
```

### Other Platforms

After building (`yarn build`), deploy the `build` directory to any static hosting service:
- Netlify
- Vercel
- AWS S3
- CloudFlare Pages

## Development Guidelines

### Adding Documentation

1. Create markdown files in the `docs/` directory
2. Update `sidebars.ts` to include new pages
3. Follow the existing markdown structure

### Writing Blog Posts

1. Add markdown files to the `blog/` directory
2. Use frontmatter for metadata
3. Name files with date prefix: `YYYY-MM-DD-title.md`

### Custom Components

1. Add React components to `src/components/`
2. Import and use in markdown files via MDX
3. Keep components modular and reusable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `yarn test`
5. Build: `yarn build`
6. Submit a pull request

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [React Documentation](https://react.dev/)
- [MDX Documentation](https://mdxjs.com/)

## License

This documentation site follows the same license as the LlamaFarm project.