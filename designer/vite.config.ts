import { defineConfig } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'dist/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // Workaround for occasional resolution issues on some setups
      'react-markdown': path.resolve(
        __dirname,
        './node_modules/react-markdown/index.js'
      ),
    },
  },
  optimizeDeps: {
    include: ['react-markdown', 'rehype-sanitize'],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // CodeMirror core packages (loaded together as they're tightly coupled)
          'codemirror-core': ['@codemirror/state', '@codemirror/view'],
          // CodeMirror language and features (loaded on-demand)
          'codemirror-features': [
            '@codemirror/language',
            '@codemirror/commands',
            '@codemirror/lang-json',
            '@codemirror/lang-yaml',
            '@codemirror/search',
            '@codemirror/theme-one-dark',
            '@lezer/highlight',
          ],
          // YAML parsing
          'yaml-validation': ['yaml'],
          // React vendor libraries
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          // UI vendor libraries
          'ui-vendor': [
            '@radix-ui/react-dialog',
            '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-label',
            '@radix-ui/react-slot',
          ],
          // Data fetching and utilities
          'utils-vendor': ['@tanstack/react-query', 'axios'],
        },
      },
    },
    // Increase chunk size warning limit since we're intentionally chunking
    chunkSizeWarningLimit: 600,
    // Enable source maps for better debugging
    sourcemap: true,
  },
  server: {
    proxy: {
      // Proxy all /api/* requests to the backend server
      // Rewrite /api/v1/* to /v1/* since backend doesn't have /api prefix
      '/api': {
        target: process.env.API_URL || 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        // Rewrite path to remove /api prefix
        rewrite: path => path.replace(/^\/api/, ''),
        // Preserve headers and cookies
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Log proxy requests for debugging
            const originalUrl = req.url
            const rewrittenUrl = originalUrl?.replace(/^\/api/, '') || ''
            console.log(
              `[PROXY] ${req.method} ${originalUrl} -> ${options.target}${rewrittenUrl}`
            )
          })
        },
      },
    },
  },
})
