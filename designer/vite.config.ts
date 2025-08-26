import { defineConfig } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
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
        rewrite: (path) => path.replace(/^\/api/, ''),
        // Preserve headers and cookies
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Log proxy requests for debugging
            const originalUrl = req.url
            const rewrittenUrl = originalUrl?.replace(/^\/api/, '') || ''
            console.log(`[PROXY] ${req.method} ${originalUrl} -> ${options.target}${rewrittenUrl}`)
          })
        }
      }
    }
  }
})
