import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'json', 'html', 'lcov'],
        exclude: [
          'node_modules/**',
          'dist/**',
          'build/**',
          '**/*.d.ts',
          '**/*.config.*',
          '**/test/**',
          '**/__tests__/**',
          '**/*.test.{ts,tsx}',
          'src/test/**',
        ],
        include: ['src/**/*.{ts,tsx}'],
      },
      include: ['src/**/*.{test,spec}.{ts,tsx}'],
      exclude: ['node_modules', 'dist', 'build'],
    },
  })
)

