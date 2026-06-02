/**
 * Rename .js → .cjs in the CJS output directory and fix internal imports.
 */
import { readdirSync, renameSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'

const dir = join(import.meta.dirname, '..', 'dist', 'cjs')

for (const file of readdirSync(dir)) {
  if (!file.endsWith('.js')) continue
  const oldPath = join(dir, file)
  const newPath = join(dir, file.replace(/\.js$/, '.cjs'))

  // Fix require('./foo.js') → require('./foo.cjs')
  let content = readFileSync(oldPath, 'utf8')
  content = content.replace(/require\("\.\/([^"]+)\.js"\)/g, 'require("./$1.cjs")')
  writeFileSync(oldPath, content)

  renameSync(oldPath, newPath)
}

// Rename .d.ts is fine as-is (TypeScript resolves .d.ts from .cjs via exports map)
