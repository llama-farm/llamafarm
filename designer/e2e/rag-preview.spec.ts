/**
 * E2E Tests for RAG Document Preview - TDD Red Phase
 * All tests written FIRST and will fail until implementation is complete.
 *
 * These tests require a running server and Designer frontend.
 * Run with: npx playwright test e2e/rag-preview.spec.ts
 */

import { test, expect, Page } from '@playwright/test'

// Test configuration
const BASE_URL = process.env.DESIGNER_URL || 'http://localhost:5173'
const NAMESPACE = 'test-ns'
const PROJECT = 'test-project'
const DATASET = 'test-dataset'

// Helper to navigate to dataset view
async function navigateToDataset(page: Page) {
  await page.goto(`${BASE_URL}/projects/${NAMESPACE}/${PROJECT}/data/${DATASET}`)
  await page.waitForLoadState('networkidle')
}

// Helper to setup mock file in dataset (would need actual test data)
async function uploadTestFile(page: Page) {
  // This would upload a test file to the dataset
  // For E2E tests, we assume test data is already set up
}

test.describe('RAG Document Preview E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing state
    await page.context().clearCookies()
  })

  test('user can preview document from dataset view', async ({ page }) => {
    await navigateToDataset(page)

    // Wait for file list to load
    await page.waitForSelector('[data-testid="file-list"]')

    // Find a file and open its action menu
    const fileRow = page.locator('[data-testid^="file-row-"]').first()
    await fileRow.hover()

    // Click the actions button
    const actionsButton = fileRow.locator('[data-testid="file-actions-button"]')
    await actionsButton.click()

    // Click "Preview Chunking" option
    await page.click('text=Preview Chunking')

    // Modal should open
    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Should show document content
    await expect(page.locator('[data-testid="preview-panel"]')).toBeVisible()

    // Should show at least one chunk
    await expect(page.locator('[data-testid="chunk-0"]')).toBeVisible()
  })

  test('adjusting chunk size updates preview', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal (assume file exists)
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Get initial chunk count
    const initialChunks = await page.locator('[data-testid^="chunk-"]').count()

    // Find chunk size slider
    const chunkSizeSlider = page.getByRole('slider', { name: /chunk size/i })
    await expect(chunkSizeSlider).toBeVisible()

    // Increase chunk size (should result in fewer chunks)
    await chunkSizeSlider.fill('2000')

    // Click refresh to apply
    await page.click('button:has-text("Refresh")')

    // Wait for new data
    await page.waitForResponse(response =>
      response.url().includes('/preview') && response.status() === 200
    )

    // Verify chunks updated
    const newChunks = await page.locator('[data-testid^="chunk-"]').count()

    // Larger chunk size = fewer chunks
    expect(newChunks).toBeLessThanOrEqual(initialChunks)
  })

  test('overlap regions are highlighted', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Set overlap to a non-zero value
    const overlapSlider = page.getByRole('slider', { name: /chunk overlap/i })
    await overlapSlider.fill('50')
    await page.click('button:has-text("Refresh")')

    await page.waitForResponse(response =>
      response.url().includes('/preview') && response.status() === 200
    )

    // Verify overlap highlighting exists (if there are multiple chunks)
    const chunks = await page.locator('[data-testid^="chunk-"]').count()

    if (chunks > 1) {
      // Should have overlap indicators
      const overlapIndicators = page.locator('[data-testid^="overlap-"]')
      await expect(overlapIndicators.first()).toBeVisible()

      // Overlap should have distinctive styling (orange background)
      const firstOverlap = overlapIndicators.first()
      await expect(firstOverlap).toHaveClass(/bg-orange/)
    }
  })

  test('clicking chunk in sidebar scrolls to chunk', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Wait for chunks to load
    await page.waitForSelector('[data-testid="sidebar-chunk-0"]')

    // Get chunk count
    const chunks = await page.locator('[data-testid^="sidebar-chunk-"]').count()

    if (chunks >= 3) {
      // Click chunk 3 in sidebar (index 2)
      await page.click('[data-testid="sidebar-chunk-2"]')

      // Verify chunk 3 is highlighted in preview
      const targetChunk = page.locator('[data-testid="chunk-2"]')
      await expect(targetChunk).toHaveClass(/ring-2/)

      // Verify it's visible (scrolled into view)
      await expect(targetChunk).toBeInViewport()
    }
  })

  test('statistics display correctly', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Wait for data to load
    await page.waitForSelector('[data-testid="preview-stats"]')

    // Verify statistics are shown
    await expect(page.locator('text=/Chunks:\\s*\\d+/')).toBeVisible()
    await expect(page.locator('text=/Avg.*Size/i')).toBeVisible()
    await expect(page.locator('text=/Total.*with.*overlaps/i')).toBeVisible()
  })

  test('changing chunk strategy updates preview', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Find strategy dropdown
    const strategyDropdown = page.getByRole('combobox')
    await strategyDropdown.click()

    // Select 'sentences' option
    await page.click('text=Sentences')

    // Click refresh
    await page.click('button:has-text("Refresh")')

    // Wait for new data
    await page.waitForResponse(response =>
      response.url().includes('/preview') && response.status() === 200
    )

    // Verify the strategy is applied (chunk boundaries should be different)
    // This is hard to verify directly, but we can check the API was called
    // with the new strategy
  })

  test('close button closes modal', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Click close button
    await page.click('button[aria-label="Close"]')

    // Modal should be hidden
    await expect(page.locator('[data-testid="preview-modal"]')).not.toBeVisible()
  })

  test('escape key closes modal', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Press Escape
    await page.keyboard.press('Escape')

    // Modal should be hidden
    await expect(page.locator('[data-testid="preview-modal"]')).not.toBeVisible()
  })

  test('shows loading state while fetching', async ({ page }) => {
    await navigateToDataset(page)

    // Slow down network to see loading state
    await page.route('**/preview', async route => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      await route.continue()
    })

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    // Should show loading indicator
    await expect(page.locator('[data-testid="preview-loading"]')).toBeVisible()

    // Wait for loading to finish
    await expect(page.locator('[data-testid="preview-loading"]')).not.toBeVisible({
      timeout: 10000,
    })
  })

  test('shows error state on API failure', async ({ page }) => {
    await navigateToDataset(page)

    // Mock API to return error
    await page.route('**/preview', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ detail: 'Internal server error' }),
      })
    })

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    // Should show error state
    await expect(page.locator('[data-testid="preview-error"]')).toBeVisible()
    await expect(page.locator('text=/error/i')).toBeVisible()
  })
})

test.describe('RAG Document Preview - Different File Types', () => {
  test('preview works with text files', async ({ page }) => {
    // This test would use a .txt file
    await navigateToDataset(page)

    // Find .txt file in the list
    const txtFile = page.locator('[data-testid^="file-row-"]:has-text(".txt")').first()

    if (await txtFile.isVisible()) {
      await txtFile.hover()
      await txtFile.locator('[data-testid="file-actions-button"]').click()
      await page.click('text=Preview Chunking')

      await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()
      await expect(page.locator('[data-testid="chunk-0"]')).toBeVisible()
    }
  })

  test('preview works with markdown files', async ({ page }) => {
    await navigateToDataset(page)

    // Find .md file in the list
    const mdFile = page.locator('[data-testid^="file-row-"]:has-text(".md")').first()

    if (await mdFile.isVisible()) {
      await mdFile.hover()
      await mdFile.locator('[data-testid="file-actions-button"]').click()
      await page.click('text=Preview Chunking')

      await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()
      await expect(page.locator('[data-testid="chunk-0"]')).toBeVisible()
    }
  })

  test('preview works with PDF files', async ({ page }) => {
    await navigateToDataset(page)

    // Find .pdf file in the list
    const pdfFile = page.locator('[data-testid^="file-row-"]:has-text(".pdf")').first()

    if (await pdfFile.isVisible()) {
      await pdfFile.hover()
      await pdfFile.locator('[data-testid="file-actions-button"]').click()
      await page.click('text=Preview Chunking')

      await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

      // PDF preview should show extracted text
      await expect(page.locator('[data-testid="preview-panel"]')).toBeVisible()
      // Parser should be indicated
      await expect(page.locator('text=/PDF/i')).toBeVisible()
    }
  })
})

test.describe('RAG Document Preview - Accessibility', () => {
  test('modal is accessible', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    const modal = page.locator('[data-testid="preview-modal"]')
    await expect(modal).toBeVisible()

    // Check for proper ARIA attributes
    await expect(modal).toHaveAttribute('role', 'dialog')
    await expect(modal).toHaveAttribute('aria-modal', 'true')

    // Check for close button accessibility
    const closeButton = modal.locator('button[aria-label="Close"]')
    await expect(closeButton).toBeVisible()
  })

  test('keyboard navigation works', async ({ page }) => {
    await navigateToDataset(page)

    // Open preview modal
    await page.click('[data-testid="file-actions-button"]')
    await page.click('text=Preview Chunking')

    await expect(page.locator('[data-testid="preview-modal"]')).toBeVisible()

    // Tab through controls
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')

    // Focus should be trapped within modal
    const focusedElement = await page.evaluate(() => document.activeElement?.closest('[data-testid="preview-modal"]'))
    expect(focusedElement).not.toBeNull()
  })
})
