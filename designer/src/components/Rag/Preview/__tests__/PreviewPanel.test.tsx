/**
 * Tests for PreviewPanel component - TDD Red Phase
 * All tests written FIRST and will fail until implementation is complete.
 */

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PreviewPanel } from '../PreviewPanel'

// Mock chunk data for testing
const mockChunks = [
  {
    chunk_index: 0,
    content: 'First chunk content',
    start_position: 0,
    end_position: 19,
    char_count: 19,
    word_count: 3,
  },
  {
    chunk_index: 1,
    content: 'Second chunk content',
    start_position: 19,
    end_position: 39,
    char_count: 20,
    word_count: 3,
  },
  {
    chunk_index: 2,
    content: 'Third chunk content',
    start_position: 39,
    end_position: 58,
    char_count: 19,
    word_count: 3,
  },
]

const mockOriginalText = 'First chunk contentSecond chunk contentThird chunk content'

describe('PreviewPanel', () => {
  it('renders without crashing', () => {
    render(
      <PreviewPanel
        originalText={mockOriginalText}
        chunks={mockChunks}
        chunkOverlap={0}
      />
    )

    expect(screen.getByTestId('preview-panel')).toBeInTheDocument()
  })

  it('renders chunks with alternating colors', () => {
    render(
      <PreviewPanel
        originalText={mockOriginalText}
        chunks={mockChunks}
        chunkOverlap={0}
      />
    )

    // Verify chunk content is rendered
    expect(screen.getByText('First chunk content')).toBeInTheDocument()
    expect(screen.getByText('Second chunk content')).toBeInTheDocument()
    expect(screen.getByText('Third chunk content')).toBeInTheDocument()

    // Verify alternating background colors
    const firstChunk = screen.getByTestId('chunk-0')
    const secondChunk = screen.getByTestId('chunk-1')

    // Chunks should have background color classes (purple/pink/indigo alternating)
    expect(firstChunk).toHaveClass('bg-purple-200/60')
    expect(secondChunk).toHaveClass('bg-pink-200/60')
  })

  it('renders overlap regions with distinct styling', () => {
    // Chunks with overlap
    const chunksWithOverlap = [
      {
        chunk_index: 0,
        content: 'ABCDEFGHIJ',
        start_position: 0,
        end_position: 10,
        char_count: 10,
        word_count: 1,
      },
      {
        chunk_index: 1,
        content: 'GHIJKLMNOP',
        start_position: 6,
        end_position: 16,
        char_count: 10,
        word_count: 1,
      },
    ]

    render(
      <PreviewPanel
        originalText="ABCDEFGHIJKLMNOP"
        chunks={chunksWithOverlap}
        chunkOverlap={4}
      />
    )

    // Verify overlap highlight exists
    const overlapRegion = screen.getByTestId('overlap-0-1')
    expect(overlapRegion).toBeInTheDocument()
    expect(overlapRegion).toHaveClass('bg-orange-200/60')
  })

  it('displays chunk indices at boundaries', () => {
    render(
      <PreviewPanel
        originalText={mockOriginalText}
        chunks={mockChunks}
        chunkOverlap={0}
      />
    )

    // Check for chunk index labels (1-indexed, displayed as plain numbers)
    expect(screen.getByText('1')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('handles empty chunks array', () => {
    render(
      <PreviewPanel
        originalText=""
        chunks={[]}
        chunkOverlap={0}
      />
    )

    expect(screen.getByTestId('preview-panel')).toBeInTheDocument()
    // Should show empty state or just the panel
    expect(screen.queryByTestId('chunk-0')).not.toBeInTheDocument()
  })

  it('handles single chunk without overlap styling', () => {
    const singleChunk = [
      {
        chunk_index: 0,
        content: 'Only one chunk here',
        start_position: 0,
        end_position: 19,
        char_count: 19,
        word_count: 4,
      },
    ]

    render(
      <PreviewPanel
        originalText="Only one chunk here"
        chunks={singleChunk}
        chunkOverlap={5}
      />
    )

    // No overlap should be shown for single chunk
    expect(screen.queryByTestId(/overlap-/)).not.toBeInTheDocument()
    expect(screen.getByText('Only one chunk here')).toBeInTheDocument()
  })

  it('scrolls to chunk when ref is provided', () => {
    const scrollRef = vi.fn()

    render(
      <PreviewPanel
        originalText={mockOriginalText}
        chunks={mockChunks}
        chunkOverlap={0}
        selectedChunkIndex={1}
        onChunkRef={scrollRef}
      />
    )

    // Verify ref callback was called for selected chunk
    expect(scrollRef).toHaveBeenCalled()
  })

  it('highlights selected chunk', () => {
    render(
      <PreviewPanel
        originalText={mockOriginalText}
        chunks={mockChunks}
        chunkOverlap={0}
        selectedChunkIndex={1}
      />
    )

    const selectedChunk = screen.getByTestId('chunk-1')
    expect(selectedChunk).toHaveClass('ring-2')
  })

  it('applies correct styling to overlap region', () => {
    const chunksWithOverlap = [
      {
        chunk_index: 0,
        content: 'ABCDEFGHIJ',
        start_position: 0,
        end_position: 10,
        char_count: 10,
        word_count: 1,
      },
      {
        chunk_index: 1,
        content: 'GHIJKLMNOP',
        start_position: 6,
        end_position: 16,
        char_count: 10,
        word_count: 1,
      },
    ]

    render(
      <PreviewPanel
        originalText="ABCDEFGHIJKLMNOP"
        chunks={chunksWithOverlap}
        chunkOverlap={4}
      />
    )

    // Check for overlap region with title tooltip
    const overlapRegion = screen.getByTestId('overlap-0-1')
    expect(overlapRegion).toHaveAttribute('title', 'Overlap between chunk 1 and 2')
  })
})

describe('PreviewPanel - Edge Cases', () => {
  it('handles chunks with -1 positions (not found in text)', () => {
    const chunksWithMissing = [
      {
        chunk_index: 0,
        content: 'Found content',
        start_position: 0,
        end_position: 13,
        char_count: 13,
        word_count: 2,
      },
      {
        chunk_index: 1,
        content: 'Not in original',
        start_position: -1,
        end_position: -1,
        char_count: 15,
        word_count: 3,
      },
    ]

    render(
      <PreviewPanel
        originalText="Found content plus more"
        chunks={chunksWithMissing}
        chunkOverlap={0}
      />
    )

    // Should still render both chunks
    expect(screen.getByText('Found content')).toBeInTheDocument()
    expect(screen.getByText('Not in original')).toBeInTheDocument()

    // Chunk with -1 position should have warning styling
    const missingChunk = screen.getByTestId('chunk-1')
    expect(missingChunk).toHaveClass('opacity-50')
  })

  it('preserves whitespace in chunk content', () => {
    const chunksWithWhitespace = [
      {
        chunk_index: 0,
        content: '  indented  content  ',
        start_position: 0,
        end_position: 21,
        char_count: 21,
        word_count: 2,
      },
    ]

    render(
      <PreviewPanel
        originalText="  indented  content  "
        chunks={chunksWithWhitespace}
        chunkOverlap={0}
      />
    )

    // Should preserve whitespace
    const panel = screen.getByTestId('preview-panel')
    expect(panel).toHaveClass('whitespace-pre-wrap')
  })

  it('handles very long text with wrapping', () => {
    const longText = 'A'.repeat(10000)
    const longChunks = [
      {
        chunk_index: 0,
        content: longText,
        start_position: 0,
        end_position: 10000,
        char_count: 10000,
        word_count: 1,
      },
    ]

    render(
      <PreviewPanel
        originalText={longText}
        chunks={longChunks}
        chunkOverlap={0}
      />
    )

    // Panel should have whitespace wrapping for long content
    const panel = screen.getByTestId('preview-panel')
    expect(panel).toHaveClass('whitespace-pre-wrap')
  })
})
