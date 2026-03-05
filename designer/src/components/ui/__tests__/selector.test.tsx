import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Selector, SelectorOption } from '../selector'

const options: SelectorOption[] = [
  { value: 'a', label: 'Alpha' },
  { value: 'b', label: 'Beta' },
  { value: 'c', label: 'Gamma', description: 'Third letter' },
]

describe('Selector', () => {
  it('renders with placeholder when no value selected', () => {
    render(<Selector value={null} options={options} onChange={() => {}} placeholder="Pick one" />)
    expect(screen.getByText('Pick one')).toBeInTheDocument()
  })

  it('renders selected option label', () => {
    render(<Selector value="b" options={options} onChange={() => {}} />)
    expect(screen.getByText('Beta')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<Selector value={null} options={options} onChange={() => {}} label="Choose" />)
    expect(screen.getByText('Choose')).toBeInTheDocument()
  })

  it('calls onChange when option clicked', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Selector value="a" options={options} onChange={onChange} />)

    // Open dropdown
    await user.click(screen.getByRole('button'))
    // Click option
    await user.click(screen.getByText('Beta'))

    expect(onChange).toHaveBeenCalledWith('b')
  })

  it('shows empty message when no options', async () => {
    const user = userEvent.setup()
    render(<Selector value={null} options={[]} onChange={() => {}} emptyMessage="Nothing here" />)

    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
  })

  it('disables button when disabled prop is true', () => {
    render(<Selector value={null} options={options} onChange={() => {}} disabled />)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('disables button when loading', () => {
    render(<Selector value={null} options={options} onChange={() => {}} loading />)
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('shows option description when provided', async () => {
    const user = userEvent.setup()
    render(<Selector value={null} options={options} onChange={() => {}} />)

    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Third letter')).toBeInTheDocument()
  })

  // size="sm" should apply text-xs classes
  it('applies sm size classes to trigger and items', async () => {
    const user = userEvent.setup()
    const { container } = render(
      <Selector value="a" options={options} onChange={() => {}} size="sm" />
    )

    const button = screen.getByRole('button')
    // sm trigger should have h-7, px-2, text-xs from sizeClasses
    expect(button.className).toContain('h-7')
    expect(button.className).toContain('px-2')

    // Inner text span should use text-xs (from textClasses.sm)
    const textSpan = button.querySelector('span.truncate')
    expect(textSpan?.className).toContain('text-xs')
    expect(textSpan?.className).not.toContain('text-sm')

    // Open dropdown and check item text size
    await user.click(button)
    const itemSpans = container.querySelectorAll('[role="menuitem"] span')
    for (const span of itemSpans) {
      if (span.textContent && options.some(o => o.label === span.textContent)) {
        expect(span.className).toContain('text-xs')
      }
    }
  })

  // default size should use text-sm
  it('applies default size classes', () => {
    render(<Selector value="a" options={options} onChange={() => {}} />)
    const button = screen.getByRole('button')
    expect(button.className).toContain('h-9')
    expect(button.className).toContain('px-3')

    const textSpan = button.querySelector('span.truncate')
    expect(textSpan?.className).toContain('text-sm')
  })

  // variant="minimal" should not have border classes
  it('applies minimal variant classes', () => {
    render(<Selector value="a" options={options} onChange={() => {}} variant="minimal" />)
    const button = screen.getByRole('button')
    expect(button.className).toContain('bg-transparent')
    expect(button.className).toContain('border-0')
    expect(button.className).not.toContain('rounded-lg')
  })

  // default variant should have border
  it('applies default variant classes', () => {
    render(<Selector value="a" options={options} onChange={() => {}} />)
    const button = screen.getByRole('button')
    expect(button.className).toContain('rounded-lg')
    expect(button.className).toContain('border-input')
  })

  // focus-visible styling should be present on both variants
  it('has focus-visible ring on default variant', () => {
    render(<Selector value="a" options={options} onChange={() => {}} />)
    const button = screen.getByRole('button')
    expect(button.className).toContain('focus-visible:ring-2')
  })

  it('has focus-visible ring on minimal variant', () => {
    render(<Selector value="a" options={options} onChange={() => {}} variant="minimal" />)
    const button = screen.getByRole('button')
    expect(button.className).toContain('focus-visible:ring-2')
  })
})
