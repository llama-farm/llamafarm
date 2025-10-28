import React from 'react'

interface MarkdownProps {
  content: string
  className?: string
}

const Markdown: React.FC<MarkdownProps> = ({ content, className }) => {
  // Minimal fallback renderer without external deps
  return (
    <div
      className={className}
      style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
    >
      {content}
    </div>
  )
}

export default Markdown
