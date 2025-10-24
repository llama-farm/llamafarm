import React from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'

interface MarkdownProps {
  content: string
  className?: string
}

const Markdown: React.FC<MarkdownProps> = ({ content, className }) => {
  return (
    <div className={className}>
      <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default Markdown
