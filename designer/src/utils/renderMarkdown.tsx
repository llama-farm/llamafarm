import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'

interface MarkdownProps {
  content: string
  className?: string
}

const Markdown: React.FC<MarkdownProps> = ({ content, className }) => {
  return (
    <ReactMarkdown
      className={className}
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize]}
      components={{
        // Style code blocks
        code: ({ node, className, children, ...props }: any) => {
          const inline = !className?.includes('language-')
          return inline ? (
            <code
              className="px-1.5 py-0.5 rounded bg-muted text-foreground font-mono text-[13px]"
              {...props}
            >
              {children}
            </code>
          ) : (
            <code
              className="block p-3 my-2 rounded bg-muted text-foreground font-mono text-sm overflow-x-auto"
              {...props}
            >
              {children}
            </code>
          )
        },
        pre: ({ node, children, ...props }) => (
          <pre className="my-2" {...props}>
            {children}
          </pre>
        ),
        // Style links
        a: ({ node, children, ...props }) => (
          <a
            className="text-primary hover:underline"
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        ),
        // Style lists
        ul: ({ node, children, ...props }) => (
          <ul className="list-disc list-outside ml-4 space-y-1" {...props}>
            {children}
          </ul>
        ),
        ol: ({ node, children, ...props }) => (
          <ol className="list-decimal list-outside ml-4 space-y-1" {...props}>
            {children}
          </ol>
        ),
        li: ({ node, children, ...props }) => (
          <li className="ml-0" {...props}>
            {children}
          </li>
        ),
        // Style blockquotes
        blockquote: ({ node, children, ...props }) => (
          <blockquote
            className="border-l-4 border-muted-foreground/30 pl-4 italic text-muted-foreground"
            {...props}
          >
            {children}
          </blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

export default Markdown
