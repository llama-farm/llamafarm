import React, { useState, useRef, useEffect } from 'react'
import FontIcon from '../../common/FontIcon'
import type { TOCNode as TOCNodeType } from '../../types/config-toc'

interface TOCNodeProps {
  node: TOCNodeType
  onNavigate: (node: TOCNodeType) => void
  defaultCollapsed?: boolean
  isActive?: boolean
  activeNodeId?: string | null
}

/**
 * Individual tree node component for the config table of contents
 * Supports collapse/expand and nested children
 */
const TOCNode: React.FC<TOCNodeProps> = ({ 
  node, 
  onNavigate, 
  defaultCollapsed = false,
  isActive = false,
  activeNodeId = null
}) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed)
  const buttonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (isActive && buttonRef.current) {
      buttonRef.current.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
        behavior: 'auto',
      })
    }
  }, [isActive])
  const hasChildren = node.children && node.children.length > 0

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()

    if (hasChildren && node.isCollapsible) {
      if (isActive) {
        // Already active: toggle collapse/expand
        setIsCollapsed(prev => !prev)
      } else {
        // Not active yet: navigate and ensure expanded
        setIsCollapsed(false)
      }
    }

    onNavigate(node)
  }

  const toggleCollapse = () => {
    setIsCollapsed(prev => !prev)
  }

  const handleCaretClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    // Only toggle collapse, don't trigger navigation
    toggleCollapse()
  }

  const handleCaretKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      e.stopPropagation()
      toggleCollapse()
    }
  }

  const indent = node.level * 12 // 12px per level

  return (
    <div className="toc-node">
      <button
        ref={buttonRef}
        onClick={handleClick}
        className={`w-full flex items-center gap-2 py-2 px-3 text-sm text-left transition-colors border-l-2 ${
          isActive 
            ? 'bg-slate-200 text-slate-600 border-slate-400 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-500' 
            : 'hover:bg-accent hover:text-accent-foreground border-transparent'
        } ${node.level === 0 ? 'font-medium' : ''}`}
        style={{ paddingLeft: `${indent + 12}px` }}
        title={`Jump to ${node.label}`}
      >
        {/* Collapse/expand caret - only for collapsible items */}
        {hasChildren && node.isCollapsible ? (
          <span
            onClick={handleCaretClick}
            onKeyDown={handleCaretKeyDown}
            className="flex-shrink-0 cursor-pointer hover:text-primary flex items-center justify-center w-4 h-4"
            aria-label={isCollapsed ? 'Expand' : 'Collapse'}
            role="button"
            tabIndex={0}
            aria-expanded={!isCollapsed}
          >
            <FontIcon
              type="chevron-down"
              className={`w-3 h-3 transition-transform ${isCollapsed ? '-rotate-90' : ''}`}
            />
          </span>
        ) : (
          <span className="w-4" />
        )}

        {/* Label */}
        <span className="flex-1 truncate">{node.label}</span>
      </button>

      {/* Render children if not collapsed */}
      {hasChildren && !isCollapsed && (
        <div className="toc-children">
          {node.children!.map((child) => (
            <TOCNode
              key={child.id}
              node={child}
              onNavigate={onNavigate}
              defaultCollapsed={child.level === 1 && child.isCollapsible}
              isActive={activeNodeId === child.id}
              activeNodeId={activeNodeId}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default TOCNode

