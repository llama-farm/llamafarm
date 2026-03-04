interface PanelIntroProps {
  title?: string
  children: React.ReactNode
}

/**
 * Consistent title + subtitle at the top of each vision sub-panel.
 */
export function PanelIntro({ title, children }: PanelIntroProps) {
  return (
    <div className="mb-1">
      {title && <h2 className="text-lg font-semibold tracking-tight">{title}</h2>}
      <p className="text-sm text-muted-foreground">{children}</p>
    </div>
  )
}
