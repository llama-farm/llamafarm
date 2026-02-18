interface PanelIntroProps {
  children: React.ReactNode
}

/**
 * Subtle intro / guidance text at the top of each vision sub-panel.
 * Uses the same text-sm text-muted-foreground pattern as other Designer panels.
 */
export function PanelIntro({ children }: PanelIntroProps) {
  return (
    <p className="text-sm text-muted-foreground mb-4">
      {children}
    </p>
  )
}
