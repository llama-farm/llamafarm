import * as React from 'react'
import { Button as ShadcnButton } from './ui/button'
import { cn } from '../lib/utils'

type LegacyVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type LegacySize = 'sm' | 'md' | 'lg'

function mapVariant(variant?: LegacyVariant) {
  if (variant === 'secondary') return 'secondary'
  if (variant === 'ghost') return 'ghost'
  if (variant === 'danger') return 'destructive'
  return 'default'
}

function mapSize(size?: LegacySize) {
  if (size === 'sm') return 'sm'
  if (size === 'lg') return 'lg'
  return 'default'
}

export type AppButtonProps = React.ComponentProps<'button'> & {
  variant?: LegacyVariant
  size?: LegacySize
  legacy?: boolean
  className?: string
}

export function Button({
  legacy,
  className,
  variant,
  size,
  ...props
}: AppButtonProps) {
  // If you need to keep an old implementation around during rollout:
  // if (legacy) return <OldButton {...props} />
  return (
    <ShadcnButton
      variant={mapVariant(variant)}
      size={mapSize(size)}
      className={cn(className)}
      {...props}
    />
  )
}

export default Button
