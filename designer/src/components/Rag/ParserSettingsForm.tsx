import { useMemo } from 'react'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Switch } from '../ui/switch'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { Button } from '../ui/button'
import type { ParserSchema, SchemaField } from './parserSchemas'

type Props = {
  schema: ParserSchema
  value: Record<string, unknown>
  onChange: (next: Record<string, unknown>) => void
  disabled?: boolean
}

export default function ParserSettingsForm({
  schema,
  value,
  onChange,
  disabled,
}: Props) {
  const entries = useMemo(() => Object.entries(schema.properties), [schema])

  const setField = (key: string, nextVal: unknown) => {
    const next = { ...value, [key]: nextVal }
    onChange(next)
  }

  const renderField = (key: string, field: SchemaField) => {
    const current = value[key]
    const label = key.replace(/_/g, ' ')

    if (field.type === 'boolean') {
      const checked = Boolean(current ?? field.default ?? false)
      return (
        <div key={key} className="flex items-center justify-between py-2">
          <div className="flex flex-col">
            <Label className="text-xs text-muted-foreground">{label}</Label>
            {field.description ? (
              <div className="text-xs text-muted-foreground">
                {field.description}
              </div>
            ) : null}
          </div>
          <Switch
            checked={checked}
            onCheckedChange={v => setField(key, v)}
            disabled={disabled}
          />
        </div>
      )
    }

    if (field.type === 'string' && field.enum && field.enum.length > 0) {
      const currentVal = (current as string) ?? (field.default as string) ?? ''
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-muted-foreground">{label}</Label>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="justify-between"
                disabled={disabled}
              >
                {currentVal || 'Select'}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {field.enum.map(opt => (
                <DropdownMenuItem key={opt} onClick={() => setField(key, opt)}>
                  {opt}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          {field.description ? (
            <div className="text-xs text-muted-foreground">
              {field.description}
            </div>
          ) : null}
        </div>
      )
    }

    if (field.type === 'integer' || field.type === 'number') {
      const n =
        typeof current === 'number' ? current : (field.default as number)
      const str = Number.isFinite(n) ? String(n) : ''
      const min = typeof field.minimum === 'number' ? field.minimum : undefined
      const max = typeof field.maximum === 'number' ? field.maximum : undefined
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-muted-foreground">{label}</Label>
          <Input
            type="number"
            className="bg-background"
            value={str}
            onChange={e => {
              const raw = e.target.value
              if (raw.trim() === '') {
                setField(key, undefined)
                return
              }
              const num = Number(raw)
              if (!Number.isFinite(num)) return
              // Coerce integers to whole numbers
              const coerced = field.type === 'integer' ? Math.trunc(num) : num
              let bounded = coerced
              if (typeof min === 'number') bounded = Math.max(min, bounded)
              if (typeof max === 'number') bounded = Math.min(max, bounded)
              setField(key, bounded)
            }}
            min={min}
            max={max}
            placeholder={String(field.default ?? '')}
            disabled={disabled}
          />
          <div className="text-xs text-muted-foreground">
            {field.description}
            {typeof min === 'number' ? ` (min ${min})` : ''}
            {typeof max === 'number' ? ` (max ${max})` : ''}
          </div>
        </div>
      )
    }

    if (field.type === 'array') {
      const nullable = field.nullable === true
      const arr = Array.isArray(current) ? (current as unknown[]) : []
      const text = arr.join(', ')
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-muted-foreground">{label}</Label>
          <Input
            className="bg-background"
            value={text}
            onChange={e => {
              const raw = e.target.value
              if (nullable && raw.trim() === '') {
                setField(key, null)
                return
              }
              const items = raw
                .split(',')
                .map(s => s.trim())
                .filter(s => s.length > 0)
              setField(key, items)
            }}
            placeholder={
              Array.isArray(field.default)
                ? (field.default as unknown[]).join(', ')
                : ''
            }
            disabled={disabled}
          />
          {field.description ? (
            <div className="text-xs text-muted-foreground">
              {field.description}
            </div>
          ) : null}
        </div>
      )
    }

    // simple string input
    if (field.type === 'string') {
      const currentVal =
        typeof current === 'string'
          ? current
          : ((field.default as string) ?? '')
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-muted-foreground">{label}</Label>
          <Input
            className="bg-background"
            value={currentVal}
            onChange={e => setField(key, e.target.value)}
            placeholder={typeof field.default === 'string' ? field.default : ''}
            disabled={disabled}
          />
          {field.description ? (
            <div className="text-xs text-muted-foreground">
              {field.description}
            </div>
          ) : null}
        </div>
      )
    }

    return null
  }

  return (
    <div className="flex flex-col gap-3">
      {entries.map(([k, f]) => renderField(k, f))}
    </div>
  )
}
