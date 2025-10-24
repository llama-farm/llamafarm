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
// Import schema types from generated file
import type { ExtractorSchema, SchemaField } from '@/types/ragTypes'

type Props = {
  schema: ExtractorSchema
  value: Record<string, unknown>
  onChange: (next: Record<string, unknown>) => void
  disabled?: boolean
}

export default function ExtractorSettingsForm({
  schema,
  value,
  onChange,
  disabled,
}: Props) {
  const entries = useMemo(() => Object.entries(schema.properties), [schema])

  const toSentenceCase = (s: string) => {
    const spaced = s
      .replace(/_/g, ' ')
      .replace(/\b([A-Z])/g, ' $1')
      .replace(/\s+/g, ' ')
      .trim()
    return spaced.charAt(0).toUpperCase() + spaced.slice(1)
  }

  const setField = (key: string, nextVal: unknown) => {
    const next = { ...value, [key]: nextVal }
    onChange(next)
  }

  const renderArrayOfObjects = (key: string, field: SchemaField) => {
    const arr = Array.isArray(value[key]) ? (value[key] as any[]) : []
    return (
      <div key={key} className="flex flex-col gap-2">
        <Label className="text-xs text-foreground">{toSentenceCase(key)}</Label>
        {arr.map((row, idx) => (
          <div key={idx} className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Input
              className="bg-background"
              placeholder="name"
              value={row?.name || ''}
              onChange={e => {
                const next = arr.slice()
                next[idx] = { ...row, name: e.target.value }
                setField(key, next)
              }}
              disabled={disabled}
            />
            <Input
              className="bg-background"
              placeholder="pattern"
              value={row?.pattern || ''}
              onChange={e => {
                const next = arr.slice()
                next[idx] = { ...row, pattern: e.target.value }
                setField(key, next)
              }}
              disabled={disabled}
            />
            <Input
              className="bg-background"
              placeholder="description"
              value={row?.description || ''}
              onChange={e => {
                const next = arr.slice()
                next[idx] = { ...row, description: e.target.value }
                setField(key, next)
              }}
              disabled={disabled}
            />
          </div>
        ))}
        <div>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setField(key, [
                ...arr,
                { name: '', pattern: '', description: '' },
              ])
            }
            disabled={disabled}
          >
            Add pattern
          </Button>
        </div>
        {field.description ? (
          <div className="text-xs text-muted-foreground">
            {field.description}
          </div>
        ) : null}
      </div>
    )
  }

  const renderField = (key: string, field: SchemaField) => {
    const current = value[key]
    const label = toSentenceCase(key)

    if (field.type === 'boolean') {
      const checked = Boolean(current ?? field.default ?? false)
      return (
        <div key={key} className="flex items-center gap-4 py-1">
          <Switch
            checked={checked}
            onCheckedChange={v => setField(key, v)}
            disabled={disabled}
          />
          <div className="flex flex-col">
            <Label className="text-xs text-foreground">{label}</Label>
            {field.description ? (
              <div className="text-xs text-muted-foreground -mt-0.5">
                {field.description}
              </div>
            ) : null}
          </div>
        </div>
      )
    }

    if (field.type === 'string' && field.enum && field.enum.length > 0) {
      const currentVal = (current as string) ?? (field.default as string) ?? ''
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-foreground">{label}</Label>
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

      // Check if current value is out of bounds for validation warning
      const isOutOfBounds =
        Number.isFinite(n) && (
          (typeof min === 'number' && n < min) ||
          (typeof max === 'number' && n > max)
        )

      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-foreground">{label}</Label>
          <Input
            type="number"
            className={`bg-background ${isOutOfBounds ? 'border-destructive' : ''}`}
            value={str}
            onChange={e => {
              const raw = e.target.value
              if (raw.trim() === '') {
                setField(key, undefined)
                return
              }
              const num = Number(raw)
              if (!Number.isFinite(num)) return
              // Coerce integers to whole numbers but don't clamp to bounds yet
              const coerced = field.type === 'integer' ? Math.trunc(num) : num
              setField(key, coerced)
            }}
            onBlur={e => {
              // Only validate and clamp on blur, not while typing
              const raw = e.target.value
              if (raw.trim() === '') return
              const num = Number(raw)
              if (!Number.isFinite(num)) return
              const coerced = field.type === 'integer' ? Math.trunc(num) : num
              let bounded = coerced
              if (typeof min === 'number') bounded = Math.max(min, bounded)
              if (typeof max === 'number') bounded = Math.min(max, bounded)
              if (bounded !== coerced) {
                setField(key, bounded)
              }
            }}
            onFocus={e => {
              // Select all text on focus so user can easily replace the value
              e.target.select()
            }}
            min={min}
            max={max}
            placeholder={String(field.default ?? '')}
            disabled={disabled}
          />
          <div className={`text-xs ${isOutOfBounds ? 'text-destructive' : 'text-muted-foreground'}`}>
            {isOutOfBounds ? '⚠️ ' : ''}
            {field.description}
            {typeof min === 'number' ? ` (min ${min})` : ''}
            {typeof max === 'number' ? ` (max ${max})` : ''}
          </div>
        </div>
      )
    }

    if (field.type === 'array' && field.items?.type === 'string') {
      const arr = Array.isArray(current) ? (current as string[]) : []
      const text = arr.join(', ')
      return (
        <div key={key} className="flex flex-col gap-1">
          <Label className="text-xs text-foreground">{label}</Label>
          <Input
            className="bg-background"
            value={text}
            onChange={e => {
              const items = e.target.value
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

    if (field.type === 'array' && (field.items as any)?.type === 'object') {
      return renderArrayOfObjects(key, field)
    }

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
      {(() => {
        const bools = entries.filter(([, f]) => f.type === 'boolean')
        const others = entries.filter(([, f]) => f.type !== 'boolean')
        return (
          <>
            {bools.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {bools.map(([k, f]) => renderField(k, f))}
              </div>
            ) : null}
            {others.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {others.map(([k, f]) => (
                  <div
                    key={k}
                    className={f.type === 'array' ? 'md:col-span-2' : ''}
                  >
                    {renderField(k, f)}
                  </div>
                ))}
              </div>
            ) : null}
          </>
        )
      })()}
    </div>
  )
}
