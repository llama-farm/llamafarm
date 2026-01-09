# Frontend Review Checklist

Domain-specific review items for React, TypeScript, Tailwind CSS, and React Query codebases.

Add these categories to the summary table:
- React Architecture
- State Management
- TypeScript Quality
- React Query
- Tailwind CSS
- Performance (React)

---

## Category: Security (Frontend)

### Dangerous HTML Rendering (XSS)

**Search patterns**:
```bash
# Find dangerouslySetInnerHTML usage
grep -r "dangerouslySetInnerHTML" --include="*.tsx" --include="*.jsx"

# Find innerHTML assignments
grep -r "\.innerHTML\s*=" --include="*.ts" --include="*.tsx" --include="*.js"
```

**Pass criteria**: No usage, OR all usage properly sanitizes input with DOMPurify or similar
**Severity**: Critical

---

### Unsanitized URL Parameters

**Search patterns**:
```bash
# Direct use of URL params
grep -rE "(window\.location|useSearchParams|URLSearchParams)" --include="*.tsx" --include="*.ts"

# Template literals in href/src
grep -rE "(href|src)=\{`" --include="*.tsx"
```

**Pass criteria**: All URL parameters validated before use
**Severity**: High

---

### Sensitive Data in localStorage

**Search patterns**:
```bash
grep -rE "localStorage\.(setItem|getItem).*['\"]?(token|auth|password|secret|key|credential)" --include="*.ts" --include="*.tsx"
```

**Pass criteria**: No auth tokens or secrets in localStorage (use httpOnly cookies)
**Severity**: High

---

## Category: React Architecture

### God Components (Oversized)

**Search technique**:
```bash
# Find large component files (>300 lines)
find . -name "*.tsx" -exec wc -l {} \; | awk '$1 > 300 {print}'

# Then review each for:
# - Multiple unrelated responsibilities
# - Many useState calls (>5)
# - Mixed concerns (fetching + logic + UI)
```

**Pass criteria**: No components over 300 lines with mixed concerns
**Severity**: Medium

---

### Array Index as Key

**Search patterns**:
```bash
grep -rE "key=\{(index|i|idx)\}" --include="*.tsx" --include="*.jsx"
grep -rE "\.map\([^)]*,\s*(index|i|idx)\)" --include="*.tsx" --include="*.jsx"
```

**Pass criteria**: No index-based keys in dynamic lists
**Severity**: Medium

---

### Missing Keys in Lists

**Search technique**:
```bash
# Find map calls, then check if key prop exists
grep -rn "\.map(" --include="*.tsx" --include="*.jsx"
# Review each to ensure returned elements have key props
```

**Pass criteria**: All mapped elements have stable, unique keys
**Severity**: Medium

---

### Prop Drilling (3+ Levels)

**Search technique**:
- Identify components passing props through without using them
- Look for same prop name appearing in multiple nested component signatures
- Search for props passed unchanged through intermediary components

**Pass criteria**: Props not passed through more than 2 component levels
**Severity**: Low

---

### Spreading Unknown Props

**Search patterns**:
```bash
grep -rE "\{\.\.\.props\}|\{\.\.\.rest\}" --include="*.tsx"
```

**Review**: Verify spread is intentional and typed, not spreading onto DOM elements
**Pass criteria**: No untyped prop spreading onto DOM elements
**Severity**: Medium

---

## Category: State Management

### State Declared as Variables

**Search patterns**:
```bash
# Variables in component body that should be state
grep -rE "^\s+(let|var)\s+\w+\s*=" --include="*.tsx" -A 5 | grep -v "const"
```

**Pass criteria**: Persistent values use useState or useRef
**Severity**: High

---

### Direct State Mutation

**Search patterns**:
```bash
# Array mutations
grep -rE "\.(push|pop|shift|unshift|splice|sort|reverse)\(" --include="*.tsx" --include="*.ts"

# Object property assignment on state
grep -rE "state\.\w+\s*=" --include="*.tsx"
```

**Pass criteria**: All state updates create new references
**Severity**: High

---

### Props Copied to State

**Search patterns**:
```bash
grep -rE "useState\(props\." --include="*.tsx"
grep -rE "useState\(\{.*props\." --include="*.tsx"
```

**Pass criteria**: No copying props to state (except for intentional "initial value" patterns)
**Severity**: Medium

---

### Derived State Stored

**Search technique**:
- Look for useState that computes from other state
- Search for "total", "count", "sum", "filtered", "sorted" state variables
- Check if these could be computed with useMemo instead

**Pass criteria**: Computed values derived in render or useMemo, not stored
**Severity**: Medium

---

### useEffect Missing Dependencies

**Search patterns**:
```bash
# Empty dependency arrays that reference outer scope
grep -rE "useEffect\([^)]+,\s*\[\s*\]\)" --include="*.tsx" -A 10
```

**Review**: Check if effect body references variables not in deps array
**Pass criteria**: All referenced variables in dependency array
**Severity**: High

---

### Data Fetching in useEffect

**Search patterns**:
```bash
grep -rE "useEffect.*fetch\(|useEffect.*axios\.|useEffect.*\.get\(" --include="*.tsx"
```

**Pass criteria**: Data fetching uses React Query, not raw useEffect
**Severity**: Medium

---

## Category: TypeScript Quality

### Any Type Usage

**Search patterns**:
```bash
grep -rE ":\s*any\b|<any>|as any" --include="*.ts" --include="*.tsx"
```

**Pass criteria**: No `any` types (or each is justified with comment)
**Severity**: High

---

### Type Assertions (as Type)

**Search patterns**:
```bash
grep -rE "\bas\s+\w+" --include="*.ts" --include="*.tsx" | grep -v "as const"
```

**Review**: Each assertion should be necessary and safe
**Pass criteria**: Minimal assertions, none that bypass type checking unsafely
**Severity**: Medium

---

### Missing Component Prop Types

**Search patterns**:
```bash
# Functions without typed props
grep -rE "function\s+\w+\s*\(\s*props\s*\)" --include="*.tsx"
grep -rE "const\s+\w+\s*=\s*\(\s*props\s*\)" --include="*.tsx"

# Arrow functions without type annotation
grep -rE "=>\s*\{" --include="*.tsx" -B 2 | grep -v "Props"
```

**Pass criteria**: All components have explicit prop interfaces
**Severity**: Medium

---

### Event Handler Types

**Search patterns**:
```bash
grep -rE "\(e\)|\(event\)|\(e:\s*any\)" --include="*.tsx"
```

**Pass criteria**: Event handlers use proper React event types
**Severity**: Low

---

### Untyped API Responses

**Search patterns**:
```bash
grep -rE "\.then\(\s*\(?\s*data\s*\)?" --include="*.ts" --include="*.tsx"
grep -rE "await fetch" --include="*.ts" --include="*.tsx"
```

**Pass criteria**: All API responses have defined types
**Severity**: High

---

### ts-ignore and ts-nocheck

**Search patterns**:
```bash
grep -rE "@ts-ignore|@ts-nocheck|@ts-expect-error" --include="*.ts" --include="*.tsx"
```

**Pass criteria**: No suppressions, or each has justifying comment
**Severity**: High

---

## Category: React Query (TanStack Query)

### Missing QueryClientProvider

**Search technique**:
- Check app root/entry point for QueryClientProvider
- Verify provider wraps the entire application

**Pass criteria**: QueryClientProvider exists at app root
**Severity**: Critical (if using React Query)

---

### Query Keys Missing Dynamic Parameters

**Search patterns**:
```bash
grep -rE "queryKey:\s*\[['\"]" --include="*.ts" --include="*.tsx" -A 5
```

**Review**: Check if queryFn uses variables not in queryKey
**Pass criteria**: All dynamic parameters in queryKey
**Severity**: High

---

### Query Data Copied to Redux/Context

**Search patterns**:
```bash
grep -rE "dispatch\(.*data\)" --include="*.tsx"
grep -rE "useEffect.*set.*\(.*data" --include="*.tsx"
```

**Pass criteria**: Query data used directly, not synced to other state
**Severity**: Medium

---

### Missing Enabled Flag for Conditional Queries

**Search patterns**:
```bash
grep -rE "useQuery\(" --include="*.tsx" -A 10 | grep -v "enabled"
```

**Review**: Check if any queries depend on undefined values
**Pass criteria**: Conditional queries use `enabled` flag
**Severity**: Medium

---

### Missing Query Invalidation After Mutations

**Search patterns**:
```bash
grep -rE "useMutation\(" --include="*.tsx" -A 15
```

**Review**: Check for onSuccess/onSettled with queryClient.invalidateQueries
**Pass criteria**: Mutations invalidate related queries
**Severity**: High

---

## Category: Tailwind CSS

### Arbitrary Values (Magic Numbers)

**Search patterns**:
```bash
grep -rE "\[([\d]+px|[\d]+rem|#[a-fA-F0-9]+)\]" --include="*.tsx" --include="*.jsx"
```

**Examples to flag**: `w-[347px]`, `text-[13px]`, `bg-[#ff5733]`
**Pass criteria**: Uses design tokens from config, not arbitrary values
**Severity**: Medium

---

### Missing Responsive Variants

**Search technique**:
- Check layout components for responsive breakpoints
- Look for fixed widths without `sm:`, `md:`, `lg:` variants

**Pass criteria**: Key layouts have responsive variants
**Severity**: Medium

---

### Missing Focus States

**Search patterns**:
```bash
grep -rE "<button|<a\s|onClick" --include="*.tsx" | grep -v "focus:"
```

**Pass criteria**: Interactive elements have focus indicators
**Severity**: High (accessibility)

---

### Important Overuse

**Search patterns**:
```bash
grep -rE "!\w+-" --include="*.tsx"
```

**Examples**: `!mt-4`, `!text-red-500`
**Pass criteria**: Minimal or no `!important` usage
**Severity**: Low

---

### Conflicting Classes

**Search patterns**:
```bash
# Look for contradictory utilities in same className
grep -rE "flex.*block|block.*flex|hidden.*visible|mt-\d+.*mt-\d+" --include="*.tsx"
```

**Pass criteria**: No conflicting utility classes
**Severity**: Low

---

### Inline Styles Mixed with Tailwind

**Search patterns**:
```bash
grep -rE "style=\{" --include="*.tsx" | grep "className"
```

**Pass criteria**: Consistent approach - either Tailwind or inline, not mixed
**Severity**: Low

---

## Category: Performance (React)

### Missing useMemo for Expensive Computations

**Search technique**:
- Find `.filter()`, `.map()`, `.reduce()`, `.sort()` in render
- Check if results are memoized

**Pass criteria**: Expensive computations memoized
**Severity**: Medium

---

### Missing useCallback for Prop Functions

**Search patterns**:
```bash
grep -rE "on\w+=\{\s*\(" --include="*.tsx"
```

**Review**: Check if inline functions are passed to memoized children
**Pass criteria**: Callback props use useCallback when appropriate
**Severity**: Low

---

### Large Lists Without Virtualization

**Search technique**:
- Find `.map()` calls rendering lists
- Check array sizes (if static) or data source sizes

**Pass criteria**: Lists >100 items use virtualization
**Severity**: Medium

---

### Missing Error Boundaries

**Search patterns**:
```bash
grep -rE "componentDidCatch|ErrorBoundary" --include="*.tsx"
```

**Pass criteria**: Error boundaries exist for critical sections
**Severity**: Medium

---

### Missing Loading States

**Search technique**:
- Review components that fetch data
- Check for loading indicators

**Search patterns**:
```bash
grep -rE "isLoading|isPending|loading" --include="*.tsx"
```

**Pass criteria**: Async operations show loading state
**Severity**: Medium

---

### Missing Error States

**Search patterns**:
```bash
grep -rE "isError|error\s*\?" --include="*.tsx"
```

**Pass criteria**: Error conditions are handled and displayed
**Severity**: Medium
