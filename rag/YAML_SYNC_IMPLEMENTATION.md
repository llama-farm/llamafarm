# YAML Config Sync Implementation

## Problem

Parsers and extractors were only being saved to localStorage, NOT to the actual `llamafarm.yaml` configuration file. This meant:
- ❌ Changes disappeared on server restart
- ❌ Config file didn't reflect UI changes
- ❌ Datasets couldn't find parsers/extractors after deletion

## Solution

Added smart YAML sync functions that update the configuration via the API whenever parsers/extractors are modified.

## Implementation

### File: `designer/src/components/Rag/StrategyView.tsx`

#### 1. Conversion Functions

**`parserRowToYaml(row: ParserRow)`**
Converts UI parser row to YAML-compliant format:

```typescript
{
  type: "PDFParser_PyPDF2",
  config: { chunk_size: 1000, ... },
  file_include_patterns: ["*.pdf", "*.PDF"],
  priority: 100
}
```

**Key transformations:**
- `row.name` → `type` (parser type identifier)
- `row.include` (comma-separated string) → `file_include_patterns` (array)
- `row.config` → `config` (as-is)
- `row.priority` → `priority` (as-is)
- Excludes: `row.exclude` (not in schema)

**`extractorRowToYaml(row: ExtractorRow)`**
Converts UI extractor row to YAML-compliant format:

```typescript
{
  type: "KeywordExtractor",
  config: { algorithm: "rake", max_keywords: 10, ... },
  priority: 50,
  file_include_patterns: ["*.pdf"]  // optional
}
```

**Key transformations:**
- `row.name` → `type` (extractor type identifier)
- `row.applyTo` (comma-separated string) → `file_include_patterns` (array, optional)
- `row.config` → `config` (as-is)
- `row.priority` → `priority` (as-is)

#### 2. Sync Functions

**`syncParsersToConfig(rows: ParserRow[])`**

Smart sync function that:
1. ✅ Checks for active project (fails gracefully if none)
2. ✅ Gets current config from API
3. ✅ Finds the strategy by name in `rag.data_processing_strategies[]`
4. ✅ Converts all parser rows to YAML format
5. ✅ Updates only the `parsers` array for that strategy
6. ✅ Preserves all other config (databases, other strategies, etc.)
7. ✅ Calls `updateProjectMutation` to save via API
8. ✅ Shows toast notification on success/failure

**`syncExtractorsToConfig(rows: ExtractorRow[])`**

Same as above but for extractors.

#### 3. Updated Save Functions

**`saveParsers(rows: ParserRow[])`**

Now does **two things**:
1. Saves to localStorage immediately (instant UI feedback)
2. Syncs to YAML config asynchronously (persistent)

```typescript
const saveParsers = (rows: ParserRow[]) => {
  try {
    // Immediate: localStorage for UI
    localStorage.setItem(storageKeys.parsers, JSON.stringify(rows))

    // Persistent: YAML config via API
    syncParsersToConfig(rows).catch(console.error)
  } catch {}
}
```

**`saveExtractors(rows: ExtractorRow[])`**

Same pattern as `saveParsers`.

## Schema Compliance

The conversion functions ensure schema compliance by:

### Required Fields (Always Present)
- ✅ `type` - Parser/extractor type identifier
- ✅ `config` - Configuration object (empty `{}` if none)
- ✅ `priority` - Numeric priority (defaults to 50)

### Optional Fields (When Present)
- ✅ `file_include_patterns` - Array of glob patterns
- ⚠️  Omits if empty (schema allows undefined)

### Fields NOT Included (Not in Schema)
- ❌ `exclude` patterns - UI-only, not in schema
- ❌ `summary` - UI display only
- ❌ `id` - UI identifier only

## YAML Structure

Changes are written to this structure in `llamafarm.yaml`:

```yaml
rag:
  databases:
    - name: main_database
      # ... database config ...

  data_processing_strategies:
    - name: universal_processor
      description: "Unified processor for multiple file types"
      parsers:
        - type: PDFParser_PyPDF2
          config:
            chunk_size: 1000
            chunk_overlap: 100
            extract_metadata: true
          file_include_patterns:
            - "*.pdf"
            - "*.PDF"
          priority: 100
        - type: TextParser_Python
          config:
            chunk_size: 1000
            encoding: "utf-8"
          file_include_patterns:
            - "*.txt"
          priority: 90
      extractors:
        - type: KeywordExtractor
          config:
            algorithm: "rake"
            max_keywords: 10
          priority: 50
```

## User Experience

### Before (Broken)
1. User deletes a parser in UI
2. Parser removed from localStorage ✅
3. **Config file unchanged** ❌
4. Server restart → parser reappears 😞
5. Datasets fail to process (missing parser) 💥

### After (Fixed)
1. User deletes a parser in UI
2. Parser removed from localStorage ✅
3. **Config file updated via API** ✅
4. Toast notification confirms save ✅
5. Server restart → parser stays deleted 🎉
6. Datasets process correctly ✅

## Error Handling

### Graceful Failures
- ❌ No active project → Warn in console, don't crash
- ❌ Config not found → Warn in console, don't crash
- ❌ Strategy not found → Warn in console, don't crash
- ❌ API error → Show destructive toast, log error

### Success Feedback
- ✅ Sync succeeds → Toast: "Parsers saved to config"
- ✅ Sync succeeds → Toast: "Extractors saved to config"

### No Blocking
- ⚡ localStorage saves synchronously (instant UI update)
- ⚡ YAML sync runs asynchronously (doesn't block UI)
- ⚡ Errors are caught and logged (UI continues working)

## Testing Checklist

To verify the fix works:

```bash
# 1. Start designer
cd designer && npm run dev

# 2. Navigate to Data page → Configure universal_processor

# 3. Test parser deletion
- Delete a parser
- Check browser console for "Parsers saved to config" toast
- Check llamafarm.yaml file - parser should be gone
- Restart server - parser should stay deleted

# 4. Test parser addition
- Add a new parser
- Verify toast notification
- Check llamafarm.yaml - new parser should be present
- Restart server - parser should persist

# 5. Test extractor changes
- Add/remove extractors
- Verify toast notifications
- Check llamafarm.yaml - changes should persist

# 6. Test error handling
- Disconnect from server
- Try to save changes
- Should see "Failed to save parsers" toast
- UI should still work (localStorage updates)
```

## Files Modified

- ✅ `designer/src/components/Rag/StrategyView.tsx`
  - Added `parserRowToYaml()` converter
  - Added `extractorRowToYaml()` converter
  - Added `syncParsersToConfig()` sync function
  - Added `syncExtractorsToConfig()` sync function
  - Updated `saveParsers()` to call sync
  - Updated `saveExtractors()` to call sync

## Performance

- ⚡ **No UI blocking** - Sync happens asynchronously
- ⚡ **Instant feedback** - localStorage updates immediately
- ⚡ **Batched API calls** - Only one request per save action
- ⚡ **No polling** - Direct mutation, React Query handles cache

## Future Enhancements

Possible improvements:
- [ ] Debounce rapid changes (wait 1s before syncing)
- [ ] Show loading indicator during sync
- [ ] Optimistic updates with rollback on error
- [ ] Validate config against schema before saving
- [ ] Add "Revert" button to undo changes
- [ ] Show diff of changes in toast notification
