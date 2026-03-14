# Pass 3: Header Consolidation, Voice Input UX, Color Tuning

## Problem
1. **Two header bars in Voice Log view** — TopBar (44px) + VoiceLog header (44px) = 88px wasted on a mobile screen. In the primary voice interaction view, this is unacceptable.
2. **Header blue too bright** — `surface-bar` (#152232) reads as a distinct bright bar against `surface-base` (#0d1b2a). For night ops / light discipline, the header should blend into the background, not stand out.
3. **Voice input missing text fallback** — Operators need to type if voice isn't working (noisy environment, mic failure). No text input exists.
4. **No stop button for mic** — Once listening, the mic icon should change to a stop button (familiar square icon) so the operator knows how to end recording.

## Changes

### 1. Merge headers in Voice Log view
- Remove the VoiceLog's own header bar entirely
- Add "Mission Log" title + Map button into TopBar when `viewMode === 'voice'`
- TopBar becomes the single persistent header across all views
- Saves 44px of vertical space in voice view

### 2. Darken header bar
- Change `surface-bar` from `#152232` → `#0f1922` (closer to base `#0d1b2a`)
- Header should be barely distinguishable from background — just enough border to separate

### 3. Add text input to VoiceBar
- Replace the tap-to-open-log text with an actual `<input>` field
- Placeholder: "Type or tap mic..."
- Enter key submits as a command (same as voice)
- Input sits left of mic button, takes remaining space

### 4. Mic → Stop button toggle
- **Mic off:** Show microphone icon (current)
- **Mic on:** Show stop icon (filled square) with accent ring
- Stop icon is universally understood as "stop recording"
- Color: accent when listening, text-secondary when idle

### 5. Tighten Detection Feed + soften collapse toggle
- Collapse button: remove the floating tab that sticks out to the left. Replace with a subtle chevron inside the panel header, same bg as panel — shouldn't draw the eye.
- Tighten the feed: reduce padding on feed items (py-2 → py-1.5), shrink header padding, reduce panel width from w-64 → w-56.
- "Live Feed" header text should be text-secondary not white — nothing in this panel should be bright.

### 6. Detection Feed visible in Voice Log view
- Feed panel should overlay on right side in voice mode too (same as map view)
- Detections don't stop coming because you're talking — operator needs to see new contacts
- Feed stays collapsible in both views

## Files Modified
- `tailwind.config.js` — darken `surface.bar`
- `src/components/TopBar.tsx` — accept viewMode prop, show Mission Log + Map button in voice mode
- `src/components/Voice/VoiceLog.tsx` — remove own header bar
- `src/components/Voice/VoiceBar.tsx` — add text input, mic→stop icon toggle
- `src/App.tsx` — pass viewMode to TopBar, wire text command handler
