# Overwatch PWA — Updated Spec (Post-AFSOC Call)

## What Changed
Original spec was built around Dani (Sensor Operator) managing 3 drones from fleet view. After the AFSOC sync with Cody (720th STG), we're refocusing:

- **1:1 — one drone to UI** (paired drone support later, same mission)
- **Voice is the primary interface** — screen is backup
- **Map opens first** — no fleet selection step
- **Video feed is secondary** — small PIP, not a primary view
- **MGRS grid coordinates** on everything
- **Lower contrast throughout** — no bright whites or saturated colors. Night-adapted eyes.
- **Voice conversation view** — reviewable log with easy map swap

## Tech stack
- React + TypeScript + Vite (existing)
- Leaflet with CartoDB dark_all tiles (existing)
- Tailwind CSS (existing)
- Web Speech API for voice input/output (NEW)
- All mock data for now (existing)

---

## Theme Update — Lower Contrast

Update THEME.md values. The current palette is good but needs to go softer:

```
Background (main):     #0d1b2a (keep)
Background (cards):    #1b2838 (keep)
Background (elevated): #243447 (keep)
Text primary:          #b8c4d0 (was #e0e6ed — softer, less white)
Text secondary:        #6b7a8a (was #8899aa — dimmer)
Borders:               #1e2d3d (was #2a3a4a — subtler)
Accent/interactive:    #3d7cc7 (was #4a9eff — less electric, more muted blue)

Status green:          #1a9e4a (was #22c55e — darker, less neon)
Status amber:          #c4820a (was #f59e0b — warmer, less bright)
Status red:            #c43434 (was #ef4444 — deeper, less glaring)

Kill button:           #b91c1c (was #dc2626 — darker red, still distinct)

Detection pin red:     #994444 (muted)
Detection pin yellow:  #997744 (muted)
Detection pin green:   #449966 (muted)
```

Rule: **Nothing pure white. Nothing neon. Everything looks like it belongs on a dimmed command center screen at 2am.**

---

## Screen Flow (Single Drone)

### Screen 1: OPEN → Map (default)

App opens directly to full-screen dark map. No fleet selection. The one active drone is shown on the map, status "Ready."

- Drone icon (arrow showing heading) on map
- Top bar: connection status (subtle dot, not a badge), time
- Telemetry HUD: altitude, battery %, speed — small, bottom-left, low opacity until needed
- Voice bar at bottom: mic button + status text
- Kill button: bottom-right, always visible, muted red until tapped
- No fleet strip yet (one drone — strip appears if/when paired drone joins)

### Screen 2: DEPLOY → Draw or Voice

**Touch:** Tap draw button → draw rectangle on map → "Launch to search area? [Confirm]"
**Voice:** "Launch search, altitude 400, lawnmower" → "Copy, launching search pattern" (spoken + displayed in voice bar)

On confirm: drone icon begins moving in search pattern. Pattern overlay shown as subtle dashed line on map.

3 interactions max. No configuration.

### Screen 3: MONITOR → Map + Detections + Voice

Primary operating screen. Operator spends most time here.

- Detection pins appear on map as drone flies (color-coded, muted colors)
- Each pin: type icon + confidence on hover/tap
- Voice callouts: "New contact — vehicle — 82% — grid 12S AB 12345 67890" (spoken via synthesis)
- Voice bar shows last callout text
- Detection feed: right-side collapsible panel (existing), shows latest 5 detections with MGRS grid
- Video PIP: small thumbnail, top-right corner, ~120x90px. Tap to expand slightly (not full screen). Shows live feed with bounding box overlay.

### Screen 4: DETAIL → Tap Detection Pin

Tap a pin or voice "detail on last contact":
- Modal card: classification, confidence %, MGRS grid, timestamp, mock photo
- "Flag to Commander" button
- "Watch Feed" → expands PIP slightly + centers map on that detection
- Tap outside or swipe down to dismiss → back to map

### Screen 5: VOICE CONVERSATION → Swipe or Tap

**New view.** Accessible by tapping the voice bar or swiping up from it.

- Full-screen scrollable log of the voice conversation:
  - Operator commands (right-aligned, subtle blue bubble)
  - Drone/Ace responses (left-aligned, muted card)
  - Detection callouts (left-aligned with type icon + color dot)
  - Timestamps on each entry
- Map PIP in corner (tap to go back to map instantly)
- Voice bar still active at bottom — can speak commands from this view
- Swipe down or tap map PIP to return to map

This gives the operator a way to review what the drone has reported without scrolling through map pins. Like a chat transcript of the mission.

### Screen 6: BATTERY SWAP

When drone battery hits 18%:
- Earpiece: "Battery low — 18%. Backup ready."
- Subtle amber banner at top (not a full overlay): "Bird 1 low — Send Bird 2? [Confirm]"
- On confirm: Bird 2 launches to same area, Bird 1 returns
- Detection pins persist — no gap

### Screen 7: COMMS LOSS (NEW)

On connection drop:
- Earpiece warning tone (two short beeps)
- Top bar connection dot goes red
- Subtle but unmissable banner: "COMMS LOST — Drone autonomous" (muted red background, not blinding)
- Stays until reconnect
- On reconnect: "COMMS RESTORED" banner (green, fades after 5s) + gap summary in voice conversation log

### Screen 8: EMERGENCY KILL

Kill button always visible, bottom-right, every screen.
- Tap → confirm slider (existing pattern)
- Voice: "Kill" → "Confirm kill?" → "Confirm" → "Drone stopped, hovering"
- After confirm: "Stopped — Hovering" overlay with "Return to Launch" button
- Audible: "Bird 1 stopped. Hovering."

---

## Voice System Design

### Input (Web Speech API — SpeechRecognition)
- Continuous listening mode when mic is active
- Push-to-talk as alternative (tap and hold mic button)
- Commands parsed for intent:
  - "Launch [pattern] [altitude] [grid]" → deploy
  - "Kill" / "Kill all" / "Stop" → emergency
  - "Status" → reads telemetry
  - "Any new contacts?" → summarizes recent detections
  - "Detail on last contact" → opens detection detail
  - "Flag to commander" → flags current/last detection
  - "Grid on that" → reads MGRS of last detection
  - "Return home" → RTH command

### Output (Web Speech API — SpeechSynthesis)
- Drone acknowledgments: "Copy," "Roger," "Returning"
- Detection callouts: "[Type] detected — [confidence]% — grid [MGRS]"
- Status responses: "Battery 78%. Altitude 400 feet. Flying search pattern."
- Warnings: "Battery low," "Comms lost," "Comms restored"
- Voice should be: flat, calm, slightly robotic. Not conversational. Military radio cadence.

### Voice Bar UI
```
┌─────────────────────────────────────────────┐
│ 🎤  "Vehicle detected — 82% — grid 12S..." │
│     ↑ tap for voice log    [MIC]            │
└─────────────────────────────────────────────┘
```
- Left: last message text (scrolls if long)
- Right: mic button (muted blue when listening, dim when off)
- Tap the text area → opens voice conversation view
- Swipe up → opens voice conversation view

---

## MGRS Grid Integration

- All detection detail cards show MGRS grid (not just lat/lng)
- Tap any point on map → MGRS grid tooltip
- Voice responses use MGRS format
- Use a JS library like `mgrs` (npm) for conversion
- Format: "12S AB 12345 67890" (spoken as individual digits)

---

## Component Changes

### Remove / Demote
- `FleetStatus` as opening view → map opens first. FleetStatus becomes accessible from a menu or long-press.
- `FeedView` as full screen → PIP only. Remove the full feed view mode.
- `ViewMode` type: change from `'fleet' | 'map' | 'feed'` to `'map' | 'voice'` (fleet and feed are overlays/PIPs, not views)

### Add
- `VoiceBar` — persistent bottom bar with mic + last message
- `VoiceLog` — full conversation view (new "screen")
- `CommsLostBanner` — connection loss overlay
- `MGRSTooltip` — map tap → grid display
- `useVoice` hook — Web Speech API integration (recognition + synthesis)
- `useMGRS` hook — lat/lng → MGRS conversion

### Modify
- `App.tsx` — default to map view, remove fleet as initial, add voice view mode
- `TopBar` — simplify, lower contrast, connection dot instead of badge
- `DetectionDetail` — add MGRS grid field
- `DetectionFeed` — add MGRS to each item
- `TelemetryHUD` — lower opacity, smaller, bottom-left
- `KillButton` / `FleetStrip` — muted red (not bright), darker until needed
- All colors → updated lower-contrast palette

---

## Implementation Order

1. **Theme update** — apply lower-contrast palette everywhere
2. **Default to map** — remove fleet as opening screen
3. **Voice bar UI** — add persistent bar (visual only first, no speech API yet)
4. **Voice conversation view** — scrollable log + map PIP to swap back
5. **MGRS integration** — add to detection details and map taps
6. **Web Speech API** — wire voice input + synthesis output
7. **Comms loss banner** — connection state warning
8. **Simplify to 1:1** — remove multi-drone fleet selection, single drone focus
9. **Video PIP only** — remove full feed view, keep small corner PIP

---

## Files to Update

```
overwatch/
├── SPEC.md              ← this file (replace)
├── THEME.md             ← update color values
├── src/
│   ├── App.tsx          ← default to map, add voice view
│   ├── types/index.ts   ← update ViewMode, add VoiceEntry type
│   ├── hooks/
│   │   ├── useVoice.ts  ← NEW: Web Speech API
│   │   └── useMGRS.ts   ← NEW: coordinate conversion
│   ├── components/
│   │   ├── Voice/
│   │   │   ├── VoiceBar.tsx     ← NEW
│   │   │   └── VoiceLog.tsx     ← NEW
│   │   ├── Map/
│   │   │   ├── MapView.tsx      ← add MGRS tooltip
│   │   │   └── TelemetryHUD.tsx ← lower opacity
│   │   ├── Detection/
│   │   │   ├── DetectionDetail.tsx ← add MGRS
│   │   │   └── DetectionFeed.tsx  ← add MGRS, muted colors
│   │   ├── Controls/
│   │   │   └── KillButton.tsx   ← muted red
│   │   ├── TopBar.tsx           ← simplify, add comms dot
│   │   └── CommsLostBanner.tsx  ← NEW
│   └── index.css        ← update color vars
└── tailwind.config.js   ← update palette
```
