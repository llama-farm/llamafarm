# Overwatch PWA — Crawl Phase Spec

## What this is
Operator dashboard for autonomous drone ISR system. PWA that runs on laptop (Mac/Linux). Primary user: Dani (Sensor Operator) — launches drones, monitors detections, manages fleet from a touch-friendly web interface.

## Tech stack
- React + TypeScript + Vite
- Leaflet or Mapbox GL JS for maps (prefer Leaflet — no API key needed)
- Tailwind CSS for styling
- No backend — all mock data for now. Use realistic mock data that tells Dani's story.

## Design rules (from benchmarking research)
- **Dark theme only** — field/night ops
- **Map-maximized** (ATAK pattern) — map fills the screen, minimal chrome
- **Large touch targets** — min 44px, designed for gloved/stressed use
- **High-contrast status colors** — green=good, amber=warning, red=critical
- **Text ≥14px, icons ≥24px** — readable at arm's length
- **Minimal UI chrome** — every pixel is map, video, or data

## Dani's flow — 8 screens in one app

The app tells this story as Dani uses it:

### 1. Fleet Status (app opens here)
- Shows 3 drone cards: Bird 1 (green, 94%), Bird 2 (green, 87%), Bird 3 (yellow, 42%)
- Each card: name, battery bar, status indicator (green/yellow/red), connection status
- Tap a drone card to select it and transition to map view
- Pattern: DJI FlightHub sidebar cards

### 2. Map View + Draw to Deploy
- Full-screen dark map (Leaflet with dark tiles — CartoDB dark_all or similar)
- Selected drone shown on map as an icon
- Draw tool: click-to-draw a rectangle search area on the map
- After drawing: "Launch Bird 1 → Search Area? [Confirm]" overlay
- On confirm: drone icon starts moving along a search pattern (simulated)
- Pattern: ATAK map-maximized + QGC simplicity

### 3. Map with Detection Pins (monitoring)
- As drone "flies," detection pins appear on the map over time (simulated with timers)
- Pin colors: 🔴 red = person, 🟡 yellow = vehicle, 🟢 green = animal
- Pins show small icon indicating type
- Fleet status strip at bottom: persistent bar showing all drones with battery levels
- Pattern: ATAK symbology + DJI fleet strip

### 4. Alert — Detection Needs Attention
- When a red (person) pin drops, it pulses/animates
- Toast notification slides in: "Person detected — 87% confidence — tap to view"
- Notification is high-contrast, impossible to miss

### 5. Detection Detail → Live Feed (PIP swap)
- Tap a detection pin → detail card pops up: mock photo, classification, confidence %, timestamp, coordinates
- "Watch Feed" button on the card
- Tapping "Watch Feed" swaps to live feed view (mock video/placeholder with drone label)
- Detection bounding box overlay shown on the feed
- PIP: small map thumbnail in corner (QGC pattern — tap to swap back)
- Pattern: QGC PIP video switcher

### 6. Flag to Commander
- On the feed view: "Flag to Commander" button
- Tapping it shows a brief confirmation: "Flagged to Marco ✓"
- Returns to map view — the flagged pin now has a small flag icon

### 7. Battery Swap / Handoff
- After some time, Bird 1's battery in the fleet strip drops to yellow (18%), then red (12%)
- Alert: "Bird 1 — Low Battery — Send Bird 2?"
- Tapping "Send Bird 2" shows Bird 2 launching to the same area
- Bird 1 returns (icon moves back toward origin)
- Detection pins persist — no gap in coverage
- Pattern: DJI fleet management

### 8. Emergency Stop
- **Kill button: always visible**, bottom-right corner, red, every screen
- Tapping it: "STOP Bird [X] — Confirm?" with a confirmation slider (QGC pattern)
- After confirm: "Bird [X] Stopped — Hovering" status
- "Return to Launch" button appears as follow-up action

## Layout structure
```
┌─────────────────────────────────────────────┐
│ Top bar: connection status, GPS, time       │
├─────────────────────────────────────────────┤
│                                             │
│              MAP / FEED VIEW                │
│          (full screen, swappable)           │
│                                             │
│  [PIP thumbnail in corner when in feed]     │
│                                             │
├─────────────────────────────────────────────┤
│ Fleet strip: [Bird1 94%] [Bird2 87%] [B3]  │
└───────────────────────────────────[🔴 KILL]─┘
```

## File structure
```
overwatch/
  package.json
  vite.config.ts
  tsconfig.json
  index.html
  tailwind.config.js
  postcss.config.js
  src/
    main.tsx
    App.tsx
    index.css (tailwind imports)
    components/
      Map/MapView.tsx        — Leaflet dark map with draw, pins, drone icons
      Feed/FeedView.tsx      — Mock video feed with overlays
      Fleet/FleetStrip.tsx   — Bottom bar with drone cards
      Fleet/DroneCard.tsx    — Individual drone status card
      Detection/DetectionPin.tsx — Map pin component
      Detection/DetectionDetail.tsx — Detail card popup
      Detection/AlertToast.tsx — Toast notification
      Controls/KillButton.tsx — Emergency stop (always visible)
      Controls/ConfirmSlider.tsx — QGC-style confirmation slider
      Controls/LaunchConfirm.tsx — Launch confirmation overlay
      TopBar.tsx             — Status bar
      PIPThumbnail.tsx       — Picture-in-picture swap thumbnail
    hooks/
      useSimulation.ts       — Mock drone flight + detection generation
      useDrones.ts           — Drone state management
    types/
      index.ts               — TypeScript types
    data/
      mockDetections.ts      — Realistic mock detection data
      mockDrones.ts          — Drone fleet mock data
```

## Mock data should feel real
- Drone names: Bird 1, Bird 2, Bird 3
- Map centered on a realistic area (use Fort Liberty, NC area — 35.14°N, 79.0°W)
- Detections: mix of persons, vehicles, animals with varying confidence (0.65-0.95)
- Timestamps: realistic intervals
- Battery drain: Bird 1 starts at 94% and slowly decreases

## What NOT to build
- No backend/API integration
- No real video streaming (use a placeholder/mock frame)
- No authentication
- No Marco or Jay screens (Crawl is Dani only)
- No multi-drone handoff logic (just visual simulation)
- No DINOv2 re-ID
- No Ace chat

## Run instructions
Should start with `npm install && npm run dev` and open on localhost.
