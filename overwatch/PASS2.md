# Pass 2: Mobile-First PWA + Full Story Flow

Read SPEC.md and MOBILE-PWA.md for full context. The app skeleton is built. This pass needs to:

## 1. Mobile-First Overhaul
- All layouts must work at 390x844 (Android phone) as PRIMARY viewport
- Touch targets min 48px
- Single-column, no sidebars — fleet strip is horizontal scroll at bottom
- Kill button: 64px, bottom-right, thumb-reachable, always visible on EVERY screen
- Add `public/manifest.json`: name "Overwatch", theme_color "#0a0a0a", display: standalone, orientation: portrait
- Add PWA meta tags to index.html
- Viewport: `<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">`
- Safe area insets for notch phones

## 2. Verify ALL 8 Screens of Dani's Story Actually Work

The app must tell this COMPLETE story as a walkable flow:

**Screen 1: OPEN → Fleet Status**
App opens to fleet cards. Bird 1 (green, 94%), Bird 2 (green, 87%), Bird 3 (yellow, 42%). Tap a drone to select and go to map.

**Screen 2: DEPLOY → Map + Draw**
Full-screen dark map centered on Fort Liberty NC (35.14°N, 79.0°W). Selected drone icon shown. User can draw a rectangle search area. After drawing: launch confirmation overlay "Launch Bird 1 → Search Area? [Confirm]". On confirm, drone icon starts moving in a search pattern (simulated with setInterval).

**Screen 3: MONITOR → Map with Detection Pins**
As drone "flies," detection pins appear over time (use setTimeout to stagger them). Colors: red=person, yellow=vehicle, green=animal. Fleet strip persistent at bottom showing all drones with live battery %.

**Screen 4: ALERT → Detection Needs Attention**
When a red (person) pin drops, it PULSES (CSS animation). Toast notification slides in from top: "Person detected — 87% — tap to view". High contrast, impossible to miss.

**Screen 5: VERIFY → Detection Detail + Live Feed (PIP swap)**
Tap detection pin → detail card: mock photo (use a colored placeholder rectangle with detection type icon), classification, confidence %, timestamp, GPS coords. "Watch Feed" button. Tapping it swaps to feed view — mock video area (dark with scan lines + "LIVE" badge + drone label "BIRD 1"). Small PIP map thumbnail in corner — tap to swap back (QGC pattern).

**Screen 6: ESCALATE → Flag to Commander**
On feed view: "Flag to Commander" button. Tap → "Flagged to Marco ✓" toast. Return to map — flagged pin now has a small flag/star indicator.

**Screen 7: MANAGE → Battery Swap**
After ~30 seconds, Bird 1 battery in fleet strip drops to yellow (18%), then red (12%). Alert overlay: "Bird 1 — Low Battery — Send Bird 2?" Tapping "Send Bird 2" shows Bird 2 icon launching. Bird 1 returns. Detection pins persist.

**Screen 8: EMERGENCY → Kill Button**
Kill button always visible. Tap → confirmation slider (drag to confirm, QGC pattern). After confirm: "Bird [X] Stopped — Hovering" overlay. "Return to Launch" follow-up button.

## 3. Make the Simulation Auto-Play
After selecting a drone and launching, the simulation should auto-progress through the story:
- t+0s: Drone launches, starts flying search pattern
- t+5s: First detection (green, animal)
- t+10s: Second detection (yellow, vehicle)  
- t+15s: Third detection (red, person — this one PULSES + triggers alert toast)
- t+30s: Bird 1 battery drops to 18%, triggers swap prompt
- Kill button available at any time

## 4. Visual Polish
- Leaflet dark tiles: use CartoDB dark_all (`https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png`)
- Drone icon: simple triangle/arrow showing heading
- Detection pins: circles with type icon inside
- Pulsing animation on critical detections: `@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.7) } 70% { box-shadow: 0 0 0 20px rgba(239,68,68,0) } }`
- Feed view mock: dark background with subtle scan lines, "LIVE" badge red, bounding box rectangle overlay
- Transitions between views should be smooth (fade or slide)

## Important
- Test that EVERY screen transition works
- The story flow must be completable from open to emergency stop
- Mobile dimensions are the priority — test at 390x844
