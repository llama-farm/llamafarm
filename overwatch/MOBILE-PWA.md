# IMPORTANT: Mobile-First PWA Requirements

This app MUST be mobile-first for Android phone dimensions. Dani uses this on her Android phone in the field.

## Mobile-first
- Design for 390x844 (iPhone 14 / Pixel 7 size) as the PRIMARY viewport
- All touch targets min 48px (not 44px — Android Material guidelines)
- Single-column layout — no sidebars on mobile
- Fleet strip scrolls horizontally if needed
- Map controls sized for thumb reach (bottom half of screen)
- Kill button: bottom-right, 64px, always visible, thumb-reachable

## PWA manifest
- Add `public/manifest.json` with app name "Overwatch", theme_color dark, display: standalone
- Add PWA meta tags to index.html
- Add a service worker (basic offline shell cache)
- App icon placeholder (simple radar/eye icon in SVG)

## Viewport
- `<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">`
- Prevent pinch zoom (operator shouldn't accidentally zoom the app, only the map)
- Safe area insets for notch phones: `env(safe-area-inset-top)` etc.

## Touch interactions
- Map: pinch to zoom, drag to pan (Leaflet handles this)
- Draw search area: long-press to start drawing
- Swipe up on fleet strip to expand fleet detail
- All buttons: visible active/pressed states for feedback
