/**
 * Simple lat/lng to MGRS-like grid reference.
 * For demo/mock purposes — produces realistic-looking grid strings.
 * Replace with proper `mgrs` npm package when npm cache is fixed.
 */

function toMGRS(lat: number, lng: number): string {
  // Simplified UTM zone calculation
  const zone = Math.floor((lng + 180) / 6) + 1
  
  // Grid zone designator (C-X, excluding I and O)
  const letters = 'CDEFGHJKLMNPQRSTUVWX'
  const letterIdx = Math.min(Math.max(Math.floor((lat + 80) / 8), 0), letters.length - 1)
  const gridLetter = letters[letterIdx]
  
  // 100km square ID (simplified)
  const colLetters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
  const rowLetters = 'ABCDEFGHJKLMNPQRSTUV'
  const col = colLetters[Math.floor(Math.abs(lng * 10)) % colLetters.length]
  const row = rowLetters[Math.floor(Math.abs(lat * 10)) % rowLetters.length]
  
  // Easting and Northing (5-digit, 1m precision)
  const easting = String(Math.floor(Math.abs((lng % 1) * 100000))).padStart(5, '0')
  const northing = String(Math.floor(Math.abs((lat % 1) * 100000))).padStart(5, '0')
  
  return `${zone}${gridLetter} ${col}${row} ${easting} ${northing}`
}

export function useMGRS() {
  return {
    toMGRS,
    formatMGRS: (lat: number, lng: number) => toMGRS(lat, lng),
    speakMGRS: (lat: number, lng: number) => {
      const grid = toMGRS(lat, lng)
      // For voice: spell out each character/digit
      return grid.split('').map(c => {
        if (c === ' ') return ', '
        return c
      }).join('')
    }
  }
}
